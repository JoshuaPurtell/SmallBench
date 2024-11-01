from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from zyk import LM
from dataclasses import dataclass
from smallbench.baselines.agents.core import Agent
from apropos.src.core.lms.cost import CostMonitor
from smallbench.benchmarks.bcb_a.abstractions import ActionResult
import uuid
from synth_sdk.tracing.decorators import trace_system


REACT_LOOKBACK = 5

@dataclass
class Step:
    messages: List[Dict]
    following_observation: Optional[ActionResult]

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.following_observation, str):
            self.following_observation = ActionResult(success=False, result=self.following_observation)
        if not self.following_observation:
            print("Warning: No observation")
            self.following_observation = ActionResult(success=False, result="No observation")
        return {
            "messages": self.messages,
            "observation": {
                "success": self.following_observation.success,
                "result": self.following_observation.result
            }
        }

@dataclass
class History:
    steps: List[Step]

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}

class ActionArgument(BaseModel):
    key: str
    value: str

class ReAct(BaseModel):
    reasoning: str
    action_name: str
    action_args: List[ActionArgument]

class SimpleReActLanguageAgent(Agent):
    obs_history: List[Dict]
    react_history: List[Dict]
    trajectory: List[Dict]
    lm: LM
    contexts: List[Dict]
    cost_monitor: CostMonitor
    history: History
    multi_threaded: bool = False

    def __init__(self, lm: LM, contexts: List[Dict], multi_threaded: bool = False):
        self.lm = lm
        self.cost_monitor = CostMonitor(model_name=lm.model_name)
        self.obs_history = []
        self.react_history = []
        self.trajectory = []
        self.contexts = contexts
        self.history = History(steps=[])
        self.multi_threaded = multi_threaded
        self.agent_id = str(uuid.uuid4())  # Initialize agent_id here

    def load_context(self, contexts: List[Dict]):
        self.contexts = contexts

    def render_history(self):
        react_history = [
            f"<{i} reasoning step(s) in the past>{item}</{i} reasoning step(s) in the past>"
            for i, item in enumerate(reversed(self.react_history[-REACT_LOOKBACK:]), 1)
        ]
        obs_history = [
            f"<{i} environment step(s) in the past>{item}</{i} environment step(s) in the past>"
            for i, item in enumerate(reversed(self.obs_history[-REACT_LOOKBACK:]), 1)
        ]
        return "\n".join(react_history), "\n".join(obs_history)

    def _prepare_messages(self):
        actions_snippet = ""
        for action_name, action_info in self.contexts["actions"].items():
            action_info_snippet = ""
            for key, value in action_info.items():
                action_info_snippet += f"<{key}>\n{value}\n</{key}>\n"
            actions_snippet += (
                f"<{action_name}>\n{action_info_snippet}\n</{action_name}>\n"
            )
        system_message = f"""
# Premise
{self.contexts['premise']}
Here is some information about this setting
<Setting Information>
{self.contexts['setting_information']}
</Setting Information>
<Actions Available>
{actions_snippet}
</Actions Available>
You'll be given your past actions/thoughts, along with recent raw observations from the environment
The environment one step in the past is your current environment.

# Objective
{self.contexts['objective']}

# Constraints
{self.contexts['constraints']}
"""
        react_history, obs_history = self.render_history()
        user_message = f"""
# Recent Actions / Thoughts
{react_history}
# Recent Observations
{obs_history}

Your next actions / thought: """
        return system_message, user_message

    def _process_response(self, react_step, system_message, user_message):
        self.history.steps.append(Step(messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}, {"role": "assistant", "content": str(react_step.dict())}], following_observation=None))
        self.cost_monitor.update_token_counts(
            system_message, user_message, str(react_step.dict())
        )
        self.react_history.append(react_step)
        arguments = [
            {arg.key: arg.value} for arg in react_step.action_args
        ]
        return react_step.action_name, arguments

    @trace_system(
        origin="agent",
        event_type="act",
        manage_event="create",
        increment_partition=True,
        log_vars_input={},
        log_vars_output={"react_step"},
        verbose=True,
    )
    async def act_async(self):
        system_message, user_message = self._prepare_messages()
        try:
            react_step = await self.lm.respond_async(
                system_message=system_message,
                user_message=user_message,
                response_model=ReAct,
            )
        except Exception as e:
            print(f"Error: {e}")
            raise e
        self.trajectory.append(
            {"messages": {
                "system": system_message,
                "user": user_message,
                "assistant": str(react_step.dict())
            }}
        )
        return self._process_response(react_step, system_message, user_message)

    @trace_system(
        origin="agent",
        event_type="act",
        manage_event="create",
        increment_partition=True,
        log_vars_input={},
        log_vars_output={"react_step"},
        verbose=True,
    )
    def act_sync(self):
        system_message, user_message = self._prepare_messages()
        try:
            react_step = self.lm.sync_respond(
                system_message=system_message,
                user_message=user_message,
                response_model=ReAct,
                multi_threaded=self.multi_threaded,
            )
        except Exception as e:
            print(f"Error: {e}")
            raise e
        result = self._process_response(react_step, system_message, user_message)
        return result

    def add_observation(self, obs: Dict):
        self.obs_history.append(obs)
        if self.history.steps:
            self.history.steps[-1].following_observation = obs

    async def act(self):
        return await self.act_async()
