# TODO: use a legit runtime, not python, for orchestration

from typing import Any, Dict, List

from pydantic import BaseModel

from apropos import LLM
from apropos.src.core.lms.cost import CostMonitor
from smallbench.baselines.agents.core import Agent

REACT_LOOKBACK = 5

class ReactResponse(BaseModel):
    reasoning: str
    action: str
    action_args: Dict[str, Any]


class SimpleReActLanguageAgent(Agent):
    obs_history: List[Dict]
    react_history: List[Dict]
    lm: LLM
    contexts: List[Dict]
    cost_monitor: CostMonitor
    multi_threaded: bool = False

    def __init__(self, lm: LLM, contexts: List[Dict], multi_threaded: bool = False):
        self.lm = lm
        self.cost_monitor = CostMonitor(model_name=lm.model_name)
        self.obs_history = []
        self.react_history = []
        self.contexts = contexts
        self.multi_threaded = multi_threaded

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
            action_info_snippet = "".join(
                f"<{key}>\n{value}\n</{key}>\n" for key, value in action_info.items()
            )
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

Your response: """
        return system_message, user_message

    def _process_response(self, react_step, system_message, user_message):
        self.cost_monitor.update_token_counts(
            system_message, user_message, str(react_step.dict())
        )
        self.react_history.append(react_step)
        return react_step.action, react_step.action_args

    async def act_async(self):
        system_message, user_message = self._prepare_messages()
        try:
            react_step = await self.lm.async_respond(
                system_prompt=system_message,
                user_prompt=user_message,
                response_model=ReactResponse,
            )
            if "'await'" in react_step.reasoning:
                raise Exception("ReAct agent found a coroutine error")
        except Exception as e:
            print(f"Error: {e}")
            raise e
        return self._process_response(react_step, system_message, user_message)

    def act_sync(self):
        system_message, user_message = self._prepare_messages()
        try:
            react_step = self.lm.sync_respond(
                system_prompt=system_message,
                user_prompt=user_message,
                response_model=ReactResponse,
                multi_threaded=self.multi_threaded,
            )
            
            print(react_step.action)
            print(react_step.reasoning)
            if "failed" in react_step.reasoning.lower():
                print("User message: ", user_message)
        except Exception as e:
            print(f"Error: {e}")
            raise e
        return self._process_response(react_step, system_message, user_message)

    def add_observation(self, obs: Dict):
        self.obs_history.append(obs)
