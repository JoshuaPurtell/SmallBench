# TODO: use a legit runtime, not python, for orchestration

from pydantic import BaseModel
from typing import List, Dict, Any
from apropos import LLM
from smallbench.baselines.agents.core import Agent
from apropos.src.core.lms.cost import CostMonitor

REACT_LOOKBACK = 5


class ReAct(BaseModel):
    reasoning: str
    action: str
    action_args: Dict[str, Any]


class SimpleReActLanguageAgent(Agent):
    obs_history: List[Dict]
    react_history: List[Dict]
    lm: LLM
    contexts: List[Dict]
    cost_monitor: CostMonitor

    def __init__(self, lm: LLM, contexts: List[Dict]):
        self.lm = lm
        self.cost_monitor = CostMonitor(model_name=lm.model_name)
        self.obs_history = []
        self.react_history = []
        self.contexts = contexts

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

    async def act(self):
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
        react_step = await self.lm.async_respond(
            system_prompt=system_message,
            user_prompt=user_message,
            response_model=ReAct,
        )
        self.cost_monitor.update_token_counts(
            system_message, user_message, str(react_step.dict())
        )

        self.react_history.append(react_step)
        return react_step.action, react_step.action_args

    def add_observation(self, obs: Dict):
        self.obs_history.append(obs)
