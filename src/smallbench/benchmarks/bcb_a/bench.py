import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Type
import copy
from pydantic import BaseModel

from apropos.src.bench.bigcodebench.backends.docker import execute_code_remotely_docker
from apropos.src.bench.bigcodebench.main import (
    BigCodeBench_Question,
    BigCodeBenchComplete_Benchmark,
)

from smallbench.benchmarks.bcb_a.aci import BCBEngine, BCBAgentComputerInterface
from smallbench.benchmarks.core import AgentBenchmark
from smallbench.benchmarks.bcb_a.aci import BCBAgentComputerInterface
from smallbench.baselines.agents.core import Agent


class BCB_AgentBenchmark(AgentBenchmark):
    def __init__(self):
        self.bcb_benchmark = BigCodeBenchComplete_Benchmark()

    def get_observation(self, action_result, aci: BCBAgentComputerInterface):
        return {
            "action_result": action_result,
            "environment_state": {
                "question": aci.bcb_backend.bcb_question.information["question"],
                "code_prompt_for_answer": aci.bcb_backend.bcb_question.information[
                    "eval_info"
                ]["code_prompt"],
                "unit_tests_you_have_written": aci.unit_tests,
                "current_solution": aci.current_solution.for_show(),
            },
        }

    async def evaluate(
        self, agent: Type[Agent], aci: BCBAgentComputerInterface, verbose: bool = False
    ):
        if verbose:
            print("Spinning up agent...")
        observation = self.get_observation(None, aci)
        agent.add_observation(observation)
        for i in range(10):
            action, action_args = await agent.act()
            if verbose:
                print(f"- {action}")
            try:
                result = await aci.accept_delta(action, action_args)
            except Exception as e:
                result = f"""Action failed. Information: 
<action>
{action}
</action>
<action_args>
{action_args}
</action_args>
<error>
{e}
</error>
"""
            observation = self.get_observation(result, aci)
            agent.add_observation(result)
            if aci.check_termination():
                break
        if verbose:
            print(
                "\033[92mSucceeded\033[0m"
                if aci.final_success
                else "\033[91mFailed\033[0m"
            )
        return aci.final_success

    async def score_agent(
        self,
        base_agent,
        split: Literal["train", "dev", "test"] = "test",
        indices: List[int] = None,
        verbose: bool = False,
    ):
        questions = self.bcb_benchmark.get_questions(
            split=split,
            n=(max(indices) + 1) if indices else 50,
            sort_type="first",
            patches=["A", "B"],
        )
        if indices:
            questions = [questions[i] for i in indices]

        async def evaluate_question(question):
            agent = copy.deepcopy(base_agent)
            backend = BCBEngine(question)
            aci = BCBAgentComputerInterface(backend)
            try:
                return await self.evaluate(agent, aci, verbose=verbose)
            except Exception as e:
                print("\033[91mError: " + str(e)[0:30] + "....\033[0m")
                return False

        successes = await asyncio.gather(*[evaluate_question(q) for q in questions])
        return sum(successes) / len(successes)
