import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Type, Tuple
import copy
from pydantic import BaseModel

from apropos.src.bench.bigcodebench.backends.docker import execute_code_remotely_docker
from apropos.src.bench.bigcodebench.backends.modal import execute_code_remotely_modal
from apropos.src.bench.bigcodebench.main import (
    BigCodeBench_Question,
    BigCodeBenchComplete_Benchmark,
)

from smallbench.benchmarks.bcb_a.aci import BCBEngine, BCBAgentComputerInterface
from smallbench.benchmarks.core import AgentBenchmark
from smallbench.benchmarks.bcb_a.aci import BCBAgentComputerInterface
from smallbench.baselines.agents.core import Agent

import ast
import random
import matplotlib.pyplot as plt
import sys


class BCB_AgentBenchmark(AgentBenchmark):
    def __init__(self, backend: Literal["docker", "modal"] = "docker"):
        self.backend = backend
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
        self,
        agent: Type[Agent],
        aci: BCBAgentComputerInterface,
        verbose: bool = False,
        max_agent_steps: int = 10,
    ) -> Tuple[bool, str]:
        if verbose:
            print("Spinning up agent...")
        observation = self.get_observation(None, aci)
        agent.add_observation(observation)
        import time

        for _ in range(max_agent_steps):
            action, action_args = await agent.act()
            time.sleep(1)
            if verbose:
                print(f"- Action: {action}")
                if action in ["test_submission", "submit_solution"]:
                    print("Spinning up container...")
            # result = await aci.accept_delta(action, action_args)
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
        dollars = None
        if hasattr(agent, "cost_monitor"):
            dollars, tokens = agent.cost_monitor.final_cost()
            # print(f"Cost: ${dollars} for {tokens} tokens")
        return aci.final_success, aci.final_submission, dollars

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
            backend = BCBEngine(question, self.backend)
            aci = BCBAgentComputerInterface(backend)
            try:
                success, submission, dollars = await self.evaluate(agent, aci, verbose)
                return success, dollars
            except Exception as e:
                print("\033[91mError: " + str(e)[0:300] + "....\033[0m")
                return False, 0

        successes_with_dollars = await asyncio.gather(
            *[evaluate_question(q) for q in questions]
        )
        successes = [s for s, d in successes_with_dollars]
        dollars = sum([d for s, d in successes_with_dollars])
        return sum(successes) / len(successes), dollars
