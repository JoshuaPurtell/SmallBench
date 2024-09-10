import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Type, Tuple
import copy
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import math

from apropos.src.bench.bigcodebench.backends.docker import (
    execute_code_remotely_docker_sync,
)
from apropos.src.bench.bigcodebench.backends.modal import (
    execute_code_remotely_modal_sync,
    execute_code_remotely_modal_async,
)
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

    async def evaluate_async(
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

        for _ in range(max_agent_steps):
            action, action_args = await agent.act_async()

            try:
                action_synchronicity = next(
                    (a for a in aci.actions if a.action_name == action),
                    None,
                ).transform.type
                if action_synchronicity == "async":
                    result = await aci.accept_delta_async(action, action_args)
                else:
                    result = aci.accept_delta_sync(action, action_args)
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
            if "409 Client Error" in result:
                raise Exception(result)
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
        return aci.final_success, aci.final_submission, dollars

    def evaluate_sync(
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

        for _ in range(max_agent_steps):
            action, action_args = agent.act_sync()
            try:
                result = aci.accept_delta_sync(action, action_args)
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
        return aci.final_success, aci.final_submission, dollars

    async def score_agent_async(
        self,
        base_agent,
        split: Literal["train", "dev", "test"] = "test",
        indices: List[int] = None,
        verbose: bool = False,
        use_persistent_container: bool = False,
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
            backend = BCBEngine(
                question,
                self.backend,
                use_persistent_container=use_persistent_container,
            )
            aci = BCBAgentComputerInterface(backend, synchronous=False)
            try:
                success, submission, dollars = await self.evaluate_async(
                    agent, aci, verbose
                )
            except Exception as e:
                print(f"Error: {e}")
                return False, 0
            return success, dollars

        successes_with_dollars = await asyncio.gather(
            *[evaluate_question(q) for q in questions]
        )
        successes = [s for s, d in successes_with_dollars]
        print("Length of successes: ", len(successes))
        print("Successes: ", sum(successes))
        dollars = sum([d for s, d in successes_with_dollars])
        return sum(successes) / len(successes), dollars

    def score_agent_sync(
        self,
        base_agent,
        split: Literal["train", "dev", "test"] = "test",
        indices: List[int] = None,
        verbose: bool = False,
        use_persistent_container: bool = False,
    ):
        questions = self.bcb_benchmark.get_questions(
            split=split,
            n=(max(indices) + 1) if indices else 50,
            sort_type="first",
            patches=["A", "B"],
        )
        if indices:
            questions = [questions[i] for i in indices]

        def evaluate_question(question):
            agent = copy.deepcopy(base_agent)
            backend = BCBEngine(
                question,
                self.backend,
                use_persistent_container=use_persistent_container,
            )
            aci = BCBAgentComputerInterface(backend, synchronous=True)
            try:
                success, submission, dollars = self.evaluate_sync(agent, aci, verbose)
            except Exception as e:
                print(f"Error: {e}")
                return False, 0
            return success, dollars

        if len(questions) > 1:
            num_cpus = os.cpu_count()
            batch_size = max(1, num_cpus - 1)
            num_batches = math.ceil(len(questions) / batch_size)
            print("N batches: ", num_batches)
            print("N questions: ", len(questions))
            successes_with_dollars = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                for i in range(num_batches):
                    start = i * batch_size
                    end = min((i + 1) * batch_size, len(questions))
                    batch_questions = questions[start:end]
                    futures = [
                        executor.submit(evaluate_question, q) for q in batch_questions
                    ]
                    for future in as_completed(futures):
                        successes_with_dollars.append(future.result())
        else:
            successes_with_dollars = [evaluate_question(questions[0])]

        successes = [s for s, d in successes_with_dollars]
        dollars = sum([d for s, d in successes_with_dollars])
        return sum(successes) / len(successes), dollars
