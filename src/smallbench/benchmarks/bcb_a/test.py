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
from smallbench.benchmarks.bcb_a.bench import BCB_AgentBenchmark


def get_contexts_extremely_hacky_please_fix():
    benchmark = BigCodeBenchComplete_Benchmark()
    question = benchmark.train[0]
    backend = BCBEngine(question)
    aci = BCBAgentComputerInterface(backend)
    contexts = {
        "premise": "You are a software engineer",
        "setting_information": "You are working to solve a computer science problem. You will need to submit a solution to the problem, which will be tested against a suite of hidden unit tests.",
        "actions": {
            action.action_name: {
                "action_context": action.action_context,
                "action_arg_spec": action.action_arg_spec,
                "action_description": action.action_description,
            }
            for action in aci.actions
        },
        "objective": "Please complete the problem by drafting a solution, creating unit tests, improving the solution, and submitting the solution.",
        "constraints": "You will be given a code_prompt_for_answer, which contains imports and the function signature. Your solution must comprise code that can be appended to code_prompt_for_answer and run as a single script.",
    }
    return contexts


if __name__ == "__main__":
    from apropos import LLM
    from smallbench.baselines.agents.react import SimpleReActLanguageAgent

    benchmark = BigCodeBenchComplete_Benchmark()
    contexts = get_contexts_extremely_hacky_please_fix()
    agent = SimpleReActLanguageAgent(lm=LLM("gpt-4o-mini"), contexts=contexts)
    agent_benchmark = BCB_AgentBenchmark()
    agent_performance = asyncio.run(
        agent_benchmark.score_agent(
            agent, split="test", indices=[0, 1, 2, 3, 4], verbose=False
        )
    )
    print("Score: " + str(agent_performance))
