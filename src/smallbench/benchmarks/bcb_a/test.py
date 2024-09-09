import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Type
import copy
from matplotlib import backend_bases
from pydantic import BaseModel

from apropos.src.bench.bigcodebench.backends.docker import execute_code_remotely_docker_sync
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
    aci = BCBAgentComputerInterface(backend, synchronous=True)
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


async def test_gold_on_split(split: str = "train", indices: List[int] = None):
    bcb = BigCodeBenchComplete_Benchmark(mode="modal")
    if split == "train":
        questions = bcb.train
    elif split == "dev":
        questions = bcb.dev
    elif split == "test":
        questions = bcb.test
    
    questions = [questions[i] for i in indices]

    async def score_gold(question: BigCodeBench_Question):
        answer = question.information["answer"]
        correctness, result_dict, container = execute_code_remotely_docker_sync(
            question.information, answer
        )
        return correctness, result_dict

    gold_scores = await asyncio.gather(
        *[score_gold(question) for question in questions]
    )
    gold_scores = [score for score, _ in gold_scores]
    import numpy as np

    print(f"Gold scores for {split}: {gold_scores}")
    print(f"Mean gold score for {split}: {np.mean(gold_scores)}")
    print(f"Num correct for {split}: {np.sum(gold_scores)}")
    print(f"Num total for {split}: {len(gold_scores)}")

if __name__ == "__main__":
    from apropos import LLM
    from smallbench.baselines.agents.react import SimpleReActLanguageAgent
    
    #asyncio.run(test_gold_on_split(split="train", indices=[i for i in range(0,20)]))
    #print("Done with gold")
    backend = "docker"
    model = "gpt-4o-2024-08-06"
    #model = "gpt-4o-mini-2024-07-18"  # claude-3-5-sonnet-20240620
    #model = "gpt-4o-mini-2024-07-18"  # claude-3-5-sonnet-20240620


    benchmark = BigCodeBenchComplete_Benchmark()
    contexts = get_contexts_extremely_hacky_please_fix()
    agent = SimpleReActLanguageAgent(lm=LLM(model), contexts=contexts)
    agent_benchmark = BCB_AgentBenchmark(backend=backend)

    import time

    t0 = time.time()
    agent_performance, agent_cost = asyncio.run(
        agent_benchmark.score_agent_async(
            agent, split="train", indices=[i for i in range(0,2)], verbose=False
        )
    )
    # Score for gemini-1.5-flash-latest: 0.06
    # Score for gpt-4o-mini-2024-07-18: 0.08
    # Seems rather low?
    # t1 = time.time()
    # print(f"Time taken: {t1-t0} seconds")
    # print(f"Score for {model}: " + str(agent_performance))
    # print(f"Cost for {model}: " + str(agent_cost))

    print("Starting sync")
    t0 = time.time()
    agent_performance, agent_cost = agent_benchmark.score_agent_sync(
        agent, split="train", indices=[i for i in range(0,2)], verbose=False
    )
    t1 = time.time()
    print(f"Time taken: {t1-t0} seconds")
    print(f"Score for {model}: " + str(agent_performance))
    print(f"Cost for {model}: " + str(agent_cost))
    # top is gemini, bottom is DSC
