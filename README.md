# SmallBench

Small, simple agent task environments for training and evaluation.

Designed to challenge a broad spectrum of lm-agent abilities.

<p align="middle">
  <img src="https://raw.githubusercontent.com/JoshuaPurtell/SmallBench/main/assets/data_science_small.gif" width="200" />
</p>

## Spinning Up
```
uv venv smallbench-dev
source smallbench-dev/bin/activate
uv sync
uv run ruff format .
```

## Easy Benchmarks

### BigCodeBench - Agent Harness

This benchmark provides a stateful environment for lm-based agents to solve coding problems from the BigCodeBench dataset. Agents are given a scratchpad (soon), a way to prepare and use unit tests, editing handlers, and a way to submit their solution.

Please see the [BigCodeBench](https://bigcode-bench.github.io) page for more information about the underlying dataset.

#### Get Started

##### Local
add GROQ_API_KEY and any other API keys supported by the [apropos-ai](https://github.com/JoshuaPurtell/Apropos) library to the .env file.
- Note: Groq, Google, and possibly other providers offer free tiers.

If you use a Docker backend, ensure you have the Docker app running. If you use Modal, please add all necessary credentials.

Then, run the test script:
```
uv run python -m src.smallbench.benchmarks.bcb_a.test
```

##### Colab
Check out the [Colab](https://drive.google.com/file/d/1bPMrS2IhWffeeWWIGAISHzktbJUsAmpX/view?usp=sharing) if you prefer to run the benchmark in the cloud.


## Medium Benchmarks
TBD

## Hard Benchmarks
TBD

## Difficult Benchmarks
TBD

## Caveats
1. This repository is still under *very* active development.
2. In particular, certain details regarding the agent computer interface contexts are very much subject to change, and there's a bit of response model instability. Let me know if you run into issues in the issues tab of the GitHub!
3. For this reason, scores will likely be artificially low until further notice. Don't take them too seriously.

## Scores - Extremely Preliminary

### BigCodeBench - Agent Harness
| LM | Success Rate (out of 100%) | Sample Size |
| --- | --- | --- |
| claude-3-5-sonnet-20240620 | ??? | 30 |
| gpt-4o-2024-08-06 | 10% | 30 |
| gpt-4o-mini-2024-07-18 | 13.3% | 30 |
