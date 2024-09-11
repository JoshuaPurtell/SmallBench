# SmallBench

Small, simple agent task environments for training and evaluation.

Designed to challenge a broad spectrum of lm-agent abilities.

<p align="middle">
  <img src="https://raw.githubusercontent.com/JoshuaPurtell/SmallBench/main/assets/data_science_small.gif" width="200" />
</p>

## Spinning Up - Use

```
uv add smallbench
```

or 

```
pip install smallbench
```

## Spinning Up - Dev

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

## Scores

### BigCodeBench - Agent Harness (ReAct)
| LM | Number Correct | Success Rate | Sample Size | Avg. Cost Per Run |
| --- | --- | --- | --- | --- |
| gpt-4o-2024-08-06 | 17 | 17% | 100 | $0.057 | 
| gpt-4o-mini-2024-07-18-ft-09-08* | 16 | 16% | 100 | $0.006 |
| deepseek-v2.5 | 12 | 12% | 100 | $0.0029 |
| gpt-4o-mini-2024-07-18 | 12 | 12% | 100 | $0.003 |
| gemini-1.5-flash-latest | 6 | 06% | 100 | $0.0018 | 

* fine-tuned on a minimal subset (500k tokens) of trajectories using a variation of the Filtered Behavioral Cloning approach.

Animation credits: [ZZ](https://x.com/mikezangus)
