# Briscola RL - Four Players

Minimal Reinforcement Learning project for four-player Briscola with teams,
self-play, frozen policy snapshots, baseline opponents, evaluation scripts, and
a small playable web interface.

Main project folder:

```text
briscola_rl_4players/
```

## Requirements

Recommended Python version:

```bash
Python 3.11+
```

The code was tested with Python 3.12.

Core dependencies:

```text
numpy
pytest
pyyaml
```

Optional notebook dependencies:

```text
jupyter
matplotlib
pandas
```

## Environment Setup

Enter the project folder:

```bash
cd briscola_rl_4players
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install minimal dependencies:

```bash
pip install numpy pytest pyyaml
```

Optional notebook dependencies:

```bash
pip install jupyter matplotlib pandas
```

## Run Tests

From the project folder:

```bash
PYTHONPATH=. pytest
```

Expected result:

```text
16 passed
```

## Train an Agent

Run a small training job:

```bash
PYTHONPATH=. python scripts/train.py --config configs/train_pool_selfplay.yaml
```

Other available configs:

```text
configs/train_random.yaml
configs/train_latest_only.yaml
configs/train_pool_selfplay.yaml
```

## Evaluate

Run evaluation:

```bash
PYTHONPATH=. python scripts/evaluate.py --config configs/eval.yaml
```

## Web Demo

Start the local playable web interface:

```bash
PYTHONPATH=. python scripts/play_web.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Project Structure

```text
briscola_rl_4players/
  briscola/
    Core environment, cards, rules, observations, policies, features, REINFORCE.

  configs/
    YAML configs for training and evaluation.

  scripts/
    CLI entry points for training, evaluation, and web play.

  tests/
    Unit tests for rules, environment, observations, policies, and training.

  notebooks/
    Analysis notebook.

  web/
    Static frontend for the playable demo.

  experiments/
    Output folders for logs, checkpoints, plots, and evaluation artifacts.
```

## Notes

Generated files such as checkpoints, logs, JSON results, plots, caches, and
virtual environments should not be committed.

Use:

```bash
git status
```

before committing to verify that only source code, configs, tests, notebooks,
and documentation are tracked.
