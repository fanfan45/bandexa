# Development

This guide describes a practical local workflow for Bandexa.

## Setup

Create a virtual environment and install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```
If you maintain dev dependencies (recommended), add them to a dev extra (e.g. `.[dev]`) and install: 

```bash
pip install -e ".[dev]"
```

## Running examples

From repo root:

```bash
python examples/01_synthetic_regret.py --help
python examples/01_synthetic_regret.py # with default args

python examples/03_two_tower_synthetic.py --help
python examples/03_two_tower_synthetic.py # with default args
```

## Testing

From repo root:

```bash
pytest -q
```
