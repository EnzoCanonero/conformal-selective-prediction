# Conformal Selective Prediction

This repository develops a conformal selective-prediction layer for reliable
automation. Instead of forcing a model to act on every input, it produces
prediction sets with coverage guarantees and defers uncertain cases to human
review.

The framework will first be validated on BANKING77 for automated banking-support
routing. It will then be applied to LLM outputs and stress-tested under
distribution shift to measure the trade-off between automation rate and
operational risk.

## Current status

The project is at its initial packaging milestone. The Python package uses a
`src` layout, with NumPy as the only core dependency. The first implementation
milestone will add a reusable multiclass split-conformal core based on least
ambiguous class (LAC) scores.

Planned work includes:

1. LAC nonconformity scores and finite-sample split-conformal calibration.
2. Prediction-set construction and basic set-quality metrics.
3. A synthetic IID multiclass example and tests of marginal coverage.
4. BANKING77, LLM and distribution-shift experiments in later milestones.

## Installation

Create and activate a virtual environment, then install the package in editable
mode:

```bash
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

The synthetic example will use scikit-learn, which is kept separate from the
core package:

```bash
python -m pip install -e ".[example]"
```

## Statistical scope

The intended split-conformal guarantee is finite-sample **marginal coverage**.
It relies on exchangeability of the calibration examples and future examples.
The guarantee does not automatically imply conditional coverage for every
subgroup, or a bound on the error rate among cases selected for automation.
