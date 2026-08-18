# Conformal Selective Prediction

[![Tests](https://github.com/EnzoCanonero/conformal-selective-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/EnzoCanonero/conformal-selective-prediction/actions/workflows/ci.yml)

This repository develops a conformal selective-prediction layer for reliable
automation. Instead of forcing a model to act on every input, it produces
prediction sets with coverage guarantees and defers uncertain cases to human
review.

The framework will first be validated on BANKING77 for automated banking-support
routing. It will then be applied to LLM outputs and stress-tested under
distribution shift to measure the trade-off between automation rate and
operational risk.

## Current status

The reusable multiclass split-conformal core now includes LAC scores,
finite-sample calibration, prediction-set construction, and basic set-quality
metrics. A synthetic IID multiclass example exercises the complete workflow and
checks its marginal coverage.

The next milestones will introduce the automation and deferral policy, then
apply it to BANKING77 before moving to LLM and distribution-shift experiments.

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

## Synthetic IID example

Run the complete training, calibration, and evaluation workflow with:

```bash
python examples/synthetic_multiclass.py
```

Using the fixed seed and a 4,000/2,000/2,000 train/calibration/test split, the
example produces:

| Target coverage | Empirical coverage | Average set size |
|----------------:|-------------------:|-----------------:|
| 0.900           | 0.908              | 1.155            |

The example reports the target coverage, empirical test coverage, and average
prediction-set size. A single finite test set can fall slightly above or below
the target because the conformal guarantee is marginal rather than a
deterministic lower bound for every realized test set.

## Statistical scope

The intended split-conformal guarantee is finite-sample **marginal coverage**.
It relies on exchangeability of the calibration examples and future examples.
The guarantee does not automatically imply conditional coverage for every
subgroup, or a bound on the error rate among cases selected for automation.

## License

This project is available under the [MIT License](LICENSE).
