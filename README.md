# Credit-offer experiment

This repository contains a synthetic, reproducible experiment for a credit-line-increase offer.

The data and effect sizes are intentionally simulated and should be treated as teaching material rather than evidence from a live portfolio. All outputs are labeled as synthetic throughout the code and design notes.

## Contents

- `config.py`: shared experiment parameters and random seeds.
- `src/simulate.py`: account-level data generation and planted treatment effects.
- `src/power.py`: planning calculations and power-curve visuals.
- `src/analysis.py`: SRM, mean-difference, CUPED, OLS, and guardrail analysis.
- `src/pitfalls.py`: A/A calibration and repeated-peeking demonstration.

## Stage summary

1. Stage 0: design and repo plan.
2. Stage 1: simulation and planted effects.
3. Stage 2: power assumptions and MDE planning.
4. Stage 3: core analysis and guardrail checks.
5. Stage 4: false-positive calibration and peeking pitfall.
6. Stage 5+: heterogeneity analysis and decision memo.

## Notes

- The experiment uses a 1:1 randomized assignment at the account level.
- The primary metric is 90-day spend per account.
- The guardrail metric is a 60+ DPD indicator over the same observation window.
- The design decision rule is intentionally conservative and simulation-only.
