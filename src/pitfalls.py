"""A/A calibration and peeking demonstration for the experiment design."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.analysis import welch_difference_in_means


def run_aa_test(
    n_simulations: int = 500,
    n_accounts: int = 5000,
    alpha: float = 0.05,
    seed: int = 101,
) -> Dict[str, float | int]:
    """Run repeated A/A tests and report the empirical false-positive rate."""
    rng = np.random.default_rng(seed)
    false_positives = 0

    for _ in range(n_simulations):
        treatment = rng.binomial(1, 0.5, size=n_accounts)
        outcome = rng.normal(loc=0.0, scale=1.0, size=n_accounts)
        df = pd.DataFrame({"treatment": treatment, "outcome": outcome})
        result = welch_difference_in_means(df, outcome_col="outcome", treatment_col="treatment", alpha=alpha)
        if result["p_value"] < alpha:
            false_positives += 1

    return {
        "n_simulations": int(n_simulations),
        "n_accounts": int(n_accounts),
        "alpha": float(alpha),
        "false_positive_rate": float(false_positives / n_simulations),
    }


def run_peeking_demo(
    days: int = 20,
    n_accounts: int = 5000,
    alpha: float = 0.05,
    seed: int = 202,
) -> Dict[str, float | int | list]:
    """Simulate repeated significance checks over time and show the inflation in false positives."""
    rng = np.random.default_rng(seed)
    false_positives = 0
    sample_size_path: List[int] = []
    stopped_days: List[int] = []

    for sim_idx in range(200):
        outcome = rng.normal(loc=0.0, scale=1.0, size=n_accounts)
        treatment = rng.binomial(1, 0.5, size=n_accounts)
        df = pd.DataFrame({"treatment": treatment, "outcome": outcome})
        for day in range(1, days + 1):
            sample_fraction = day / days
            sample = df.iloc[: max(10, int(n_accounts * sample_fraction))]
            result = welch_difference_in_means(sample, outcome_col="outcome", treatment_col="treatment", alpha=alpha)
            if result["p_value"] < alpha:
                false_positives += 1
                stopped_days.append(day)
                sample_size_path.append(len(sample))
                break

    aa_result = run_aa_test(n_simulations=200, n_accounts=n_accounts, alpha=alpha, seed=seed + 1)
    return {
        "days": int(days),
        "alpha": float(alpha),
        "false_positive_rate": float(false_positives / 200),
        "aa_false_positive_rate": float(aa_result["false_positive_rate"]),
        "stopped_days": stopped_days,
        "sample_sizes": sample_size_path,
    }


__all__ = ["run_aa_test", "run_peeking_demo"]
