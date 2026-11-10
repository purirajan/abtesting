"""Power and sample-size calculations for the synthetic credit-offer experiment."""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def normal_approx_sample_size_for_means(
    alpha: float = 0.05,
    power: float = 0.80,
    baseline_mean: float = 100.0,
    baseline_variance: float = 400.0,
    effect_fraction: float = 0.05,
) -> int:
    """Return the approximate per-arm sample size for a two-sample mean comparison.

    Uses the standard normal approximation for equal allocation:

        n ≈ 2 * (z_(1-alpha/2) + z_power)^2 * sigma^2 / delta^2

    where delta is the target absolute MDE. This is the design calculation described
    in the experiment plan, and the value is rounded up to the next integer.
    """
    if baseline_mean <= 0:
        raise ValueError("baseline_mean must be positive")
    if baseline_variance <= 0:
        raise ValueError("baseline_variance must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie between 0 and 1")
    if not (0.0 < power < 1.0):
        raise ValueError("power must lie between 0 and 1")
    if effect_fraction <= 0:
        raise ValueError("effect_fraction must be positive")

    effect_size = baseline_mean * effect_fraction
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_power = norm.ppf(power)
    per_arm_n = 2.0 * (z_alpha + z_power) ** 2 * baseline_variance / (effect_size ** 2)
    return int(np.ceil(per_arm_n))


def cuped_adjusted_sample_size(
    alpha: float = 0.05,
    power: float = 0.80,
    baseline_variance: float = 400.0,
    effect_fraction: float = 0.05,
    baseline_mean: float = 100.0,
    variance_reduction_fraction: float = 0.75,
) -> int:
    """Approximate sample size under a variance-reduction assumption for CUPED."""
    adjusted_variance = baseline_variance * variance_reduction_fraction
    return normal_approx_sample_size_for_means(
        alpha=alpha,
        power=power,
        baseline_mean=baseline_mean,
        baseline_variance=adjusted_variance,
        effect_fraction=effect_fraction,
    )


def make_power_curve(
    baseline_mean: float = 100.0,
    baseline_variance: float = 400.0,
    alpha: float = 0.05,
    power: float = 0.80,
    effect_grid: List[float] | None = None,
) -> Tuple[np.ndarray, np.ndarray, plt.Figure, plt.Axes]:
    """Return a sample size curve across a range of MDE fractions.

    The effect grid is expressed as a fraction of the baseline mean, e.g. 0.01 for a
    1% treatment lift.
    """
    if effect_grid is None:
        effect_grid = np.linspace(0.01, 0.20, 40)

    sample_sizes = []
    for effect_fraction in effect_grid:
        sample_sizes.append(
            normal_approx_sample_size_for_means(
                alpha=alpha,
                power=power,
                baseline_mean=baseline_mean,
                baseline_variance=baseline_variance,
                effect_fraction=float(effect_fraction),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.asarray(effect_grid) * 100.0, np.asarray(sample_sizes), marker="o")
    ax.set_xlabel("Minimum detectable effect (% of baseline mean)")
    ax.set_ylabel("Per-arm sample size")
    ax.set_title("Power planning curve for the credit-offer experiment")
    ax.grid(True, alpha=0.3)
    return np.asarray(effect_grid), np.asarray(sample_sizes), fig, ax


def power_summary(
    baseline_mean: float = 100.0,
    baseline_variance: float = 400.0,
    alpha: float = 0.05,
    power: float = 0.80,
    effect_fraction: float = 0.05,
    variance_reduction_fraction: float = 0.75,
) -> Dict[str, float | int]:
    """Summarize the planning assumptions in a single dictionary."""
    return {
        "baseline_mean": float(baseline_mean),
        "baseline_variance": float(baseline_variance),
        "effect_fraction": float(effect_fraction),
        "mde_absolute": float(baseline_mean * effect_fraction),
        "alpha": float(alpha),
        "power": float(power),
        "unadjusted_per_arm_n": normal_approx_sample_size_for_means(
            alpha=alpha,
            power=power,
            baseline_mean=baseline_mean,
            baseline_variance=baseline_variance,
            effect_fraction=effect_fraction,
        ),
        "cuped_per_arm_n": cuped_adjusted_sample_size(
            alpha=alpha,
            power=power,
            baseline_variance=baseline_variance,
            baseline_mean=baseline_mean,
            effect_fraction=effect_fraction,
            variance_reduction_fraction=variance_reduction_fraction,
        ),
    }


__all__ = [
    "normal_approx_sample_size_for_means",
    "cuped_adjusted_sample_size",
    "make_power_curve",
    "power_summary",
]
