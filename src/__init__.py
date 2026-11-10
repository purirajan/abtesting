"""Package marker for the credit-offer experiment source code."""

from .analysis import (
    cuped_estimate,
    estimate_effects_by_risk_band,
    ols_adjusted_estimate,
    proportion_difference,
    sample_ratio_mismatch,
    summarize_experiment,
    welch_difference_in_means,
)
from .pitfalls import run_aa_test, run_peeking_demo
from .power import (
    cuped_adjusted_sample_size,
    make_power_curve,
    normal_approx_sample_size_for_means,
    power_summary,
)
from .simulate import simulate_accounts

__all__ = [
    "simulate_accounts",
    "normal_approx_sample_size_for_means",
    "cuped_adjusted_sample_size",
    "make_power_curve",
    "power_summary",
    "sample_ratio_mismatch",
    "welch_difference_in_means",
    "cuped_estimate",
    "ols_adjusted_estimate",
    "proportion_difference",
    "summarize_experiment",
    "estimate_effects_by_risk_band",
    "run_aa_test",
    "run_peeking_demo",
]
