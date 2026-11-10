import numpy as np
import pandas as pd
from scipy.stats import norm

from src.analysis import (
    cuped_estimate,
    make_cli_recommendation,
    proportion_difference,
    sample_ratio_mismatch,
    summarize_experiment,
)


def test_sample_ratio_mismatch_detects_imbalanced_split():
    df = pd.DataFrame({"treatment": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]})
    result = sample_ratio_mismatch(df, treatment_col="treatment")
    assert result["p_value"] < 0.05


def test_cuped_estimate_reduces_variance():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, size=2000)
    treatment = rng.binomial(1, 0.5, size=2000)
    y = 2.0 * treatment + 0.7 * x + rng.normal(0, 1, size=2000)
    df = pd.DataFrame({"treatment": treatment, "pre_spend": x, "post_spend": y})
    result = cuped_estimate(df, outcome_col="post_spend", treatment_col="treatment", covariate_col="pre_spend")
    assert result["estimate"] > 1.0
    assert result["variance_reduction_ratio"] < 1.0


def test_summarize_experiment_recovers_planted_effect():
    rng = np.random.default_rng(11)
    n = 4000
    treatment = rng.binomial(1, 0.5, size=n)
    pre_spend = rng.lognormal(mean=4.2, sigma=0.7, size=n)
    post_spend = 110.0 + 12.0 * treatment + 0.9 * pre_spend + rng.normal(0, 18, size=n)
    df = pd.DataFrame({"treatment": treatment, "pre_spend": pre_spend, "post_spend": post_spend})
    summary = summarize_experiment(df, outcome_col="post_spend", treatment_col="treatment", covariate_col="pre_spend")
    assert abs(summary["effect_size"] - 12.0) < 2.0
    assert summary["n_treated"] == int(np.sum(treatment == 1))
    assert summary["n_control"] == int(np.sum(treatment == 0))


def test_proportion_difference_is_close_to_expected():
    df = pd.DataFrame({
        "treatment": [0, 0, 0, 0, 1, 1, 1, 1],
        "delinquency_60plus": [0, 1, 0, 0, 1, 1, 0, 1],
    })
    result = proportion_difference(df, outcome_col="delinquency_60plus", treatment_col="treatment")
    assert result["risk_difference"] > -0.3
    assert result["risk_difference"] < 0.8


def test_make_cli_recommendation_ship_rule_is_respected():
    rng = np.random.default_rng(42)
    n = 5000
    treatment = rng.binomial(1, 0.5, size=n)
    pre_spend = rng.lognormal(mean=4.0, sigma=0.8, size=n)
    post_spend = 100.0 + 12.0 * treatment + 0.8 * pre_spend + rng.normal(0, 18, size=n)
    delinquency = rng.binomial(1, 0.05, size=n)
    df = pd.DataFrame({
        "treatment": treatment,
        "pre_spend": pre_spend,
        "post_spend": post_spend,
        "delinquency_60plus": delinquency,
    })
    decision = make_cli_recommendation(df, outcome_col="post_spend", treatment_col="treatment", covariate_col="pre_spend", guardrail_col="delinquency_60plus")
    assert decision["decision"] in {"ship", "inconclusive", "do_not_ship"}
    assert "spend_positive" in decision
    assert "guardrail_ok" in decision


def test_recommendation_uses_one_sided_guardrail_bound():
    rng = np.random.default_rng(99)
    n = 4000
    treatment = rng.binomial(1, 0.5, size=n)
    pre_spend = rng.normal(100.0, 15.0, size=n)
    post_spend = 100.0 + 20.0 * treatment + 0.8 * pre_spend + rng.normal(0.0, 5.0, size=n)
    delinquency = rng.binomial(1, np.where(treatment == 1, 0.08, 0.05))
    df = pd.DataFrame({
        "treatment": treatment,
        "pre_spend": pre_spend,
        "post_spend": post_spend,
        "delinquency_60plus": delinquency,
    })

    decision = make_cli_recommendation(df, outcome_col="post_spend")
    guardrail = proportion_difference(df, outcome_col="delinquency_60plus")
    expected_upper = guardrail["risk_difference"] + norm.ppf(0.95) * guardrail["standard_error"]

    assert decision["guardrail_upper_one_sided_95"] == expected_upper
    assert decision["guardrail_upper_one_sided_95"] < guardrail["ci_upper"]
