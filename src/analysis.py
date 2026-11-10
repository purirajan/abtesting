"""Core analysis utilities for the simulated credit-offer experiment."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm


def _treatment_groups(df: pd.DataFrame, treatment_col: str = "treatment") -> Tuple[pd.Series, pd.Series]:
    """Split a dataframe into treatment and control subsets."""
    if treatment_col not in df.columns:
        raise ValueError(f"Column '{treatment_col}' not found in dataframe")

    treated = df.loc[df[treatment_col] == 1]
    control = df.loc[df[treatment_col] == 0]
    if treated.empty or control.empty:
        raise ValueError("Both treatment and control groups must be non-empty")
    return treated, control


def sample_ratio_mismatch(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    alpha: float = 0.05,
) -> Dict[str, float | int]:
    """Check if the treatment/control split deviates materially from a 50/50 design."""
    n_total = len(df)
    if n_total == 0:
        raise ValueError("DataFrame is empty")

    n_treated = int(df[treatment_col].sum())
    n_control = int(n_total - n_treated)
    expected_treated = n_total / 2.0
    expected_control = n_total / 2.0

    chi_sq = (
        (n_treated - expected_treated) ** 2 / expected_treated
        + (n_control - expected_control) ** 2 / expected_control
    )
    p_value = 1.0 - chi2.cdf(chi_sq, df=1)
    return {
        "n_treated": n_treated,
        "n_control": n_control,
        "expected_treated": expected_treated,
        "expected_control": expected_control,
        "chi_square": float(chi_sq),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
    }


def welch_difference_in_means(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment",
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Estimate the difference in means between treated and control groups."""
    treated, control = _treatment_groups(df, treatment_col)

    y_t = treated[outcome_col].astype(float)
    y_c = control[outcome_col].astype(float)

    estimate = float(y_t.mean() - y_c.mean())
    var_t = float(y_t.var(ddof=1)) if len(y_t) > 1 else 0.0
    var_c = float(y_c.var(ddof=1)) if len(y_c) > 1 else 0.0
    se = float(np.sqrt(var_t / len(y_t) + var_c / len(y_c)))
    z_stat = estimate / se if se > 0 else 0.0
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
    ci_lower = estimate - norm.ppf(1.0 - alpha / 2.0) * se
    ci_upper = estimate + norm.ppf(1.0 - alpha / 2.0) * se
    return {
        "estimate": estimate,
        "standard_error": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": float(p_value),
        "z_stat": float(z_stat),
    }


def cuped_estimate(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment",
    covariate_col: str = "pre_spend",
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Estimate the treatment effect after adjusting for a pre-treatment covariate."""
    if covariate_col not in df.columns:
        raise ValueError(f"Covariate '{covariate_col}' not found in dataframe")

    treated, control = _treatment_groups(df, treatment_col)
    x = df[covariate_col].astype(float)
    y = df[outcome_col].astype(float)

    x_mean = float(x.mean())
    x_var = float(np.var(x, ddof=1))
    theta = float(np.cov(x, y, ddof=1)[0, 1] / x_var) if x_var > 0 else 0.0
    y_adjusted = y - theta * (x - x_mean)

    y_t_adjusted = y_adjusted[df[treatment_col] == 1]
    y_c_adjusted = y_adjusted[df[treatment_col] == 0]

    estimate = float(y_t_adjusted.mean() - y_c_adjusted.mean())
    var_t = float(y_t_adjusted.var(ddof=1)) if len(y_t_adjusted) > 1 else 0.0
    var_c = float(y_c_adjusted.var(ddof=1)) if len(y_c_adjusted) > 1 else 0.0
    se = float(np.sqrt(var_t / len(y_t_adjusted) + var_c / len(y_c_adjusted)))
    ci_lower = estimate - norm.ppf(1.0 - alpha / 2.0) * se
    ci_upper = estimate + norm.ppf(1.0 - alpha / 2.0) * se

    y_t_unadjusted = treated[outcome_col].astype(float)
    y_c_unadjusted = control[outcome_col].astype(float)
    unadjusted_var = (
        float(np.var(y_t_unadjusted, ddof=1) / len(y_t_unadjusted))
        + float(np.var(y_c_unadjusted, ddof=1) / len(y_c_unadjusted))
    )
    adjusted_var = (
        float(np.var(y_t_adjusted, ddof=1) / len(y_t_adjusted))
        + float(np.var(y_c_adjusted, ddof=1) / len(y_c_adjusted))
    )
    variance_reduction_ratio = float(adjusted_var / unadjusted_var) if unadjusted_var > 0 else 1.0

    return {
        "estimate": estimate,
        "theta": theta,
        "standard_error": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "variance_reduction_ratio": variance_reduction_ratio,
    }


def ols_adjusted_estimate(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment",
    covariate_col: str = "pre_spend",
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Fit a simple OLS cross-check: post_spend ~ treatment + pre_spend."""
    model = smf.ols(
        f"{outcome_col} ~ {treatment_col} + {covariate_col}",
        data=df,
    ).fit()
    treatment_coef = model.params[treatment_col]
    ci = model.conf_int(alpha=alpha).loc[treatment_col]
    return {
        "estimate": float(treatment_coef),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "p_value": float(model.pvalues[treatment_col]),
    }


def proportion_difference(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment",
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Compute the estimated risk difference and 95% confidence interval."""
    treated, control = _treatment_groups(df, treatment_col)
    p_t = float(treated[outcome_col].mean())
    p_c = float(control[outcome_col].mean())
    estimate = p_t - p_c
    n_t = len(treated)
    n_c = len(control)
    se = float(np.sqrt(p_t * (1.0 - p_t) / n_t + p_c * (1.0 - p_c) / n_c))
    z_value = estimate / se if se > 0 else 0.0
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_value)))
    ci_lower = estimate - norm.ppf(1.0 - alpha / 2.0) * se
    ci_upper = estimate + norm.ppf(1.0 - alpha / 2.0) * se
    return {
        "risk_difference": estimate,
        "treated_rate": p_t,
        "control_rate": p_c,
        "standard_error": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": float(p_value),
    }


def summarize_experiment(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment",
    covariate_col: str = "pre_spend",
    risk_band_col: str = "risk_band",
    outcome_guardrail_col: str = "delinquency_60plus",
    alpha: float = 0.05,
) -> Dict[str, float | int | dict]:
    """Return a compact summary of the principal spend and guardrail estimates."""
    df = df.copy()
    if outcome_guardrail_col not in df.columns:
        df[outcome_guardrail_col] = 0.0

    split = sample_ratio_mismatch(df, treatment_col=treatment_col, alpha=alpha)
    spend = welch_difference_in_means(df, outcome_col=outcome_col, treatment_col=treatment_col, alpha=alpha)
    cuped = cuped_estimate(df, outcome_col=outcome_col, treatment_col=treatment_col, covariate_col=covariate_col, alpha=alpha)
    ols = ols_adjusted_estimate(df, outcome_col=outcome_col, treatment_col=treatment_col, covariate_col=covariate_col, alpha=alpha)
    guardrail = proportion_difference(df, outcome_col=outcome_guardrail_col, treatment_col=treatment_col, alpha=alpha)

    summary = {
        "n_treated": int(df[treatment_col].sum()),
        "n_control": int(len(df) - int(df[treatment_col].sum())),
        "effect_size": float(spend["estimate"]),
        "effect_ci_lower": float(spend["ci_lower"]),
        "effect_ci_upper": float(spend["ci_upper"]),
        "effect_p_value": float(spend["p_value"]),
        "cuped_estimate": float(cuped["estimate"]),
        "cuped_ci_lower": float(cuped["ci_lower"]),
        "cuped_ci_upper": float(cuped["ci_upper"]),
        "cuped_variance_reduction_ratio": float(cuped["variance_reduction_ratio"]),
        "ols_estimate": float(ols["estimate"]),
        "ols_p_value": float(ols["p_value"]),
        "guardrail_risk_difference": float(guardrail["risk_difference"]),
        "guardrail_ci_upper": float(guardrail["ci_upper"]),
        "guardrail_ci_lower": float(guardrail["ci_lower"]),
        "split_check": split,
    }
    return summary


def estimate_effects_by_risk_band(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment",
    risk_band_col: str = "risk_band",
    guardrail_col: str = "delinquency_60plus",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute spend and delinquency contrasts for each PD risk band."""
    rows: List[Dict[str, float | str]] = []
    for band, band_df in df.groupby(risk_band_col, dropna=False):
        spend = welch_difference_in_means(band_df, outcome_col=outcome_col, treatment_col=treatment_col, alpha=alpha)
        delinquency = proportion_difference(band_df, outcome_col=guardrail_col, treatment_col=treatment_col, alpha=alpha)
        rows.append(
            {
                "risk_band": band,
                "spend_effect": float(spend["estimate"]),
                "spend_ci_lower": float(spend["ci_lower"]),
                "spend_ci_upper": float(spend["ci_upper"]),
                "delinquency_risk_difference": float(delinquency["risk_difference"]),
                "delinquency_ci_upper": float(delinquency["ci_upper"]),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "sample_ratio_mismatch",
    "welch_difference_in_means",
    "cuped_estimate",
    "ols_adjusted_estimate",
    "proportion_difference",
    "summarize_experiment",
    "estimate_effects_by_risk_band",
]
