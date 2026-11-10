"""Synthetic account-level simulation for the credit-offer experiment."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, ExperimentConfig


def risk_band_from_pd(pd_value: float, low_cutoff: float = 0.15, medium_cutoff: float = 0.35) -> str:
    """Map a predicted default probability into a PD risk band."""
    if pd_value < low_cutoff:
        return "low"
    if pd_value < medium_cutoff:
        return "medium"
    return "high"


def simulate_accounts(
    n_accounts: Optional[int] = None,
    config: ExperimentConfig = DEFAULT_CONFIG,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic accounts with treatment assignment and planted effects.

    The returned dataframe is designed for downstream analysis in later stages. Each
    row represents one simulated account, and the planted treatment effects are stored
    explicitly so tests can compare estimate recovery against the known truth.
    """
    if n_accounts is None:
        n_accounts = config.n_accounts

    rng = np.random.default_rng(random_state if random_state is not None else config.seed)

    pre_spend = rng.lognormal(
        mean=config.pre_spend_lognormal_mean,
        sigma=config.pre_spend_lognormal_sigma,
        size=n_accounts,
    )

    # Credit limit and utilization are generated as correlated economic features.
    credit_limit = rng.uniform(
        low=config.credit_limit_min,
        high=config.credit_limit_max,
        size=n_accounts,
    )
    utilization = rng.beta(
        config.utilization_alpha,
        config.utilization_beta,
        size=n_accounts,
    )

    # FICO-like scores are centered around 690, but lower scores raise default risk.
    fico_score = rng.normal(
        loc=config.fico_score_mean,
        scale=config.fico_score_sd,
        size=n_accounts,
    )
    fico_score = np.clip(fico_score, 450.0, 830.0)

    # Create a modest relationship between credit limit and pre-spend to preserve realism.
    credit_limit = np.clip(
        credit_limit * (0.75 + 0.45 * (pre_spend / np.median(pre_spend))),
        config.credit_limit_min,
        config.credit_limit_max,
    )

    # Logistic PD model using account features.
    logit_pd = (
        config.pd_intercept
        + config.pd_pre_spend_coef * np.log1p(pre_spend)
        + config.pd_utilization_coef * utilization
        + config.pd_fico_coef * fico_score
        + config.pd_credit_coef * credit_limit
    )
    pd_prob = 1.0 / (1.0 + np.exp(-logit_pd))

    risk_band = np.array(
        [
            risk_band_from_pd(
                value,
                low_cutoff=config.low_pd_cutoff,
                medium_cutoff=config.medium_pd_cutoff,
            )
            for value in pd_prob
        ],
        dtype=object,
    )

    # Randomize accounts 50/50 using a fixed seed.
    treatment = rng.binomial(1, 0.5, size=n_accounts)

    # Ground-truth effects vary by risk band, and the delinquency increase is focused on high-risk accounts.
    true_spend_effect = np.array([config.spend_effect_by_risk[band] for band in risk_band], dtype=float)
    true_delinquency_effect = np.array(
        [config.delinquency_effect_by_risk[band] for band in risk_band],
        dtype=float,
    )

    # Post-period spend is strongly correlated with pre-period spend to provide CUPED with signal.
    post_spend_baseline = pre_spend * (
        1.0
        + config.post_spend_baseline_alpha
        + config.post_spend_pre_spend_coef * (np.log1p(pre_spend) / np.median(np.log1p(pre_spend)))
        + 0.25 * utilization
        - 0.10 * ((fico_score - config.fico_score_mean) / 100.0)
        + rng.normal(0.0, config.post_spend_noise_sd, size=n_accounts)
    )
    post_spend = post_spend_baseline * (1.0 + treatment * true_spend_effect)
    post_spend = np.maximum(post_spend, 0.0)

    # Secondary balance metric: a function of credit use and post-period spend.
    balance = np.clip(
        0.35 * post_spend + 0.25 * credit_limit * utilization + rng.normal(0.0, 150.0, size=n_accounts),
        0.0,
        None,
    )

    # Delinquency is based on underlying PD plus a modest treatment effect only for the high-risk band.
    delinquency_probability = np.clip(pd_prob * (1.0 + 0.55 * utilization), 0.0, 0.95)
    delinquency_probability = np.clip(
        delinquency_probability + treatment * np.where(risk_band == "high", true_delinquency_effect, 0.0),
        0.0,
        0.99,
    )
    delinquency_60plus = rng.binomial(1, delinquency_probability)

    data = pd.DataFrame(
        {
            "account_id": np.arange(n_accounts),
            "pre_spend": pre_spend,
            "post_spend": post_spend,
            "credit_limit": credit_limit,
            "utilization": utilization,
            "fico_score": fico_score,
            "pd": pd_prob,
            "risk_band": risk_band,
            "treatment": treatment,
            "balance": balance,
            "delinquency_60plus": delinquency_60plus,
            "true_spend_effect": true_spend_effect,
            "true_delinquency_effect": np.where(risk_band == "high", true_delinquency_effect, 0.0),
        }
    )
    return data


__all__ = ["simulate_accounts", "risk_band_from_pd"]
