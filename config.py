from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ExperimentConfig:
    """Single source of truth for the synthetic credit-offer experiment."""

    n_accounts: int = 5000
    seed: int = 202406
    randomization_seed: int = 202406

    # Risk band thresholds for PD bucket assignment.
    low_pd_cutoff: float = 0.15
    medium_pd_cutoff: float = 0.35

    # Ground-truth treatment effects by risk band.
    spend_effect_by_risk: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.08, "medium": 0.12, "high": 0.18}
    )
    delinquency_effect_by_risk: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.000, "medium": 0.000, "high": 0.025}
    )

    # Pre-period spend distribution.
    pre_spend_lognormal_mean: float = 4.25
    pre_spend_lognormal_sigma: float = 0.75

    # Account feature generation.
    credit_limit_min: float = 500.0
    credit_limit_max: float = 20000.0
    utilization_alpha: float = 2.2
    utilization_beta: float = 4.5
    fico_score_mean: float = 690.0
    fico_score_sd: float = 42.0

    # Logistic PD model specification.
    # Calibrated for the raw FICO-like score term so simulated accounts span all bands.
    pd_intercept: float = 5.2
    pd_pre_spend_coef: float = 0.42
    pd_utilization_coef: float = 2.4
    pd_fico_coef: float = -0.013
    pd_credit_coef: float = -0.00008

    # Post-period spend generation.
    post_spend_baseline_alpha: float = 0.22
    post_spend_pre_spend_coef: float = 0.62
    post_spend_noise_sd: float = 0.18


DEFAULT_CONFIG = ExperimentConfig()
