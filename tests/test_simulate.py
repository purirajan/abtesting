import pandas as pd

from config import DEFAULT_CONFIG
from src.simulate import simulate_accounts


def test_simulation_is_reproducible_and_contains_all_risk_bands():
    first = simulate_accounts(n_accounts=5000, random_state=123)
    second = simulate_accounts(n_accounts=5000, random_state=123)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["risk_band"]) == {"low", "medium", "high"}
    assert set(first["treatment"].unique()) == {0, 1}
    assert abs(first["treatment"].mean() - 0.5) < 0.05


def test_simulation_exposes_configured_planted_effects():
    simulated = simulate_accounts(n_accounts=5000, config=DEFAULT_CONFIG, random_state=456)

    for band, expected_effect in DEFAULT_CONFIG.spend_effect_by_risk.items():
        band_rows = simulated.loc[simulated["risk_band"] == band]
        assert not band_rows.empty
        assert (band_rows["true_spend_effect"] == expected_effect).all()

    high_risk = simulated.loc[simulated["risk_band"] == "high"]
    other_risk = simulated.loc[simulated["risk_band"] != "high"]
    assert (high_risk["true_delinquency_effect"] > 0).all()
    assert (other_risk["true_delinquency_effect"] == 0).all()