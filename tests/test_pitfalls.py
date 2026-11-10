import numpy as np

from src.pitfalls import run_aa_test, run_peeking_demo


def test_run_aa_test_has_approximate_alpha_rate():
    result = run_aa_test(n_simulations=1000, n_accounts=2000, alpha=0.05, seed=123)
    assert 0.03 < result["false_positive_rate"] < 0.07


def test_run_peeking_demo_is_more_liberal_than_aa():
    result = run_peeking_demo(days=20, n_accounts=2000, alpha=0.05, seed=123)
    assert result["false_positive_rate"] > result["aa_false_positive_rate"]
