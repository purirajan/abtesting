import numpy as np

from src.power import normal_approx_sample_size_for_means


def test_normal_approx_sample_size_for_means_is_positive():
    n = normal_approx_sample_size_for_means(
        alpha=0.05,
        power=0.8,
        baseline_mean=100.0,
        baseline_variance=400.0,
        effect_fraction=0.05,
    )
    assert n > 0
    assert isinstance(n, int)


def test_sample_size_scales_with_variance():
    n_low = normal_approx_sample_size_for_means(
        alpha=0.05,
        power=0.8,
        baseline_mean=100.0,
        baseline_variance=400.0,
        effect_fraction=0.05,
    )
    n_high = normal_approx_sample_size_for_means(
        alpha=0.05,
        power=0.8,
        baseline_mean=100.0,
        baseline_variance=900.0,
        effect_fraction=0.05,
    )
    assert n_high > n_low
