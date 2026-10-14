"""Kern: Mini-Instanz von Hand, Eigenschaften der geschlossenen Form, Schätzer, zweites Diagramm, KS-Test."""

import math

import numpy as np
import pytest

import kfd_constants as C
import kfd_model as m
from kausal_generator import SplitMix64


# ----------------------------------------------------------------------------- Mini-Instanz

def test_mini_instance_naive_difference_and_group_means(mini):
    y, x, _ = mini
    assert y[x == 1].mean() == pytest.approx(87.5)
    assert y[x == 0].mean() == pytest.approx(88.25)
    d, se = m.welch_difference(y[None, :], x[None, :])
    assert d[0] == pytest.approx(-0.75) and se[0] > 0


def test_mini_instance_effects_within_groups_and_standardized_difference(mini):
    y, x, z = mini
    effects = []
    for g in (0.0, 1.0):
        sel = z == g
        d, _ = m.welch_difference(y[sel][None, :], x[sel][None, :])
        effects.append(d[0])
    assert effects == pytest.approx([4.0, 5.0])
    assert m.standardized_difference(effects, [8, 8]) == pytest.approx(4.5)
    assert m.standardized_difference(effects, [3, 1]) == pytest.approx(4.25)      # Gewichte werden normiert


def test_mini_instance_regression_with_the_group_indicator_gives_the_standardized_difference(mini):
    """Zwei Gruppen mit gleich großen Gewichten: die Regression mit dem Gruppenmerkmal trifft nicht exakt 4,5, aber liegt zwischen +4 und +5 und weit über -0,75."""
    y, x, z = mini
    b, _ = m.adjusted_difference(y[None, :], x[None, :], z[None, :])
    assert 4.0 < b[0] < 5.0


# ----------------------------------------------------------------------------- geschlossene Form

def test_p_treated_without_selection_is_the_threshold_probability():
    assert m.p_treated(0.0, 0.0) == pytest.approx(0.5)
    assert m.p_treated(0.0, 1.0) == pytest.approx(0.8413447460685429)
    assert m.p_treated(1.0, 0.0) == pytest.approx(0.5)


@pytest.mark.parametrize("gamma,a", [(0.5, 0.0), (1.0, 0.0), (2.0, 0.0), (1.0, 1.0), (1.0, -1.0), (3.0, 0.7)])
def test_mean_fluctuation_averages_to_zero_over_both_groups(gamma, a):
    e0, e1 = m.mean_z_given_x(gamma, a)
    p = m.p_treated(gamma, a)
    assert p * e1 + (1.0 - p) * e0 == pytest.approx(0.0, abs=1e-12)
    assert e1 > 0.0 > e0


def test_mean_fluctuation_vanishes_without_selection_and_is_odd_at_threshold_zero():
    assert m.mean_z_given_x(0.0, 0.0) == (0.0, 0.0)
    e0, e1 = m.mean_z_given_x(1.0, 0.0)
    assert e0 == pytest.approx(-e1)


def test_naive_limit_equals_tau_without_selection_or_without_influence():
    assert m.naive_limit(0.0, 4.0, 3.5) == pytest.approx(3.5)
    assert m.naive_limit(1.0, 0.0, 3.5) == pytest.approx(3.5)
    assert m.naive_limit(1.0, 4.0, 3.5) < 0.0 < 3.5


def test_naive_limit_falls_with_selection_and_with_influence():
    by_gamma = [m.naive_limit(g, 4.0, 3.5) for g in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)]
    by_beta = [m.naive_limit(1.0, b, 3.5) for b in (0.0, 1.0, 2.0, 4.0, 6.0)]
    assert all(a > b for a, b in zip(by_gamma, by_gamma[1:])) and all(a > b for a, b in zip(by_beta, by_beta[1:]))


def test_naive_limit_is_linear_in_tau_with_unit_slope():
    assert m.naive_limit(1.0, 4.0, 5.5) - m.naive_limit(1.0, 4.0, 3.5) == pytest.approx(2.0)


def test_standard_world_changes_sign():
    assert m.naive_limit(C.DEFAULT_GAMMA, C.DEFAULT_BETA, C.DEFAULT_TAU) == pytest.approx(-1.0135, abs=1e-4)


# ----------------------------------------------------------------------------- Schätzer

def test_welch_matches_a_loop_version():
    rng = SplitMix64(4)
    y = rng.normal(5 * 30).reshape(5, 30) * 3 + 80
    x = (rng.uniform(5 * 30).reshape(5, 30) < 0.4).astype(float)
    d, se = m.welch_difference(y, x)
    for r in range(5):
        a, b = y[r][x[r] == 1], y[r][x[r] == 0]
        assert d[r] == pytest.approx(a.mean() - b.mean())
        assert se[r] == pytest.approx(math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def test_adjusted_difference_matches_least_squares_per_row():
    rng = SplitMix64(6)
    z = rng.normal(4 * 60).reshape(4, 60)
    x = (z + rng.normal(4 * 60).reshape(4, 60) > 0).astype(float)
    y = 80 - 4 * z + 3.5 * x + rng.normal(4 * 60).reshape(4, 60) * 5
    b, se = m.adjusted_difference(y, x, z)
    for r in range(4):
        design = np.stack([np.ones(60), x[r], z[r]], axis=1)
        coef, res, *_ = np.linalg.lstsq(design, y[r], rcond=None)
        sigma2 = res[0] / (60 - 3)
        cov = sigma2 * np.linalg.inv(design.T @ design)
        assert b[r] == pytest.approx(coef[1], rel=1e-9)
        assert se[r] == pytest.approx(math.sqrt(cov[1, 1]), rel=1e-9)


def test_covers_includes_the_interval_boundary():
    one = np.array([1.0])
    assert m.covers(np.array([0.0]), one, C.Z_CRIT).tolist() == [True]
    assert m.covers(np.array([0.0]), one, -C.Z_CRIT).tolist() == [True]


def test_covers_is_symmetric_around_the_estimate():
    est, se = np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])
    assert m.covers(est, se, 1.0 + 1.9).tolist() == [True, True, True]
    assert m.covers(est, se, 1.0 + 2.0).tolist() == [False, False, False]
    assert m.covers(est, se, 1.0 - 2.0).tolist() == [False, False, False]


def test_adjusting_for_the_confounder_recovers_the_effect_and_the_naive_difference_does_not():
    scm = m.model_scm(1.0, 4.0, 3.5, 5.0)
    z, x, y = m.observe(scm, 20000, SplitMix64(12))
    b, se = m.adjusted_difference(y, x, z)
    d, se_d = m.welch_difference(y, x)
    assert abs(b[0] - 3.5) < 3 * se[0]
    assert abs(d[0] - 3.5) > 10 * se_d[0]


def test_coin_assignment_covers_about_ninety_five_percent():
    scm = m.model_scm(1.0, 4.0, 3.5, 5.0, "muenzwurf")
    _, x, y = m.observe(scm, 200, SplitMix64(13), reps=400)
    d, se = m.welch_difference(y, x)
    assert 0.92 <= m.covers(d, se, 3.5).mean() <= 0.98


def test_naive_coverage_collapses_with_more_data():
    scm = m.model_scm(1.0, 4.0, 3.5, 5.0)
    cover = []
    for n in (50, 1000):
        _, x, y = m.observe(scm, n, SplitMix64(14 + n), reps=200)
        d, se = m.welch_difference(y, x)
        cover.append(m.covers(d, se, 3.5).mean())
    assert cover[1] < cover[0] and cover[1] < 0.01


# ----------------------------------------------------------------------------- Zuteilung

def test_model_scm_with_coin_has_the_same_share_of_treated_and_no_dependence():
    obs, coin = m.model_scm(1.5, 4.0, 3.5, 5.0), m.model_scm(1.5, 4.0, 3.5, 5.0, "muenzwurf")
    z, x, _ = m.observe(coin, 100000, SplitMix64(1))
    assert x.mean() == pytest.approx(m.p_treated(1.5), abs=0.01)
    assert abs(np.corrcoef(z[0], x[0])[0, 1]) < 0.02
    z2, x2, _ = m.observe(obs, 100000, SplitMix64(1))
    assert np.corrcoef(z2[0], x2[0])[0, 1] > 0.4


# ----------------------------------------------------------------------------- zweites Diagramm

@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
def test_mediator_sample_has_the_same_marginals_as_the_confounded_world(gamma):
    scm = m.model_scm(gamma, 4.0, 3.5, 5.0)
    za, xa, ya = (v[0] for v in m.observe(scm, 100000, SplitMix64(1)))
    zb, xb, yb = m.mediator_sample(gamma, 4.0, 3.5, 5.0, 100000, SplitMix64(2))
    for xv in (0.0, 1.0):
        assert zb[xb == xv].mean() == pytest.approx(za[xa == xv].mean(), abs=0.02)
        assert zb[xb == xv].std() == pytest.approx(za[xa == xv].std(), abs=0.02)
        assert yb[xb == xv].mean() == pytest.approx(ya[xa == xv].mean(), abs=0.1)
    assert xb.mean() == pytest.approx(xa.mean(), abs=0.01)


def test_in_the_mediator_world_the_naive_difference_is_the_effect():
    zb, xb, yb = m.mediator_sample(1.0, 4.0, 3.5, 5.0, 300000, SplitMix64(3))
    d, _ = m.welch_difference(yb[None, :], xb[None, :])
    assert d[0] == pytest.approx(m.naive_limit(1.0, 4.0, 3.5), abs=0.06)


# ----------------------------------------------------------------------------- Kolmogorov-Smirnov

def test_ks_same_sample_has_distance_zero_and_p_one():
    a = SplitMix64(1).normal(500)
    d, p = m.ks_two_sample(a, a.copy())
    assert d == 0.0 and p == 1.0


def test_ks_separated_samples_have_distance_one_and_tiny_p():
    d, p = m.ks_two_sample(np.arange(100.0), np.arange(100.0) + 1000)
    assert d == 1.0 and p < 1e-10


def test_ks_p_values_are_roughly_uniform_for_the_same_law():
    ps = [m.ks_two_sample(SplitMix64(s).normal(400), SplitMix64(1000 + s).normal(400))[1] for s in range(80)]
    assert 0.0 <= min(ps) and max(ps) <= 1.0
    assert 0.02 <= np.mean(np.array(ps) < 0.1) <= 0.25
