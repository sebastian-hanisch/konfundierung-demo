"""Orakel: Kernroutinen gegen einen unabhängigen Rechenweg (scipy). Alles in unter zehn Sekunden; ohne scipy überspringt sich die Datei."""

import math

import numpy as np
import pytest

scipy_stats = pytest.importorskip("scipy.stats")

import kausal_generator as gen  # noqa: E402
import kfd_model as m  # noqa: E402
from kausal_generator import SplitMix64  # noqa: E402


@pytest.mark.parametrize("p", [1e-9, 1e-5, 0.003, 0.05, 0.3, 0.5, 0.8, 0.97, 0.9999, 1 - 1e-9])
def test_ppf_cdf_pdf_against_scipy(p):
    x = float(gen.normal_ppf(np.array([p]))[0])
    assert x == pytest.approx(float(scipy_stats.norm.ppf(p)), abs=1e-6)
    assert float(gen.normal_cdf(x)) == pytest.approx(float(scipy_stats.norm.cdf(x)), abs=1e-12)
    assert float(gen.normal_pdf(x)) == pytest.approx(float(scipy_stats.norm.pdf(x)), rel=1e-12)


@pytest.mark.parametrize("gamma,a", [(0.5, 0.0), (1.0, 0.0), (2.0, 0.5), (1.5, -1.0), (0.25, 2.0)])
def test_conditional_means_against_truncated_normal(gamma, a):
    """E[Z | X] über die abgeschnittene Normalverteilung von S = a + gamma*Z + eta (scipy.stats.truncnorm), nicht über die Mills-Formel."""
    v = 1.0 + gamma ** 2
    sd = math.sqrt(v)
    up = scipy_stats.truncnorm(a=(0.0 - a) / sd, b=np.inf, loc=a, scale=sd).mean()
    down = scipy_stats.truncnorm(a=-np.inf, b=(0.0 - a) / sd, loc=a, scale=sd).mean()
    e0, e1 = m.mean_z_given_x(gamma, a)
    assert e1 == pytest.approx(gamma * (up - a) / v, rel=1e-9)
    assert e0 == pytest.approx(gamma * (down - a) / v, rel=1e-9)
    assert m.p_treated(gamma, a) == pytest.approx(float(scipy_stats.norm.sf(0.0, loc=a, scale=sd)), rel=1e-9)


@pytest.mark.parametrize("gamma,beta", [(0.5, 4.0), (1.0, 4.0), (2.0, 2.0), (1.0, 6.0)])
def test_naive_limit_against_monte_carlo(gamma, beta):
    scm = m.model_scm(gamma, beta, 3.5, 5.0)
    _, x, y = (v[0] for v in m.observe(scm, 300000, SplitMix64(21)))
    d = y[x == 1].mean() - y[x == 0].mean()
    se = math.sqrt(y[x == 1].var() / (x == 1).sum() + y[x == 0].var() / (x == 0).sum())
    assert abs(d - m.naive_limit(gamma, beta, 3.5)) < 4.5 * se


def test_welch_against_scipy_ttest():
    scm = m.model_scm(1.0, 4.0, 3.5, 5.0)
    _, x, y = m.observe(scm, 300, SplitMix64(22), reps=5)
    d, se = m.welch_difference(y, x)
    for r in range(5):
        res = scipy_stats.ttest_ind(y[r][x[r] == 1], y[r][x[r] == 0], equal_var=False)
        assert d[r] / se[r] == pytest.approx(float(res.statistic), rel=1e-9)


@pytest.mark.parametrize("p", [1e-5, 0.003, 0.02, 0.03, 0.2, 0.5, 0.77, 0.97, 0.99, 1 - 1e-5])
def test_ppf_is_accurate_to_ten_digits_where_the_halley_step_applies(p):
    assert float(gen.normal_ppf(np.array([p]))[0]) == pytest.approx(float(scipy_stats.norm.ppf(p)), abs=1e-10)


@pytest.mark.parametrize("seed,n,scale", [(1, 5000, 1.08), (3, 400, 1.12), (4, 600, 1.0), (5, 300, 1.2), (6, 800, 1.1)])
def test_ks_against_scipy(seed, n, scale):
    a = SplitMix64(seed).normal(n)
    b = SplitMix64(seed + 100).normal(n) * scale
    d, p = m.ks_two_sample(a, b)
    ref = scipy_stats.ks_2samp(a, b, method="asymp")
    assert d == pytest.approx(float(ref.statistic), abs=1e-12)
    # Maßstab für den p-Wert ist die Kolmogorov-Grenzverteilung (kstwobign); scipys method="asymp" nutzt die Ein-Stichproben-Verteilung mit gerundetem
    # n und weicht bei kleinem n um einige Hundertstel ab.
    ne = n * n / (2.0 * n)
    assert p == pytest.approx(float(scipy_stats.kstwobign.sf(np.sqrt(ne) * d)), abs=1e-6)
    assert abs(p - float(ref.pvalue)) < 0.03


def test_ks_p_values_cover_the_moderate_range():
    """Mindestens ein Fall mit p zwischen 0,05 und 0,95, damit Vorfaktor und Exponent der Reihe geprüft sind."""
    ps = []
    for seed in range(1, 12):
        a, b = SplitMix64(seed).normal(300), SplitMix64(seed + 50).normal(300)
        ps.append(m.ks_two_sample(a, b)[1])
    assert any(0.05 < p < 0.95 for p in ps)


@pytest.mark.parametrize("gamma,a", [(1.0, 1.0), (2.0, -1.0), (0.5, 0.7)])
def test_mediator_world_matches_the_confounded_world_for_a_threshold_other_than_zero(gamma, a):
    scm = gen.standorte_confounded(gamma, a, 4.0, 3.5, 5.0)
    d = gen.sample(scm, 200000, SplitMix64(31))
    za, xa, ya = d["Schwankung"][0], d["Faktor"][0], d["Servicegrad"][0]
    zb, xb, yb = m.mediator_sample(gamma, 4.0, 3.5, 5.0, 200000, SplitMix64(32), a=a)
    assert xb.mean() == pytest.approx(xa.mean(), abs=0.01)
    for xv in (0.0, 1.0):
        assert zb[xb == xv].mean() == pytest.approx(za[xa == xv].mean(), abs=0.02)
        assert zb[xb == xv].std() == pytest.approx(za[xa == xv].std(), abs=0.02)
        assert yb[xb == xv].mean() == pytest.approx(ya[xa == xv].mean(), abs=0.1)
        assert scipy_stats.ks_2samp(za[xa == xv], zb[xb == xv]).pvalue > 1e-4


def test_adjusted_difference_against_numpy_lstsq_at_scale():
    scm = m.model_scm(1.0, 4.0, 3.5, 5.0)
    z, x, y = (v[0] for v in m.observe(scm, 5000, SplitMix64(23)))
    b, _ = m.adjusted_difference(y[None, :], x[None, :], z[None, :])
    design = np.stack([np.ones(5000), x, z], axis=1)
    assert b[0] == pytest.approx(float(np.linalg.lstsq(design, y, rcond=None)[0][1]), rel=1e-9)


def test_mediator_world_is_indistinguishable_from_the_confounded_world():
    """Zwei verschiedene Ziehwege, dieselbe Verteilung: 24 KS-Tests (scipy) über 6 Läufe, kaum mehr Treffer unter 0,05 als der Zufall."""
    scm = m.model_scm(1.0, 4.0, 3.5, 5.0)
    pvals = []
    for r in range(6):
        za, xa, ya = (v[0] for v in m.observe(scm, 20000, SplitMix64(300 + r)))
        zb, xb, yb = m.mediator_sample(1.0, 4.0, 3.5, 5.0, 20000, SplitMix64(400 + r))
        for xv in (0.0, 1.0):
            pvals.append(scipy_stats.ks_2samp(za[xa == xv], zb[xb == xv]).pvalue)
            pvals.append(scipy_stats.ks_2samp(ya[xa == xv], yb[xb == xv]).pvalue)
    assert np.mean(np.array(pvals) < 0.05) <= 0.25 and min(pvals) > 1e-4
