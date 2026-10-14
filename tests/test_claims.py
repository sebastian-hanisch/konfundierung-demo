"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über feste Zufallsströme (kfd_constants.SEED_BASE), nie über einen einzelnen Seed als Befund; die Standard-Stichprobe der
Primäransicht ist ausdrücklich eine einzelne Stichprobe (Seed 35). Prozentwerte mit Toleranz 1 Punkt, Punktwerte mit 0,05, damit Rundung und
Bibliotheksversionen (CI installiert ungepinnt) nichts kippen.
"""

import pytest

import kfd_constants as C
import kfd_evaluation as ev
import kfd_model as model


@pytest.fixture(scope="module")
def cover():
    return {r["n"]: r for r in ev.coverage_experiment()}


@pytest.fixture(scope="module")
def cells():
    return ev.sign_map()


@pytest.fixture(scope="module")
def mech():
    return ev.mechanism_experiment()


@pytest.fixture(scope="module")
def proxy():
    return ev.proxy_experiment()


def pct(x):
    return 100 * x


# ----------------------------------------------------------------------------- Geschlossene Form und Standard-Stichprobe

def test_standard_world_closed_form_limit_is_minus_one():
    assert model.naive_limit(1.0, 4.0, 3.5) == pytest.approx(-1.01, abs=0.005)
    assert model.p_treated(1.0) == pytest.approx(0.5)


def test_standard_sample_shows_a_harm_where_the_effect_is_positive():
    a = ev.analyse(ev.Settings())
    assert a.naive == pytest.approx(-1.10, abs=0.05)
    assert (a.lo, a.hi) == (pytest.approx(-2.23, abs=0.05), pytest.approx(0.04, abs=0.05))
    assert pct(a.share_treated) == pytest.approx(53, abs=1) and a.mean_treated == pytest.approx(80.8, abs=0.05) and a.mean_untreated == pytest.approx(81.9, abs=0.05)
    assert a.covers is False and a.wrong_sign is True and a.truth == 3.5


def test_coin_preset_covers_and_is_the_same_as_no_selection():
    a = ev.analyse(ev.Settings(assign="muenzwurf"))
    b = ev.analyse(ev.Settings(gamma=0.0))
    assert a.naive == pytest.approx(2.84, abs=0.05) and a.covers is True
    assert a.naive == b.naive


def test_five_thousand_sites_preset_has_a_narrow_interval_around_the_wrong_number():
    a = ev.analyse(ev.Settings(**C.PRESETS["Fünftausend Standorte"]))
    assert a.hi - a.lo < 1.0 and a.covers is False and a.hi < 0.0


def test_weak_selection_preset_limit_is_a_fifth_of_the_effect():
    assert model.naive_limit(0.5, 4.0, 3.5) == pytest.approx(0.65, abs=0.005)
    assert model.naive_limit(0.5, 4.0, 3.5) / 3.5 == pytest.approx(0.18, abs=0.01)


# ----------------------------------------------------------------------------- Experiment 1

def test_experiment_1_naive_coverage_falls_to_zero(cover):
    assert pct(cover[50]["naive_cover"]) == pytest.approx(27, abs=2)
    for n in (200, 1000, 5000):
        assert cover[n]["naive_cover"] == 0.0
    assert all(cover[n]["naive_mean"] == pytest.approx(-1.0, abs=0.15) for n in cover)


def test_experiment_1_wrong_sign_share_grows_to_one(cover):
    assert pct(cover[50]["naive_wrong"]) == pytest.approx(69, abs=2.5)
    assert pct(cover[200]["naive_wrong"]) == pytest.approx(90, abs=2.5)
    assert pct(cover[1000]["naive_wrong"]) >= 98 and cover[5000]["naive_wrong"] == 1.0


def test_experiment_1_coin_and_adjusted_cover_about_ninety_five_percent(cover):
    for n, row in cover.items():
        assert 92 <= pct(row["coin_cover"]) <= 98, n
        assert 92 <= pct(row["adj_cover"]) <= 98, n
        assert row["coin_mean"] == pytest.approx(3.5, abs=0.15) and row["adj_mean"] == pytest.approx(3.5, abs=0.15)


def test_experiment_1_standard_error_shrinks_with_the_square_root_of_n(cover):
    assert cover[5000]["naive_se"] < cover[50]["naive_se"] / 8


# ----------------------------------------------------------------------------- Experiment 2

def test_experiment_2_five_of_sixteen_mixed_cells_are_negative(cells):
    mixed = [r for r in cells if r["gamma"] > 0 and r["beta"] > 0]
    assert len(mixed) == 16 and sum(r["limit"] < 0 for r in mixed) == 5


def test_experiment_2_no_error_without_selection_or_influence(cells):
    for r in cells:
        if r["gamma"] == 0.0 or r["beta"] == 0.0:
            assert r["limit"] == pytest.approx(3.5) and r["wrong"] == 0.0


def test_experiment_2_wrong_sign_shares_at_two_hundred_sites(cells):
    by = {(r["gamma"], r["beta"]): r for r in cells}
    assert pct(by[(1.0, 4.0)]["wrong"]) == pytest.approx(86, abs=3.5)
    assert pct(by[(1.0, 2.0)]["wrong"]) == pytest.approx(6, abs=3)
    assert pct(by[(2.0, 2.0)]["wrong"]) == pytest.approx(18, abs=4)
    assert pct(by[(2.0, 4.0)]["wrong"]) >= 98 and pct(by[(0.5, 4.0)]["wrong"]) == pytest.approx(22, abs=4)


# ----------------------------------------------------------------------------- Experiment 3

def test_experiment_3_forty_eight_tests_and_the_two_truths(mech):
    assert mech["tests"] == 48
    assert round(mech["below_05"] * mech["tests"]) == 5
    assert mech["min_p"] == pytest.approx(0.003, abs=0.002)
    assert mech["truth_A"] == 3.5 and mech["truth_B"] == pytest.approx(-1.01, abs=0.005)


def test_experiment_3_estimates_in_both_diagrams(mech):
    assert mech["A"]["naive"] == pytest.approx(-1.02, abs=0.1) and mech["A"]["adjusted"] == pytest.approx(3.64, abs=0.15)
    assert mech["B"]["naive"] == pytest.approx(-0.97, abs=0.1) and mech["B"]["adjusted"] == pytest.approx(3.57, abs=0.15)


# ----------------------------------------------------------------------------- Experiment 4 und Annahmen-Tabelle

def test_experiment_4_remaining_error_by_reliability(proxy):
    by = {r["r"]: r for r in proxy["rows"]}
    assert pct(by[0.9]["left"]) == pytest.approx(15, abs=2) and by[0.9]["estimate"] == pytest.approx(2.82, abs=0.1)
    assert by[0.5]["estimate"] == pytest.approx(0.74, abs=0.1) and pct(by[0.5]["left"]) == pytest.approx(60, abs=3)
    assert by[0.2]["estimate"] == pytest.approx(-0.41, abs=0.1) and pct(by[0.2]["left"]) == pytest.approx(86, abs=3)
    assert by[1.0]["estimate"] == pytest.approx(3.5, abs=0.15)


def test_overlap_numbers_of_the_assumptions_table():
    from kausal_generator import normal_cdf

    assert pct(float(normal_cdf(-2.0 * 1.0))) == pytest.approx(2.3, abs=0.1)
    assert pct(float(normal_cdf(-2.0 * 2.0))) == pytest.approx(0.003, abs=0.001)
