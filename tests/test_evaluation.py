"""Auswertung: Kennzahlen der Stichprobe, Struktur der Experimente (klein gerechnet), Verdikte an handgesetzten Werten."""

import numpy as np
import pytest

import kfd_constants as C
import kfd_evaluation as ev


def test_de_and_signed_use_decimal_comma_and_a_real_minus():
    assert ev.de(1.5, 1) == "1,5" and ev.de(-0.05, 2) == "−0,05"
    assert ev.signed(3.5, 1) == "+3,5" and ev.signed(-1.0135, 2) == "−1,01" and ev.signed(0.0, 1) == "+0,0"


def test_default_settings_are_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard (Disponent wählt nach Schwankung)"]) == ev.Settings()


def test_analysis_of_the_default_instance_is_internally_consistent():
    a = ev.analyse(ev.Settings())
    s = a.settings
    assert len(a.y) == s.n and set(np.unique(a.x)) <= {0.0, 1.0}
    assert a.share_treated == pytest.approx(a.x.mean())
    assert a.naive == pytest.approx(a.mean_treated - a.mean_untreated)
    assert a.lo < a.naive < a.hi
    assert a.truth == pytest.approx(C.DEFAULT_TAU)
    assert a.limit == pytest.approx(-1.0135, abs=1e-3)
    assert a.wrong_sign == (a.naive < 0.0) and a.covers == (a.lo <= a.truth <= a.hi)


def test_coin_assignment_has_the_unbiased_limit():
    a = ev.analyse(ev.Settings(assign="muenzwurf"))
    assert a.limit == C.DEFAULT_TAU and a.truth == C.DEFAULT_TAU


def test_without_selection_disponent_and_coin_agree_exactly():
    a = ev.analyse(ev.Settings(gamma=0.0))
    b = ev.analyse(ev.Settings(gamma=0.0, assign="muenzwurf"))
    assert a.naive == b.naive and a.y.tolist() == b.y.tolist()


def test_wrong_sign_is_false_for_zero_truth_and_for_matching_signs():
    a = ev.analyse(ev.Settings(tau=0.0))
    assert a.wrong_sign is False
    assert ev.analyse(ev.Settings(gamma=0.0)).wrong_sign is False


def test_a_tiny_sample_keeps_every_number_finite_or_infinite_se():
    for n in (C.N_MIN, 40):
        a = ev.analyse(ev.Settings(n=n, seed=3))
        assert np.isfinite(a.naive) and a.se > 0


def test_coverage_experiment_structure_and_direction():
    rows = ev.coverage_experiment(ns=(50, 400), reps=120)
    assert [r["n"] for r in rows] == [50, 400]
    for r in rows:
        assert 0.0 <= r["naive_cover"] <= 1.0 and 0.0 <= r["coin_cover"] <= 1.0 and 0.0 <= r["adj_cover"] <= 1.0
    assert rows[1]["naive_cover"] <= rows[0]["naive_cover"]
    assert rows[1]["coin_cover"] > 0.85 and rows[1]["naive_mean"] < 0.0 < rows[1]["coin_mean"]


def test_sign_map_has_one_row_per_cell_and_the_closed_form_limit():
    rows = ev.sign_map(gammas=(0.0, 1.0), betas=(0.0, 4.0), n=100, reps=60)
    assert len(rows) == 4
    by = {(r["gamma"], r["beta"]): r for r in rows}
    assert by[(0.0, 4.0)]["limit"] == pytest.approx(3.5) and by[(1.0, 0.0)]["limit"] == pytest.approx(3.5)
    assert by[(1.0, 4.0)]["limit"] < 0.0
    assert by[(1.0, 4.0)]["wrong"] > by[(0.0, 4.0)]["wrong"]


def test_mechanism_experiment_structure_and_truths():
    res = ev.mechanism_experiment(runs=2, n=3000)
    assert res["tests"] == 8 and len(res["p_values"]) == 8
    assert res["truth_A"] == C.DEFAULT_TAU and res["truth_B"] == pytest.approx(-1.0135, abs=1e-3)
    assert res["A"]["adjusted"] > 0.0 > res["A"]["naive"]
    assert res["B"]["naive"] == pytest.approx(res["truth_B"], abs=1.0)       # Standardfehler bei n = 3 000 rund 0,22


def test_proxy_experiment_estimates_fall_with_the_reliability():
    res = ev.proxy_experiment(n=20000)
    est = [r["estimate"] for r in res["rows"]]
    assert [r["r"] for r in res["rows"]] == list(C.EXP_PROXY_RELIABILITY)
    assert all(a > b for a, b in zip(est, est[1:]))
    assert res["rows"][0]["left"] < 0.1 and res["rows"][-1]["left"] > 0.7
    assert res["truth"] == C.DEFAULT_TAU and res["naive"] < 0.0


@pytest.mark.parametrize("settings,kind", [
    (ev.Settings(), "warning"),                                 # Standard: verfehlt, falsches Vorzeichen
    (ev.Settings(assign="muenzwurf"), "success"),
])
def test_instance_verdict_kinds(settings, kind):
    assert ev.verdict_instance(ev.analyse(settings))[0] == kind


def test_instance_verdict_info_when_the_interval_covers_by_chance_at_small_n():
    a = ev.analyse(ev.Settings(n=C.N_MIN, seed=1))
    if a.covers:
        assert ev.verdict_instance(a)[0] == "info"
    else:
        assert ev.verdict_instance(a)[0] == "warning"


def test_instance_verdict_success_without_bias():
    a = ev.analyse(ev.Settings(gamma=0.0, n=2000, seed=7))
    kind, text = ev.verdict_instance(a)
    assert kind == "success" if a.covers else kind in ("info", "warning")
    assert "wahre Wirkung" in text or "Intervall" in text


def test_verdict_texts_quote_the_numbers_they_are_given():
    rows = [{"n": 50, "naive_cover": 0.3, "coin_cover": 0.95}, {"n": 5000, "naive_cover": 0.0, "coin_cover": 0.94}]
    kind, text = ev.verdict_coverage(rows)
    assert kind == "warning" and "5000" in text and "0 %" in text and "94 %" in text
    rows_map = [{"gamma": 1.0, "beta": 4.0, "limit": -1.0}, {"gamma": 0.0, "beta": 4.0, "limit": 3.5}, {"gamma": 1.0, "beta": 1.0, "limit": 2.4}]
    assert "1 von 2 Zellen" in ev.verdict_map(rows_map)[1]
    res = {"tests": 48, "below_05": 5 / 48, "min_p": 0.003, "truth_A": 3.5, "truth_B": -1.0135}
    kind, text = ev.verdict_mechanism(res)
    assert "5 unter 0,05" in text and "+3,5" in text and "−1,0" in text
    kind, text = ev.verdict_proxy({"rows": [{"r": 0.9, "left": 0.149}]})
    assert kind == "warning" and "15 %" in text
