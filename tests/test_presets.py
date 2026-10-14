"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen."""

import pytest

import kfd_constants as C
import kfd_evaluation as ev
import kfd_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_on_the_step_grid():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0, name
        assert C.GAMMA_MIN <= p["gamma"] <= C.GAMMA_MAX and C.BETA_MIN <= p["beta"] <= C.BETA_MAX, name
        assert C.TAU_MIN <= p["tau"] <= C.TAU_MAX and C.SIGMA_MIN <= p["sigma"] <= C.SIGMA_MAX, name
        assert p["assign"] in C.ASSIGN_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name
        for key, lo, step in (("gamma", C.GAMMA_MIN, C.GAMMA_STEP), ("beta", C.BETA_MIN, C.BETA_STEP), ("tau", C.TAU_MIN, C.TAU_STEP),
                              ("sigma", C.SIGMA_MIN, C.SIGMA_STEP)):
            assert (p[key] - lo) / step == pytest.approx(round((p[key] - lo) / step)), (name, key)


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standard (Disponent wählt nach Schwankung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_default_is_a_valid_widget_value():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_the_presets_show_different_outcomes():
    naive = {name: ev.analyse(ev.Settings(**p)).limit for name, p in C.PRESETS.items()}
    assert naive["Standard (Disponent wählt nach Schwankung)"] < 0.0 < naive["Effekt fast unsichtbar"] < naive["Münzwurf statt Auswahl"]
    assert naive["Fünftausend Standorte"] == naive["Standard (Disponent wählt nach Schwankung)"]
    assert ev.analyse(ev.Settings(**C.PRESETS["Fünftausend Standorte"])).covers is False
    assert ev.analyse(ev.Settings(**C.PRESETS["Standard (Disponent wählt nach Schwankung)"])).wrong_sign is True


def test_sweep_constants_are_consistent():
    assert len(set(C.EXP_N_SWEEP)) == len(C.EXP_N_SWEEP) and C.EXP_REPS >= 200
    assert C.EXP_PROXY_RELIABILITY[0] == 1.0 and 0.9 in C.EXP_PROXY_RELIABILITY
    assert 1.0 in C.EXP_MAP_GAMMAS and 4.0 in C.EXP_MAP_BETAS
