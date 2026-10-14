"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden
Slider an seine Grenzen und prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei st.plotly_chart ohne eindeutiges key=)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180
AUTOPLAY_MARKER = "▶"


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_all_sliders_at_min_and_at_max_together_do_not_raise():
    for edge in ("min", "max"):
        at = _fresh_app()
        for key in ("n_slider", "gamma_slider", "beta_slider", "tau_slider", "sigma_slider"):
            s = at.slider(key=key)
            s.set_value(s.min if edge == "min" else s.max)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    import kfd_constants as C

    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.slider(key="n_slider").value == preset["n"]
        assert abs(at.slider(key="gamma_slider").value - preset["gamma"]) < 1e-9
        assert abs(at.slider(key="beta_slider").value - preset["beta"]) < 1e-9
        assert abs(at.slider(key="tau_slider").value - preset["tau"]) < 1e-9
        assert abs(at.slider(key="sigma_slider").value - preset["sigma"]) < 1e-9
        assert at.selectbox(key="assign_select").value == preset["assign"]


def test_revealing_the_truth_adds_the_metrics_and_does_not_raise():
    at = _fresh_app()
    before = len(at.metric)
    at.toggle(key="reveal_truth").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.metric) == before + 3
    assert any(m.label == "Wahre Wirkung (Faktor festgesetzt)" for m in at.metric)


def test_the_diagram_switch_changes_the_explanation_and_does_not_raise():
    at = _fresh_app()
    assert any("Diagramm A" in md.value for md in at.markdown)
    at.radio(key="diagram_choice").set_value("B").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("In **Diagramm B**" in md.value for md in at.markdown)


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_clipped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "999999"
    at.query_params["g"] = "-3"
    at.query_params["b"] = "abc"
    at.query_params["z"] = "wuerfel"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == 5000
    assert at.slider(key="gamma_slider").value == 0.0
    assert at.slider(key="beta_slider").value == 4.0
    assert at.selectbox(key="assign_select").value == "disponent"


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.slider(key="gamma_slider").set_value(2.0).run(timeout=TIMEOUT)
    assert at.query_params["g"] == ["2.0"] or at.query_params["g"] == "2.0"
