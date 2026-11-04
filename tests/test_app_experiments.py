"""AppTest der Experimente, der Spielwiese, des PDF-Exports und des Permalinks (Einzelheiten, die der allgemeine Rauchtest nicht trifft)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import bst_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


@pytest.mark.parametrize("key,fragment", [
    ("run_stats", "Maximum einer Gleichverteilung: Perzentil"),
    ("run_b", "Die Endpunkte streuen von Lauf zu Lauf"),
    ("run_dep", "der iid-Bootstrap"),
])
def test_experiment_auf_klick_zeigt_sein_verdikt(key, fragment):
    at = _app()
    assert fragment not in _texte(at)                       # vor dem Klick steht noch nichts da
    at.button(key=key).click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert fragment in _texte(at)


def test_kernfrage_und_exakte_experimente_stehen_ohne_klick_da():
    t = _texte(_app())
    assert "Mittelwert log-normaler Daten (σ = 1,0), n = 50, B = 499, 500 Wiederholungen" in t
    assert "Ausfallwahrscheinlichkeit π = 0,0530: Abdeckung des Perzentil-Bootstraps" in t


def test_experiment_bleibt_nach_einer_aenderung_der_einstellungen_sichtbar():
    at = _app()
    at.button(key="run_dep").click().run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="phi_select").set_value(0.5).run(timeout=TIMEOUT)
    assert not at.exception
    assert "AR(1) mit φ = 0,50" in _texte(at)


def test_kernfrage_verdikt_wechselt_mit_dem_preset():
    at = _app()
    at.sidebar.button(key="preset_Standardabweichung").click().run(timeout=TIMEOUT)
    assert "Standardabweichung log-normaler Daten" in _texte(at)
    at.sidebar.button(key="preset_Seltene Ausfälle").click().run(timeout=TIMEOUT)
    assert "Mittelwert log-normaler Daten" in _texte(at)


def test_jede_statistik_laeuft_durch():
    at = _app()
    for stat in C.STATS:
        at.sidebar.selectbox(key="stat_select").set_value(stat).run(timeout=TIMEOUT)
        assert not at.exception, (stat, [str(e) for e in at.exception])


def test_spielwiese_exakte_verteilung_und_fehlermeldungen():
    at = _app()
    assert not at.error
    at.text_area(key="eigene_werte").set_value("1 2 x").run(timeout=TIMEOUT)
    assert not at.exception and "keine Zahl" in " ".join(e.value for e in at.error)
    at.text_area(key="eigene_werte").set_value("1 2").run(timeout=TIMEOUT)
    assert not at.exception and "Mindestens" in " ".join(e.value for e in at.error)
    at.text_area(key="eigene_werte").set_value("1 2 3 10 4").run(timeout=TIMEOUT)
    assert not at.exception and not at.error
    assert any("Exakte Bootstrap-Verteilung des Mittels" in c.value for c in at.caption)


def test_pdf_wird_erst_nach_klick_erzeugt():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1
    at.sidebar.select_slider(key="sigma_select").set_value(0.5).run(timeout=TIMEOUT)         # andere Einstellung: altes PDF nicht mehr anbieten
    assert not at.get("download_button")


def test_permalink_wird_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["sg"] = "1.2"
    at.query_params["n"] = "999"
    at.query_params["b"] = "500"
    at.query_params["s"] = "unbekannt"        # unbekannte Statistik: verworfen
    at.query_params["g"] = "hantel"
    at.query_params["seed"] = "-4"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.select_slider(key="sigma_select").value in (1.0, 1.5)                  # 1,2 liegt näher an 1,0
    assert at.sidebar.select_slider(key="n_select").value == 200                             # oberhalb des Größten: das Größte
    assert at.sidebar.select_slider(key="b_select").value == 499                             # 500 liegt näher an 499 als an 999
    assert at.sidebar.selectbox(key="stat_select").value == C.DEFAULT_STAT
    assert at.sidebar.selectbox(key="netz_select").value == "hantel"
    assert at.sidebar.number_input(key="seed_input").value == 0
