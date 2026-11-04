"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset rechnet und hat das erwartete Verdikt."""

import pytest

import bst_constants as C
import bst_evaluation as ev
import bst_presets as P

STANDARD = "Standardfall (Voreinstellung)"


def _settings(name):
    return ev.Settings(**C.PRESETS[name])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        assert 0 <= preset["seed"] <= C.SEED_MAX and preset["stat"] in C.STATS and preset["netz"] in C.NETZE, name
        assert preset["sigma"] in C.SIGMA_CHOICES and preset["n"] in C.N_CHOICES and preset["B"] in C.B_CHOICES and preset["p"] in C.P_CHOICES and preset["phi"] in C.PHI_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[STANDARD]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_jedes_preset_unterscheidet_sich_vom_standard():
    for name, preset in C.PRESETS.items():
        if name != STANDARD:
            assert preset != C.PRESETS[STANDARD], name


def test_standardfall_zeigt_ein_perzentil_intervall_unter_dem_niveau():
    s = _settings(STANDARD)
    a = ev.analyse(s)
    assert a.rows[0]["res"]["perz"]["cov"] < 0.9 and a.rows[-1]["res"]["perz"]["cov"] > a.rows[0]["res"]["perz"]["cov"] + 0.08


def test_preset_schiefe_daten_zeigt_bootstrap_t_besser_und_breiter():
    s = _settings("Schiefe Daten, kleines n")
    r = next(x for x in ev.analyse(s).rows if x["n"] == s.n)["res"]
    assert s.sigma == 1.5 and r["perz"]["cov"] < 0.8 and r["boott"]["cov"] > r["perz"]["cov"] + 0.08 and r["boott"]["width"] > 5 * r["perz"]["width"]


def test_preset_standardabweichung():
    s = _settings("Standardabweichung")
    a = ev.analyse(s)
    r = next(x for x in a.rows if x["n"] == s.n)["res"]
    assert s.stat == "sd" and set(r) == {"perz", "basic", "bca"} and r["perz"]["cov"] < 0.75 and ev.verdict_main(a, s)[0] == "warning"


def test_preset_seltene_ausfaelle_hat_den_exakten_wert():
    s = _settings("Seltene Ausfälle")
    ex = ev.rare_experiment(s.netz, s.p)
    assert ex["pi"] == pytest.approx(0.0118, abs=1e-4) and ex["rows"][0]["perz"] < 0.35 and ex["rows"][0]["p0"] > 0.65


def test_preset_gedaechtnis_laesst_den_iid_bootstrap_scheitern():
    s = _settings("Daten mit Gedächtnis")
    r = ev.dep_experiment(s.phi, s.seed, 60)["rows"]
    assert s.phi == 0.95 and r["iid"]["cov"] < 0.4 and r["l50"]["cov"] > r["iid"]["cov"] + 0.3


def test_preset_zu_wenige_ziehungen():
    s = _settings("Zu wenige Ziehungen")
    ex = ev.b_experiment(s.stat, s.sigma, s.n, s.seed, 200)
    assert s.B == 99 and ex["rows"][2]["B"] == 99 and ex["rows"][-1]["perz"] > ex["rows"][0]["perz"]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_rechnet(name):
    a = ev.analyse(_settings(name))
    assert a.truth > 0 and len(a.rows) == len(C.COV_NS)
