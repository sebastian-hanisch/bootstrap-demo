"""Eigenschaften, die für jede Wahl gelten müssen (nicht nur an einem Zahlenwert): Abdeckung gegen n, Intervallform, Statistiken, seltene Ereignisse, B, Abhängigkeit, Reproduzierbarkeit."""

import numpy as np
import pytest

import bst_algorithm as A
import bst_constants as C
import bst_evaluation as ev


def _cov(stat="mean", sigma=1.0, B=199, seed=3, R=300, ns=(10, 50)):
    return ev.coverage_experiment(stat, sigma, B, seed, R, ns)


def test_abdeckung_des_perzentil_intervalls_waechst_mit_n_und_liegt_bei_kleinem_n_unter_dem_niveau():
    rows = _cov(ns=(10, 50, 200), R=400)
    c = [r["res"]["perz"]["cov"] for r in rows]
    assert c[0] < c[1] < c[2] + 0.03 and c[0] < 0.88 and c[2] > 0.88


def test_fehltreffer_liegen_bei_schiefen_daten_einseitig():
    r = _cov(sigma=1.5, ns=(20,), R=400)[0]["res"]["perz"]
    assert r["right"] > 3 * r["left"] and r["cov"] + r["left"] + r["right"] == pytest.approx(1.0)


def test_schiefere_daten_sind_schwerer():
    c = [_cov(sigma=s, ns=(20,), R=400)[0]["res"]["perz"]["cov"] for s in (0.5, 1.0, 1.5)]
    assert c[0] > c[1] > c[2]


def test_bootstrap_t_deckt_besser_und_ist_breiter_als_das_perzentil_intervall():
    r = _cov(sigma=1.5, ns=(20,), R=400)[0]["res"]
    assert r["boott"]["cov"] > r["perz"]["cov"] + 0.04 and r["boott"]["width"] > 1.5 * r["perz"]["width"]
    assert r["basic"]["cov"] < r["perz"]["cov"] + 0.01


def test_basic_ist_die_spiegelung_des_perzentil_intervalls_und_alle_enthalten_das_mittel():
    x = A.sample_lognormal(1.0, 40, 4)
    th, ints = A.boot_intervals(x, "mean", 999, 5)
    lo, hi = ints["perz"]
    assert ints["basic"] == pytest.approx((2 * th - hi, 2 * th - lo))
    for m in ("perz", "basic", "bca", "boott", "t"):
        assert ints[m][0] < th < ints[m][1], m


def test_median_und_quantil_statistiken_haben_die_erwartete_reihenfolge_der_abdeckung():
    ex = ev.stat_experiment(1.0, 50, 199, 3, 300)
    r = {s: ex["rows"][s]["res"]["perz"]["cov"] for s in C.STATS}
    assert r["median"] > 0.9 and r["median"] > r["q90"] > r["sd"] and r["sd"] < 0.75
    assert ex["rows"]["q90"]["res"]["bca"]["cov"] > ex["rows"]["q90"]["res"]["perz"]["cov"]
    assert ex["max"]["perz"]["cov"] == 0.0 and ex["max"]["basic"]["cov"] > 0.8 and ex["max_same"] == pytest.approx(1 - (1 - 1 / 50) ** 50)


def test_seltene_ereignisse_exakt_perzentil_gleich_wald_und_wilson_besser():
    ex = ev.rare_experiment("3x3", 0.95)
    for r in ex["rows"][:3]:
        assert abs(r["perz"] - r["wald"]) < 0.01 and r["wilson"] > r["perz"] + 0.05
    perz = [r["perz"] for r in ex["rows"]]
    assert perz == sorted(perz) and ex["rows"][0]["p0"] == pytest.approx(ex["rows"][0]["degenerate"]) and ex["rows"][0]["p0"] > 0.6


def test_haeufige_ereignisse_exakt_alle_nahe_am_niveau():
    ex = ev.rare_experiment("3x3", 0.7)               # π = 0,489
    for r in ex["rows"]:
        assert 0.93 < r["perz"] < 0.975 and r["p0"] < 1e-6


def test_abdeckung_steigt_mit_der_zahl_der_ziehungen_und_die_endpunkte_werden_schaerfer():
    ex = ev.b_experiment("mean", 1.0, 50, 3, 300)
    rows = ex["rows"]
    assert rows[-1]["perz"] > rows[0]["perz"] + 0.04
    assert rows[0]["sd_hi"] > rows[-1]["sd_hi"] * 3 and all(rows[i]["sd_lo"] > rows[i + 1]["sd_lo"] - 0.005 for i in range(len(rows) - 1))


def test_blockbootstrap_deckt_abhaengige_daten_besser_als_iid_und_bei_phi_null_gleich():
    dep = ev.dep_experiment(0.8, 3, 100)["rows"]
    assert dep["l25"]["cov"] > dep["iid"]["cov"] + 0.2 and abs(dep["t"]["cov"] - dep["iid"]["cov"]) < 0.1 and dep["l25"]["width"] > 2 * dep["iid"]["width"]
    ind = ev.dep_experiment(0.0, 3, 100)["rows"]
    assert abs(ind["l25"]["cov"] - ind["iid"]["cov"]) < 0.1 and ind["iid"]["cov"] > 0.88


def test_gleiche_einstellung_gleiche_zahlen():
    a = ev.coverage_experiment("mean", 1.0, 99, 3, 100, (10,))
    assert a == ev.coverage_experiment("mean", 1.0, 99, 3, 100, (10,))
    x = A.sample_lognormal(1.0, 30, 2)
    assert A.boot_intervals(x, "mean", 200, 5) == A.boot_intervals(x, "mean", 200, 5) and A.boot_intervals(x, "mean", 200, 5) != A.boot_intervals(x, "mean", 200, 6)
