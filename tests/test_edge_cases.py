"""Randfälle: kleinste Stichproben, konstante Daten, B = 1, null Ereignisse, Blocklänge ≥ n, Eingabeprüfung der Spielwiese, Zahlenformate."""

import numpy as np
import pytest

import bst_algorithm as A
import bst_constants as C
import bst_evaluation as ev
from bst_scenario import named_netz, parse_numbers, unreliability


def test_zwei_beobachtungen_rechnen_alle_intervalle():
    th, ints = A.boot_intervals(np.array([1.0, 3.0]), "mean", 99, 1)
    assert th == 2.0 and set(ints) == {"perz", "basic", "bca", "boott", "t"} and ints["t"][1] - ints["t"][0] > 10          # t mit einem Freiheitsgrad: Halbbreite 12,7 s/√2


def test_konstante_stichprobe_hat_breite_null_und_stuerzt_nicht_ab():
    x = np.full(20, 2.5)
    th, ints = A.boot_intervals(x, "mean", 99, 1, methods=("perz", "basic", "bca", "t"))
    for m, (lo, hi) in ints.items():
        assert lo == pytest.approx(2.5) and hi == pytest.approx(2.5), m
    th, ints = A.boot_intervals(x, "sd", 99, 1, methods=("perz", "bca"))
    assert th == 0.0 and ints["perz"] == (0.0, 0.0)


def test_eine_einzige_ziehung_gibt_einen_punkt():
    th, ints = A.boot_intervals(A.sample_lognormal(1.0, 20, 3), "mean", 1, 1, methods=("perz", "basic"))
    assert ints["perz"][0] == ints["perz"][1]


def test_bootstrap_t_ohne_streuung_der_ziehungen_wird_ausgelassen():
    x = np.full(10, 1.0)
    _, ints = A.boot_intervals(x, "mean", 50, 1, methods=("boott",))
    assert "boott" not in ints


def test_blocklaenge_groesser_als_n_wird_gekappt():
    x = np.arange(6.0)
    idx = A.block_indices(1, 5, 6, 100)
    assert idx.shape == (5, 6) and (idx == np.arange(6)[None, :]).all()


def test_keine_ereignisse_im_bernoulli_bootstrap():
    x = np.zeros(30)
    _, ints = A.boot_intervals(x, "mean", 99, 1, methods=("perz", "bca"))
    assert ints["perz"] == (0.0, 0.0)
    e = A.bernoulli_exact(0.0118, 30)
    assert e["p0"] == pytest.approx(0.7, abs=0.01) and e["degenerate"] == pytest.approx(e["p0"], abs=1e-9)


def test_bernoulli_exakt_mit_einer_beobachtung():
    e = A.bernoulli_exact(0.3, 1)
    assert e["perz"] == 0.0 and e["p0"] == pytest.approx(0.7) and e["degenerate"] == pytest.approx(1.0)


def test_exakte_bootstrap_verteilung_nur_bis_n_8():
    with pytest.raises(ValueError):
        A.exact_boot_mean(np.arange(9.0))
    v, p = A.exact_boot_mean(np.arange(8.0))
    assert p.sum() == pytest.approx(1.0)


def test_bca_bei_einer_stichprobe_mit_extremem_wert_bleibt_endlich():
    x = np.concatenate([np.ones(15), [1000.0]])
    lo, hi = A.boot_intervals(x, "mean", 499, 2, methods=("bca",))[1]["bca"]
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


def test_sehr_stark_abhaengige_daten_rechnen():
    r = ev.dep_experiment(0.95, 3, 40)["rows"]
    assert r["iid"]["cov"] < 0.5 and r["l50"]["cov"] > r["iid"]["cov"]


def test_netze_und_ausfallwahrscheinlichkeiten():
    for key in C.NETZE:
        assert named_netz(key)[1] >= 6
        us = [unreliability(key, p) for p in C.P_CHOICES]
        assert us == sorted(us, reverse=True) and 0 < us[-1] < us[0] < 1
    assert unreliability("3x3", 0.7) == pytest.approx(0.48901, abs=2e-5) and unreliability("3x3", 0.95) == pytest.approx(0.01184, abs=2e-5)


def test_eingabepruefung_der_spielwiese():
    assert parse_numbers("1,5 2;3\n4").tolist() == [1.5, 2.0, 3.0, 4.0]
    for text, frag in (("", "Keine Werte"), ("1 2", "Mindestens"), ("1 2 x", "keine Zahl"), ("1 2 inf", "keine endliche"), (" ".join(["1"] * 31), "Höchstens"), ("1 2 nan", "keine endliche")):
        with pytest.raises(ValueError, match=frag):
            parse_numbers(text)


def test_spielwiese_ergebnis_hat_exakte_verteilung_nur_bei_kleinem_n_und_mittelwert():
    r = ev.custom_result(np.array([1.0, 2.0, 3.0]), "mean", 1)
    assert r["exact"] is not None and r["exact"]["perz"] == (1.0, 3.0) and r["exact"]["mean"] == pytest.approx(2.0) and r["exact"]["var"] == pytest.approx(2 / 9)
    assert ev.custom_result(np.arange(1.0, 10.0), "mean", 1)["exact"] is None
    assert ev.custom_result(np.array([1.0, 2.0, 3.0]), "median", 1)["exact"] is None and set(ev.custom_result(np.array([1.0, 2.0, 3.0]), "median", 1)["ints"]) == {"perz", "basic", "bca"}


def test_zahlenformate():
    assert ev.de(1234.5, 1) == "1.234,5" and ev.de(-0.5, 1) == "−0,5"
    assert ev.pr(0.8764) == "0,876" and ev.pr(float("nan")) == "nicht bestimmbar" and ev.prozent(0.114, 1) == "11,4 %" and ev.dn(float("inf")) == "unendlich"
    assert ev.pi_text(0.0118) == "0,0118" and ev.pi_text(0.00027) == "0,00027"
