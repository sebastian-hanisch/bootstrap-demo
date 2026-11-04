"""Jede Zahl der README, nachgerechnet über dieselben Auswertungsfunktionen wie in der App (DEMO-PLAYBOOK §4).

Exakte Zahlen (Binomialverteilung, geschlossene Formeln) mit engen Bändern. Gemessene Zahlen stammen aus festen Strömen (Strom 39; 500 Wiederholungen je Stichprobengröße, 200 bei abhängigen Daten) und sind reproduzierbar;
sie stehen hier mit Bändern, die das Rauschen (Standardfehler der Abdeckung etwa 0,015, bei abhängigen Daten 0,03) abdecken."""

import pytest

import bst_algorithm as A
import bst_constants as C
import bst_evaluation as ev

S = ev.Settings()


def _near(x, soll, tol):
    return abs(x - soll) <= tol


def _a(**kw):
    return ev.analyse(ev.Settings(**kw))


def _res(a, n):
    return next(r for r in a.rows if r["n"] == n)["res"]


# --- exakte Werte ------------------------------------------------------------------------------------------------------------------------------

def test_exakte_wahrheiten_und_wahrscheinlichkeiten():
    assert A.lognormal_truth("mean", 1.0) == pytest.approx(1.6487, abs=5e-5) and A.lognormal_truth("mean", 1.5) == pytest.approx(3.0802, abs=5e-5)
    assert A.lognormal_truth("sd", 1.0) == pytest.approx(2.1612, abs=5e-5) and A.lognormal_truth("q90", 1.0) == pytest.approx(3.6022, abs=5e-5)
    from bst_scenario import unreliability
    for netz, p, soll in (("3x3", 0.95, 0.01184), ("3x3", 0.9, 0.05302), ("hantel", 0.9, 0.14969), ("3x3", 0.7, 0.48901)):
        assert unreliability(netz, p) == pytest.approx(soll, abs=2e-5)


def test_bernoulli_exakt():
    for p, rows in ((0.95, {30: (0.300, 0.951, 0.300, 0.700), 100: (0.695, 0.969, 0.695, 0.304), 300: (0.861, 0.972, 0.868, 0.028), 1000: (0.933, 0.945, 0.941, 0.0), 3000: (0.942, 0.958, 0.948, 0.0)}),
                    (0.9, {30: (0.801, 0.928, 0.801, 0.195), 100: (0.898, 0.956, 0.898, 0.004), 300: (0.941, 0.949, 0.948, 0.0), 1000: (0.948, 0.944, 0.948, 0.0), 3000: (0.948, 0.954, 0.948, 0.0)})):
        ex = ev.rare_experiment("3x3", p)
        for r in ex["rows"]:
            perz, wil, wald, p0 = rows[r["n"]]
            assert _near(r["perz"], perz, 0.002) and _near(r["wilson"], wil, 0.002) and _near(r["wald"], wald, 0.002) and _near(r["p0"], p0, 0.002), (p, r["n"])
    assert _near(ev.rare_experiment("hantel", 0.9)["rows"][0]["perz"], 0.942, 0.002) and _near(ev.rare_experiment("3x3", 0.7)["rows"][0]["perz"], 0.933, 0.002)


# --- Abdeckung des Mittelwerts -----------------------------------------------------------------------------------------------------------------------

MEAN1 = {"t": (0.862, 0.886, 0.878, 0.916, 0.938), "perz": (0.826, 0.872, 0.880, 0.916, 0.924), "basic": (0.802, 0.836, 0.850, 0.890, 0.924),
         "boott": (0.902, 0.920, 0.906, 0.934, 0.944), "bca": (0.844, 0.876, 0.884, 0.918, 0.928)}
MEAN15 = {"t": (0.754, 0.760, 0.800, 0.846, 0.882), "perz": (0.716, 0.750, 0.804, 0.854, 0.892), "basic": (0.672, 0.708, 0.770, 0.814, 0.858),
          "boott": (0.876, 0.872, 0.866, 0.898, 0.920), "bca": (0.772, 0.798, 0.822, 0.880, 0.896)}


@pytest.mark.parametrize("sigma,table", [(1.0, MEAN1), (1.5, MEAN15)])
def test_abdeckung_des_mittelwerts_gegen_n(sigma, table):
    a = _a(sigma=sigma)
    for m, soll in table.items():
        for n, s in zip(C.COV_NS, soll):
            assert _near(_res(a, n)[m]["cov"], s, 0.055), (sigma, m, n, _res(a, n)[m]["cov"], s)


def test_perzentil_deckt_nicht_besser_als_t_und_bootstrap_t_am_besten():
    for sigma in (1.0, 1.5):
        a = _a(sigma=sigma)
        for n in (10, 20, 50):
            r = _res(a, n)
            abstand = 0.03 if n <= 20 else 0.0        # bei n = 50, σ = 1 liegen Bootstrap-t (0,906) und Perzentil (0,880) nur 0,026 auseinander
            assert r["perz"]["cov"] <= r["t"]["cov"] + 0.04 and r["boott"]["cov"] > r["perz"]["cov"] + abstand and r["basic"]["cov"] < r["perz"]["cov"] + 0.01, (sigma, n)


def test_fehltreffer_liegen_einseitig_rechts():
    a = _a()
    r = _res(a, 10)["perz"]
    assert _near(r["right"], 0.164, 0.045) and _near(r["left"], 0.010, 0.02) and r["right"] > 6 * r["left"]
    assert _near(_res(a, 50)["perz"]["right"], 0.114, 0.04)


def test_breiten_bootstrap_t_gegen_perzentil():
    a, b = _a(), _a(sigma=1.5)
    r = _res(a, 10)
    assert _near(r["boott"]["width"] / r["perz"]["width"], 2.36, 0.25) and _near(r["perz"]["width"], 1.94, 0.15) and _near(r["boott"]["width"], 4.58, 0.5)
    r = _res(b, 10)
    assert _near(r["boott"]["width"] / r["perz"]["width"], 5.6, 1.2) and _near(r["boott"]["width"], 29.3, 6.0)
    assert _near(_res(b, 20)["boott"]["width"] / _res(b, 20)["perz"]["width"], 2.9, 0.5)


def test_weniger_schiefe_ist_leichter():
    a = _a(sigma=0.5)
    assert _res(a, 20)["perz"]["cov"] > 0.9 and _near(_res(a, 10)["perz"]["cov"], 0.880, 0.055) and _near(_res(a, 10)["t"]["cov"], 0.922, 0.055) and _near(_res(a, 50)["perz"]["cov"], 0.922, 0.055)


# --- Statistiken -----------------------------------------------------------------------------------------------------------------------------------

def test_statistiken_bei_n_50():
    ex = ev.stat_experiment(1.0, 50, 499, S.seed)
    for stat, soll in (("mean", (0.882, 0.856, 0.910)), ("median", (0.946, 0.854, 0.944)), ("q90", (0.884, 0.748, 0.926)), ("sd", (0.586, 0.614, 0.670))):
        for m, s in zip(("perz", "basic", "bca"), soll):
            assert _near(ex["rows"][stat]["res"][m]["cov"], s, 0.055), (stat, m)
    assert ex["max"]["perz"]["cov"] == 0.0 and _near(ex["max"]["basic"]["cov"], 0.860, 0.05) and ex["max_same"] == pytest.approx(1 - (1 - 1 / 50) ** 50) and _near(ex["max_same"], 0.636, 0.001)
    ex2 = ev.stat_experiment(1.5, 10, 499, S.seed)
    assert _near(ex2["rows"]["sd"]["res"]["perz"]["cov"], 0.238, 0.06) and _near(ex2["rows"]["median"]["res"]["perz"]["cov"], 0.966, 0.04)


def test_quantil_und_standardabweichung_gegen_n():
    for stat, soll in (("q90", (0.688, 0.860, 0.880, 0.928, 0.932)), ("sd", (0.412, 0.536, 0.582, 0.682, 0.742))):
        a = _a(stat=stat, B=999)
        for n, s in zip(C.COV_NS, soll):
            assert _near(_res(a, n)["perz"]["cov"], s, 0.06), (stat, n)
    q = _a(stat="q90", B=999)
    assert _res(q, 50)["bca"]["cov"] > _res(q, 50)["perz"]["cov"] + 0.02


# --- Zahl der Ziehungen und Abhängigkeit -------------------------------------------------------------------------------------------------------------

def test_zahl_der_ziehungen():
    ex = ev.b_experiment("mean", 1.0, 50, S.seed)
    rows = {r["B"]: r for r in ex["rows"]}
    assert rows[19]["perz"] < rows[999]["perz"] - 0.03 and rows[19]["bca"] < rows[999]["bca"] - 0.05
    for B, lo, hi in ((19, 0.103, 0.154), (99, 0.045, 0.076), (999, 0.015, 0.026)):
        assert _near(rows[B]["sd_lo"], lo, 0.02) and _near(rows[B]["sd_hi"], hi, 0.03), B
    assert rows[999]["sd_hi"] < rows[99]["sd_hi"] < rows[19]["sd_hi"] and rows[999]["perz"] > 0.85


def test_abhaengige_daten():
    for phi, tab in ((0.0, {"t": 0.960, "iid": 0.950, "l25": 0.915}), (0.5, {"t": 0.695, "iid": 0.695, "l25": 0.905}), (0.8, {"t": 0.535, "iid": 0.510, "l5": 0.810, "l10": 0.855, "l25": 0.900, "l50": 0.890}),
                     (0.95, {"t": 0.230, "iid": 0.230, "l5": 0.465, "l10": 0.615, "l25": 0.760, "l50": 0.815})):
        r = ev.dep_experiment(phi, S.seed)["rows"]
        for k, soll in tab.items():
            assert _near(r[k]["cov"], soll, 0.09), (phi, k, r[k]["cov"], soll)
    r = ev.dep_experiment(0.95, S.seed)["rows"]
    assert _near(r["l50"]["width"] / r["iid"]["width"], 4.65, 0.8) and _near(r["iid"]["width"], 0.528, 0.04)
    r = ev.dep_experiment(0.8, S.seed)["rows"]
    assert _near(r["l25"]["width"], 0.751, 0.08) and _near(r["iid"]["width"], 0.284, 0.03)
    assert ev.dep_experiment(0.8, S.seed)["var_factor"] == pytest.approx(25.0)
