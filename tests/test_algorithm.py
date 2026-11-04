"""Handrechnung der Einheiten (DEMO-PLAYBOOK §1b): jede Zahl hier ist von Hand nachgerechnet, nicht aus dem Code gelesen."""

import math

import numpy as np
import pytest

import bst_algorithm as A

X3 = np.array([1.0, 2.0, 3.0])


def test_exakte_bootstrap_verteilung_des_mittels_von_hand():
    # n = 3: 27 gleich wahrscheinliche Ziehungen, Summen 3 bis 9 mit Häufigkeiten 1, 3, 6, 7, 6, 3, 1 ⇒ Mittel 1, 4/3, 5/3, 2, 7/3, 8/3, 3
    v, p = A.exact_boot_mean(X3)
    assert v == pytest.approx([1, 4 / 3, 5 / 3, 2, 7 / 3, 8 / 3, 3]) and p * 27 == pytest.approx([1, 3, 6, 7, 6, 3, 1])
    assert float((v * p).sum()) == pytest.approx(2.0)                                       # E* x̄* = x̄
    assert float((((v - 2) ** 2) * p).sum()) == pytest.approx(2 / 9)                        # Var* = σ̂²/n mit σ̂² = 2/3


def test_perzentil_und_basic_von_hand():
    v, p = A.exact_boot_mean(X3)
    lo, hi = A.discrete_quantile(v, p, 0.025), A.discrete_quantile(v, p, 0.975)
    assert (lo, hi) == (1.0, 3.0)                                       # kumuliert 1/27 > 0,025; 26/27 < 0,975 ⇒ 3
    assert (2 * 2.0 - hi, 2 * 2.0 - lo) == (1.0, 3.0)                  # basic


def test_bca_von_hand():
    # θ* = (1, …, 10), θ̂ = 4,5: Anteil θ* < θ̂ = 0,4 ⇒ z₀ = Φ⁻¹(0,4) = −0,2533; Jackknife (2,5; 2; 1,5) symmetrisch ⇒ a = 0;
    # korrigierte Anteile Φ(2z₀ ∓ 1,96): Φ(−2,4666) = 0,00682 und Φ(1,4534) = 0,92697; lineare Quantile 1 + 9 · 0,00682 = 1,0614 und 1 + 9 · 0,92697 = 9,3427
    ts = np.arange(1.0, 11.0)
    lo, hi = A.bca_interval(ts, 4.5, A.jackknife(X3, "mean"), 0.05)
    assert A.ndtri(0.4) == pytest.approx(-0.2533, abs=1e-3)
    assert lo == pytest.approx(1.0614, abs=0.01) and hi == pytest.approx(9.3427, abs=0.02)


def test_bca_bei_gleichstand_ist_z0_null_und_gleicht_dem_perzentil():
    # exakte Bootstrap-Verteilung von (1, 2, 3) als 27 Werte: Anteil θ* < 2 ist 10/27, mit halbem Gewicht der 7 Gleichstände genau ½ ⇒ z₀ = 0, a = 0 ⇒ BCa = Perzentil (lineare Quantile)
    ts = np.array([1.0] + [4 / 3] * 3 + [5 / 3] * 6 + [2.0] * 7 + [7 / 3] * 6 + [8 / 3] * 3 + [3.0])
    lo, hi = A.bca_interval(ts, 2.0, A.jackknife(X3, "mean"), 0.05)
    q = np.quantile(ts, [0.025, 0.975])
    assert lo == pytest.approx(float(q[0])) and hi == pytest.approx(float(q[1]))


def test_jackknife_von_hand():
    assert A.jackknife(X3, "mean") == pytest.approx([2.5, 2.0, 1.5])
    assert A.jackknife(X3, "median") == pytest.approx([2.5, 2.0, 1.5])               # Median von (2,3), (1,3), (1,2)


def test_t_intervall_von_hand():
    # x = (1, 2, 3): x̄ = 2, s = 1, t_{2; 0,975} = 4,3027 ⇒ Halbbreite 4,3027/√3 = 2,4843
    th, ints = A.boot_intervals(X3, "mean", 50, 1, methods=("t",))
    assert th == 2.0 and ints["t"][1] - 2 == pytest.approx(2.4843, abs=2e-4) and 2 - ints["t"][0] == pytest.approx(2.4843, abs=2e-4)


def test_t_verteilung_von_hand():
    assert A.t_cdf(1.0, 1) == pytest.approx(0.75) and A.t_cdf(1.0, 2) == pytest.approx(0.5 + 1 / (2 * math.sqrt(3)))
    assert A.t_quantile(0.975, 1) == pytest.approx(math.tan(math.pi * 0.475), rel=1e-9) and A.t_quantile(0.975, 1) == pytest.approx(12.706, abs=1e-3)
    assert A.t_cdf(-1.0, 3) == pytest.approx(1 - A.t_cdf(1.0, 3))


def test_normalverteilung_von_hand():
    assert A.ndtri(0.5) == pytest.approx(0.0, abs=1e-12) and A.ndtri(0.975) == pytest.approx(1.959964, abs=1e-6) and A.Phi(1.959964) == pytest.approx(0.975, abs=1e-6)


def test_blockindizes_von_hand(monkeypatch):
    # n = 4, l = 2: Start in {0, 1, 2}: u · 3 abgerundet; u = (0,1; 0,9) ⇒ Starts (0, 2) ⇒ Indizes (0,1,2,3) ⇒ Mittel 2,5
    monkeypatch.setattr(A, "uniforms", lambda seed, count: np.array([0.1, 0.9]))
    idx = A.block_indices(0, 1, 4, 2)
    assert idx.tolist() == [[0, 1, 2, 3]] and np.array([1.0, 2, 3, 4])[idx].mean() == 2.5
    monkeypatch.setattr(A, "uniforms", lambda seed, count: np.array([0.5, 0.5]))
    assert A.block_indices(0, 1, 4, 2).tolist() == [[1, 2, 1, 2]]


def test_bernoulli_exakt_von_hand():
    # n = 2, π = ½: S = 0 (¼): Intervall [0, 0]; S = 1 (½): Bin(2, ½)-Quantile 0 und 1 ⇒ [0, 1]; S = 2 (¼): [1, 1] ⇒ Abdeckung ½, Breite 0 mit ½, links ¼, rechts ¼
    e = A.bernoulli_exact(0.5, 2)
    assert e["perz"] == pytest.approx(0.5) and e["p0"] == pytest.approx(0.25) and e["degenerate"] == pytest.approx(0.5)
    assert e["left"] == pytest.approx(0.25) and e["right"] == pytest.approx(0.25)


def test_binomial_zaehldichte_von_hand():
    assert A.binom_pmf(3, 0.5) == pytest.approx([1 / 8, 3 / 8, 3 / 8, 1 / 8]) and A.binom_pmf(4, 0.0)[0] == 1.0 and A.binom_pmf(4, 1.0)[-1] == 1.0


def test_wahrheit_der_lognormalen_statistiken_von_hand():
    # σ = 1: Mittel e^½ = 1,6487; Median 1; 90-%-Quantil e^1,2816 = 3,6022; Standardabweichung √(e (e − 1)) = 2,1612
    assert A.lognormal_truth("mean", 1.0) == pytest.approx(1.6487, abs=1e-4) and A.lognormal_truth("median", 1.0) == 1.0
    assert A.lognormal_truth("q90", 1.0) == pytest.approx(3.6022, abs=1e-4) and A.lognormal_truth("sd", 1.0) == pytest.approx(2.1612, abs=1e-4)


def test_ar1_varianz_von_hand():
    # Stationäre Varianz 1/(1 − φ²): φ = ½ ⇒ 4/3
    x = np.concatenate([A.sample_ar1(0.5, 2000, s) for s in range(1, 41)])
    assert x.var() == pytest.approx(4 / 3, rel=0.05) and abs(x.mean()) < 0.05


def test_bootstrap_indizes_liegen_im_bereich_und_sind_reproduzierbar():
    idx = A.boot_indices(5, 20, 7)
    assert idx.shape == (20, 7) and idx.min() >= 0 and idx.max() <= 6 and np.array_equal(idx, A.boot_indices(5, 20, 7)) and not np.array_equal(idx, A.boot_indices(6, 20, 7))


def test_strom_ist_reproduzierbar_und_verschieden():
    assert A.stream_seed(3, 4) == A.stream_seed(3, 4) and A.stream_seed(3, 4) != A.stream_seed(4, 3)
    u = A.uniforms(A.stream_seed(1), 10000)
    assert u.min() > 0 and u.max() < 1 and abs(A.normals(A.stream_seed(2), 100000).mean()) < 0.02


def test_lanczos_gegen_math_lgamma():
    for x in (0.5, 1.5, 3.7, 20.5, 1000.5):
        assert float(A.lgam(x)) == pytest.approx(math.lgamma(x), rel=1e-12, abs=1e-12)
