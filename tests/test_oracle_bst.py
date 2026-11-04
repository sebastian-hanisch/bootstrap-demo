"""Unabhängige Gegenproben (scipy): Normal-, t- und Binomialverteilung, Lognormal, BCa, exakte Bootstrap-Verteilung gegen Simulation; ORACLE-PLAYBOOK §5."""

import math

import numpy as np
import pytest
from scipy import special, stats

import bst_algorithm as A


def test_inverse_normalverteilung_gegen_scipy():
    for p in (1e-10, 1e-6, 1e-3, 0.01, 0.025, 0.2, 0.5, 0.8, 0.975, 0.999, 1 - 1e-8):
        assert A.ndtri(p) == pytest.approx(float(special.ndtri(p)), rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("df", [1, 2, 3, 4, 5, 9, 19, 49, 99, 199])
def test_t_verteilung_gegen_scipy(df):
    for t in (0.0, 0.3, 1.0, 2.5, -1.7, 6.0):
        assert A.t_cdf(t, df) == pytest.approx(float(stats.t.cdf(t, df)), abs=1e-12)
    assert A.t_quantile(0.975, df) == pytest.approx(float(stats.t.ppf(0.975, df)), rel=1e-9)
    assert A.t_quantile(0.9, df) == pytest.approx(float(stats.t.ppf(0.9, df)), rel=1e-9) and A.t_quantile(0.1, df) == pytest.approx(-A.t_quantile(0.9, df))


@pytest.mark.parametrize("n,p", [(10, 0.3), (50, 0.05), (300, 0.5), (1000, 0.0118)])
def test_binomial_zaehldichte_gegen_scipy(n, p):
    assert A.binom_pmf(n, p) == pytest.approx(stats.binom.pmf(np.arange(n + 1), n, p), abs=1e-12)


@pytest.mark.parametrize("pi,n", [(0.0118, 30), (0.0118, 300), (0.053, 100), (0.15, 30), (0.4, 100)])
def test_bernoulli_exakt_gegen_scipy_quantile(pi, n):
    S = np.arange(n + 1)
    pm = stats.binom.pmf(S, n, pi)
    plo, phi = stats.binom.ppf(0.025, n, S / n) / n, stats.binom.ppf(0.975, n, S / n) / n
    ref = float(pm[(plo <= pi) & (pi <= phi)].sum())
    e = A.bernoulli_exact(pi, n)
    assert e["perz"] == pytest.approx(ref, abs=1e-9)
    assert e["p0"] == pytest.approx((1 - pi) ** n, rel=1e-9)


def test_bernoulli_exakt_gegen_werte_der_vormessung():
    # exakte Abdeckung aus der Vormessung (Plan 15): π = 0,0118, n = 30 / 100 / 300 / 1000
    for n, soll in ((30, 0.299), (100, 0.694), (300, 0.860), (1000, 0.932)):
        assert A.bernoulli_exact(0.0118, n)["perz"] == pytest.approx(soll, abs=2e-3)


def test_bernoulli_exakt_gegen_simulation_mit_endlichem_b():
    # Perzentil-Bootstrap mit B = 999 an echten Bernoulli-Stichproben stimmt mit der exakten Abdeckung (B → ∞) bis auf das Simulationsrauschen überein
    pi, n, R = 0.15, 40, 1500
    hits = 0
    for r in range(R):
        x = A.sample_bernoulli(pi, n, A.stream_seed(7, r))
        lo, hi = A.boot_intervals(x, "mean", 999, A.stream_seed(8, r), methods=("perz",))[1]["perz"]
        hits += lo <= pi <= hi
    assert hits / R == pytest.approx(A.bernoulli_exact(pi, n)["perz"], abs=0.03)


def test_lognormal_wahrheit_gegen_scipy():
    for sigma in (0.5, 1.0, 1.5):
        d = stats.lognorm(s=sigma, scale=1.0)
        assert A.lognormal_truth("mean", sigma) == pytest.approx(d.mean(), rel=1e-12) and A.lognormal_truth("sd", sigma) == pytest.approx(d.std(), rel=1e-12)
        assert A.lognormal_truth("median", sigma) == pytest.approx(d.median()) and A.lognormal_truth("q90", sigma) == pytest.approx(d.ppf(0.9), rel=1e-12)


def test_exakte_bootstrap_verteilung_gegen_simulation():
    x = np.array([0.5, 1.0, 4.0, 2.0])
    v, p = A.exact_boot_mean(x)
    assert p.sum() == pytest.approx(1.0) and float((v * p).sum()) == pytest.approx(x.mean())
    assert float((((v - x.mean()) ** 2) * p).sum()) == pytest.approx(x.var() / len(x))
    ts = x[A.boot_indices(11, 200000, len(x))].mean(axis=1)
    assert ts.mean() == pytest.approx(x.mean(), abs=0.003) and ts.var() == pytest.approx(x.var() / len(x), rel=0.02)
    for q in (0.1, 0.5, 0.9):                                                  # Verteilungsfunktion an den Atomen
        assert float((ts <= np.quantile(x, 0.5) + 1e-9).mean()) == pytest.approx(float(p[v <= np.quantile(x, 0.5) + 1e-9].sum()), abs=0.01)


def test_bootstrap_varianz_des_mittels_gleich_sigma_dach_quadrat_durch_n():
    x = A.sample_lognormal(1.0, 60, 3)
    ts = x[A.boot_indices(5, 40000, len(x))].mean(axis=1)
    assert ts.var() == pytest.approx(x.var() / len(x), rel=0.03)


def test_bca_gegen_scipy_bootstrap():
    x = A.sample_lognormal(1.0, 40, 9)
    th, ints = A.boot_intervals(x, "mean", 20000, 3, methods=("bca", "perz"))
    ref = stats.bootstrap((x,), np.mean, n_resamples=20000, confidence_level=0.95, method="BCa", random_state=1).confidence_interval
    width = ints["bca"][1] - ints["bca"][0]
    assert abs(ints["bca"][0] - ref.low) < 0.04 * width and abs(ints["bca"][1] - ref.high) < 0.04 * width
    refp = stats.bootstrap((x,), np.mean, n_resamples=20000, confidence_level=0.95, method="percentile", random_state=1).confidence_interval
    assert abs(ints["perz"][0] - refp.low) < 0.04 * width and abs(ints["perz"][1] - refp.high) < 0.04 * width


def test_t_intervall_gegen_scipy():
    x = A.sample_lognormal(1.0, 25, 2)
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    got = A.boot_intervals(x, "mean", 10, 1, methods=("t",))[1]["t"]
    assert got[0] == pytest.approx(lo, rel=1e-9) and got[1] == pytest.approx(hi, rel=1e-9)


def test_t_intervall_deckt_bei_normaldaten_95_prozent():
    R, hits = 3000, 0
    for r in range(R):
        x = A.normals(A.stream_seed(21, r), 15)
        lo, hi = A.boot_intervals(x, "mean", 10, 1, methods=("t",))[1]["t"]
        hits += lo <= 0 <= hi
    assert hits / R == pytest.approx(0.95, abs=3 * math.sqrt(0.05 * 0.95 / R))


def test_maximum_der_gleichverteilung_haelt_dasselbe_maximum_in_63_prozent():
    for n in (20, 50, 200):
        x = A.sample_uniform(n, 3)
        same = float((x[A.boot_indices(4, 20000, n)].max(axis=1) == x.max()).mean())
        assert same == pytest.approx(1 - (1 - 1 / n) ** n, abs=0.01)


def test_ar1_varianz_des_mittels_gegen_langzeitformel():
    n, phi = 500, 0.5
    means = np.array([A.sample_ar1(phi, n, A.stream_seed(31, r)).mean() for r in range(3000)])
    assert means.var() == pytest.approx((1 / n) / (1 - phi) ** 2, rel=0.1)


def test_blockbootstrap_varianz_naeher_an_der_langzeitvarianz_als_iid():
    n, phi = 500, 0.8
    x = A.sample_ar1(phi, n, 5)
    true = (1 / n) / (1 - phi) ** 2
    iid = x[A.boot_indices(6, 4000, n)].mean(axis=1).var()
    blk = x[A.block_indices(7, 4000, n, 50)].mean(axis=1).var()
    assert iid < 0.3 * true and blk > 2 * iid
