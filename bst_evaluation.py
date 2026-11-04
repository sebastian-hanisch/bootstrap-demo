"""Kennzahlen, Verdikt-Logik und Experimente über feste Ströme (reine Funktionen, kein Streamlit).

Gemessen wird, wann Resampling die Verteilung ersetzt: Abdeckung der 95-%-Intervalle (Wahrheit exakt bekannt) gegen Stichprobengröße, Statistik, seltene Ereignisse, Zahl der Ziehungen und Abhängigkeit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import bst_algorithm as A
import bst_constants as C
from bst_scenario import unreliability


def de(x: float, nd: int = 2) -> str:
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen, echtes Minuszeichen."""
    return f"{x:,.{nd}f}".translate(str.maketrans(",.", ".,")).replace("-", "−")


def dn(x: float, nd: int = 2) -> str:
    if np.isnan(x):
        return "nicht bestimmbar"
    if np.isinf(x):
        return "unendlich"
    return de(x, nd)


def pr(x: float) -> str:
    """Anteil mit drei Nachkommastellen."""
    return "nicht bestimmbar" if np.isnan(x) else de(x, 3)


def prozent(x: float, nd: int = 0) -> str:
    return "nicht bestimmbar" if np.isnan(x) else f"{de(x * 100, nd)} %"


def pi_text(x: float) -> str:
    return de(x, 4 if x >= 0.001 else 5)


@dataclass(frozen=True)
class Settings:
    stat: str = C.DEFAULT_STAT
    sigma: float = C.DEFAULT_SIGMA
    n: int = C.DEFAULT_N
    B: int = C.DEFAULT_B
    netz: str = C.DEFAULT_NETZ
    p: float = C.DEFAULT_P
    phi: float = C.DEFAULT_PHI
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    label: str
    stat: str
    sigma: float
    truth: float
    B: int
    R: int
    rows: tuple                              # je n {n, res: {Methode: {cov, left, right, width, degenerate}}}


def methods_for(stat: str) -> tuple:
    return C.METHODS if stat == "mean" else ("perz", "basic", "bca")


def label_of(settings: Settings) -> str:
    return f"{C.STATS[settings.stat]} log-normaler Daten (σ = {de(settings.sigma, 1)})"


# --- Abdeckung ------------------------------------------------------------------------------------------------------------------------------

def coverage(sampler, stat: str, truth: float, n: int, R: int, B: int, seed: int, tag: int, methods=C.METHODS, alpha: float = C.ALPHA) -> dict:
    """Abdeckung, Fehltreffer (links: Wahrheit unter dem Intervall, das Intervall liegt zu hoch; rechts: Wahrheit über dem Intervall, das Intervall liegt zu niedrig),
    mittlere Breite und Anteil mit Breite 0 je Methode über R Stichproben; `sampler(n, seed)` liefert eine Stichprobe."""
    res = {m: [0, 0, 0, 0.0, 0] for m in methods}
    for r in range(R):
        x = sampler(n, A.stream_seed(seed, tag, n, r))
        _, ints = A.boot_intervals(x, stat, B, A.stream_seed(seed, tag + 1, n, r), alpha, methods)
        for m, (lo, hi) in ints.items():
            c = res[m]
            c[0] += int(lo <= truth <= hi)
            c[1] += int(truth < lo)
            c[2] += int(truth > hi)
            c[3] += hi - lo
            c[4] += int(hi - lo <= 1e-12)
    return {m: {"cov": c[0] / R, "left": c[1] / R, "right": c[2] / R, "width": c[3] / R, "degenerate": c[4] / R} for m, c in res.items()}


@lru_cache(maxsize=64)
def coverage_experiment(stat: str, sigma: float, B: int, seed: int, R: int = C.COV_R, ns: tuple = C.COV_NS) -> tuple:
    """Abdeckung gegen die Stichprobengröße für log-normale Daten: je n die Ergebnisse aller Intervalle."""
    truth = A.lognormal_truth(stat, sigma)
    sampler = lambda n, s: A.sample_lognormal(sigma, n, s)
    return tuple({"n": n, "res": coverage(sampler, stat, truth, n, R, B, seed, 1, methods_for(stat))} for n in ns)


def analyse(settings: Settings) -> Analysis:
    rows = coverage_experiment(settings.stat, settings.sigma, settings.B, settings.seed)
    return Analysis(label_of(settings), settings.stat, settings.sigma, A.lognormal_truth(settings.stat, settings.sigma), settings.B, C.COV_R, rows)


# --- Experiment 2: Statistiken ------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=32)
def stat_experiment(sigma: float, n: int, B: int, seed: int, R: int = C.STAT_R) -> dict:
    """Abdeckung von Perzentil, basic und BCa für Mittelwert, Median, 90-%-Quantil und Standardabweichung log-normaler Daten und für das Maximum einer Gleichverteilung auf (0, 1) (Wahrheit 1)."""
    sampler = lambda m, s: A.sample_lognormal(sigma, m, s)
    rows = {}
    for i, stat in enumerate(C.STATS):
        rows[stat] = {"truth": A.lognormal_truth(stat, sigma), "res": coverage(sampler, stat, A.lognormal_truth(stat, sigma), n, R, B, seed, 10 + 2 * i, ("perz", "basic", "bca"))}
    mx = coverage(lambda m, s: A.sample_uniform(m, s), "max", 1.0, n, R, B, seed, 30, ("perz", "basic"))
    return {"sigma": sigma, "n": n, "B": B, "R": R, "rows": rows, "max": mx, "max_same": 1 - (1 - 1 / n) ** n}


# --- Experiment 3: seltene Ereignisse (exakt) ----------------------------------------------------------------------------------------------------

def rare_experiment(netz: str, p: float) -> dict:
    """Exakte Abdeckung (ohne Simulation) von Perzentil-Bootstrap (B → ∞), Wilson und Wald für die Ausfallwahrscheinlichkeit des Netzes bei n = 30 … 3000 Beobachtungen."""
    pi = unreliability(netz, p)
    return {"pi": pi, "rows": [{"n": n, **A.bernoulli_exact(pi, n)} for n in C.RARE_NS]}


# --- Experiment 4: Zahl der Ziehungen -------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=32)
def b_experiment(stat: str, sigma: float, n: int, seed: int, R: int = C.B_R) -> dict:
    """Abdeckung (Perzentil, BCa) und Monte-Carlo-Streuung der Perzentil-Endpunkte (gleiche Daten, B_ENDPOINT_RUNS unabhängige Bootstrap-Läufe, Mittel über B_ENDPOINT_DATASETS Datensätze, relativ zur Intervallbreite) gegen B."""
    truth = A.lognormal_truth(stat, sigma)
    sampler = lambda m, s: A.sample_lognormal(sigma, m, s)
    datasets = [sampler(n, A.stream_seed(seed, 50, n, i)) for i in range(C.B_ENDPOINT_DATASETS)]
    rows = []
    for B in C.B_LIST:
        res = coverage(sampler, stat, truth, n, R, B, seed, 40 + B, ("perz", "bca"))
        sd_lo, sd_hi = [], []
        for i, x in enumerate(datasets):
            lo, hi = [], []
            for k in range(C.B_ENDPOINT_RUNS):
                _, ints = A.boot_intervals(x, stat, B, A.stream_seed(seed, 60, B, 1000 * i + k), methods=("perz",))
                lo.append(ints["perz"][0]), hi.append(ints["perz"][1])
            w = np.mean(hi) - np.mean(lo)
            sd_lo.append(np.std(lo, ddof=1) / w if w > 0 else float("nan")), sd_hi.append(np.std(hi, ddof=1) / w if w > 0 else float("nan"))
        rows.append({"B": B, "perz": res["perz"]["cov"], "bca": res["bca"]["cov"], "sd_lo": float(np.nanmean(sd_lo)), "sd_hi": float(np.nanmean(sd_hi))})
    return {"stat": stat, "sigma": sigma, "n": n, "R": R, "rows": rows}


# --- Experiment 5: abhängige Daten ---------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=32)
def dep_experiment(phi: float, seed: int, R: int = C.DEP_R) -> dict:
    """AR(1) mit Koeffizient φ, n = DEP_N, Wahrheit μ = 0: Abdeckung (und mittlere Breite) des t-Intervalls, des iid-Bootstraps und des gleitenden Blockbootstraps für die Blocklängen DEP_BLOCKS."""
    n, B = C.DEP_N, C.DEP_B
    keys = ["t", "iid"] + [f"l{l}" for l in C.DEP_BLOCKS]
    cnt = {k: [0, 0.0] for k in keys}
    for r in range(R):
        x = A.sample_ar1(phi, n, A.stream_seed(seed, 70, int(phi * 100), r))
        m = float(x.mean())
        h = A.t_quantile(0.975, n - 1) * float(x.std(ddof=1)) / math.sqrt(n)
        cnt["t"][0] += int(m - h <= 0 <= m + h)
        cnt["t"][1] += 2 * h
        _, ints = A.boot_intervals(x, "mean", B, A.stream_seed(seed, 71, int(phi * 100), r), methods=("perz",))
        lo, hi = ints["perz"]
        cnt["iid"][0] += int(lo <= 0 <= hi)
        cnt["iid"][1] += hi - lo
        for l in C.DEP_BLOCKS:
            lo, hi = A.block_interval(x, l, B, A.stream_seed(seed, 72, int(phi * 100) * 100 + l, r))
            cnt[f"l{l}"][0] += int(lo <= 0 <= hi)
            cnt[f"l{l}"][1] += hi - lo
    return {"phi": phi, "n": n, "R": R, "B": B, "rows": {k: {"cov": c[0] / R, "width": c[1] / R} for k, c in cnt.items()},
            "var_factor": 1.0 / (1 - phi) ** 2}


# --- Spielwiese ------------------------------------------------------------------------------------------------------------------------------------

def custom_result(x: np.ndarray, stat: str, seed: int) -> dict:
    """Alle Intervalle für eine eigene kleine Stichprobe (CUSTOM_B Ziehungen); für n ≤ CUSTOM_EXACT_N zusätzlich die exakte Bootstrap-Verteilung des Mittels und ihr Perzentil-Intervall."""
    th, ints = A.boot_intervals(x, stat, C.CUSTOM_B, A.stream_seed(seed, 80, len(x)), methods=methods_for(stat))
    out = {"theta": th, "ints": ints, "n": len(x), "exact": None}
    if stat == "mean" and len(x) <= C.CUSTOM_EXACT_N:
        vals, probs = A.exact_boot_mean(x)
        out["exact"] = {"values": vals, "probs": probs, "perz": (A.discrete_quantile(vals, probs, C.ALPHA / 2), A.discrete_quantile(vals, probs, 1 - C.ALPHA / 2)),
                        "mean": float((vals * probs).sum()), "var": float((((vals - th) ** 2) * probs).sum())}
    return out


# --- Verdikte -----------------------------------------------------------------------------------------------------------------------------

def _row(a: Analysis, n: int) -> dict:
    return next(r for r in a.rows if r["n"] == n)


def verdict_main(a: Analysis, settings: Settings) -> tuple[str, str]:
    """Kernfrage: Darf man dem Bootstrap-Intervall bei dieser Stichprobengröße trauen?"""
    r = _row(a, settings.n)["res"]
    first, last = a.rows[0], a.rows[-1]
    parts = ", ".join(f"{C.METHOD_LABELS[m]} {pr(r[m]['cov'])}" for m in methods_for(a.stat) if m in r)
    text = (f"{a.label}, n = {settings.n}, B = {settings.B}, {a.R} Wiederholungen: Abdeckung {parts} (nominell 0,95). Das Perzentil-Intervall steigt von {pr(first['res']['perz']['cov'])} bei n = {first['n']} auf {pr(last['res']['perz']['cov'])} bei n = {last['n']}. "
            f"Seine Fehltreffer liegen bei n = {settings.n} in {prozent(r['perz']['right'], 1)} der Läufe auf der einen Seite (Wahrheit über dem Intervall) und in {prozent(r['perz']['left'], 1)} auf der anderen.")
    if "t" in r:
        text += f" Das t-Intervall deckt {pr(r['t']['cov'])}: " + ("der Perzentil-Bootstrap ist nicht besser." if r["perz"]["cov"] <= r["t"]["cov"] + 0.02 else "der Perzentil-Bootstrap ist hier besser.")
    cov = r["perz"]["cov"]
    return ("warning" if cov < 0.9 else "info" if cov < 0.93 else "success"), text


def verdict_stats(ex: dict) -> tuple[str, str]:
    r = ex["rows"]
    parts = "; ".join(f"{C.STATS[s]} {pr(r[s]['res']['perz']['cov'])} / {pr(r[s]['res']['basic']['cov'])} / {pr(r[s]['res']['bca']['cov'])}" for s in C.STATS)
    mx = ex["max"]
    text = (f"Abdeckung Perzentil / basic / BCa bei n = {ex['n']} (σ = {de(ex['sigma'], 1)}, B = {ex['B']}): {parts}. Maximum einer Gleichverteilung: Perzentil {pr(mx['perz']['cov'])}, basic {pr(mx['basic']['cov'])}; "
            f"{prozent(ex['max_same'], 0)} der Bootstrap-Stichproben haben dasselbe Maximum wie die Stichprobe und keine zieht darüber hinaus.")
    worst = min(r[s]["res"]["perz"]["cov"] for s in C.STATS)
    return ("warning" if worst < 0.85 or mx["perz"]["cov"] < 0.5 else "info"), text


def verdict_rare(ex: dict) -> tuple[str, str]:
    rows = ex["rows"]
    first, last = rows[0], rows[-1]
    text = (f"Ausfallwahrscheinlichkeit π = {pi_text(ex['pi'])}: Abdeckung des Perzentil-Bootstraps bei n = {first['n']} / {rows[1]['n']} / {rows[2]['n']} / {last['n']}: "
            + " / ".join(pr(r["perz"]) for r in (first, rows[1], rows[2], last)) + "; Wilson " + " / ".join(pr(r["wilson"]) for r in (first, rows[1], rows[2], last)) + "; Wald "
            + " / ".join(pr(r["wald"]) for r in (first, rows[1], rows[2], last)) + f". Bei n = {first['n']} hat die Stichprobe in {prozent(first['p0'], 0)} der Fälle kein Ereignis; dann ist das Bootstrap-Intervall ein Punkt (Breite 0).")
    return ("warning" if first["perz"] < 0.85 else "info"), text


def verdict_b(ex: dict) -> tuple[str, str]:
    rows = ex["rows"]
    lo, hi = rows[0], rows[-1]
    r99 = next((r for r in rows if r["B"] == 99), rows[2])
    text = (f"Abdeckung des Perzentil-Intervalls bei B = {lo['B']} / {r99['B']} / {hi['B']}: {pr(lo['perz'])} / {pr(r99['perz'])} / {pr(hi['perz'])} (BCa {pr(lo['bca'])} / {pr(r99['bca'])} / {pr(hi['bca'])}). "
            f"Die Endpunkte streuen von Lauf zu Lauf um {prozent(lo['sd_lo'], 1)} bis {prozent(lo['sd_hi'], 1)} der Breite bei B = {lo['B']}, um {prozent(r99['sd_lo'], 1)} bis {prozent(r99['sd_hi'], 1)} bei B = {r99['B']} und um {prozent(hi['sd_lo'], 1)} bis {prozent(hi['sd_hi'], 1)} bei B = {hi['B']}.")
    return "info", text


def verdict_dep(ex: dict) -> tuple[str, str]:
    r = ex["rows"]
    blocks = ", ".join(f"l = {l}: {pr(r[f'l{l}']['cov'])}" for l in C.DEP_BLOCKS)
    text = (f"AR(1) mit φ = {de(ex['phi'], 2)}, n = {ex['n']}: das t-Intervall deckt {pr(r['t']['cov'])}, der iid-Bootstrap {pr(r['iid']['cov'])}; Blockbootstrap {blocks}. "
            f"Die Varianz des Mittels ist {de(ex['var_factor'], 1)}-mal so groß wie bei unabhängigen Daten mit der Rauschvarianz der Reihe ({de((1 + ex['phi']) / (1 - ex['phi']), 1)}-mal so groß wie bei unabhängigen Daten mit derselben Streuung wie die Reihe selbst, nach der Theorie); der Block mit l = 25 ist {dn(r['l25']['width'] / r['iid']['width'], 1)}-mal so breit wie der iid-Bootstrap.")
    return ("warning" if r["iid"]["cov"] < 0.85 else "info"), text
