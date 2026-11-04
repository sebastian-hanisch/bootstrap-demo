"""Kern der Demo (nur numpy und math).

Daten x_1..x_n (iid, bei AR(1) abhängig) aus einer bekannten Verteilung; die Wahrheit der Statistik ist exakt. Intervalle zum Niveau 1 − α:
  t      klassisches t-Intervall für den Mittelwert (Quantil der t-Verteilung aus der endlichen Reihe für ganzzahlige Freiheitsgrade, ohne scipy)
  perz   Perzentil-Bootstrap [q_{α/2}, q_{1−α/2}] der Bootstrap-Verteilung θ*
  basic  umgekehrtes Perzentil-Intervall [2θ̂ − q_{1−α/2}, 2θ̂ − q_{α/2}]
  boott  studentisiert (Bootstrap-t, nur Mittelwert): t* = (θ* − θ̂)/SE*, [θ̂ − t*_{1−α/2} SE, θ̂ − t*_{α/2} SE]
  bca    BCa: z₀ aus dem Anteil θ* < θ̂, Beschleunigung a aus dem Jackknife
  block  gleitender Blockbootstrap (abhängige Daten): Blöcke der Länge l zufällig mit Zurücklegen aneinandergesetzt
Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b).
"""

from __future__ import annotations

import math
from itertools import combinations_with_replacement

import numpy as np

GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1
Z95 = 1.959963984540054
Z90 = 1.2815515655446004


# --- Zufallsstrom ---------------------------------------------------------------------------------------------------------------------------

def mix_int(z):
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def _mix_arr(z):
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stream_seed(*keys):
    s = 0x1234_5678_9ABC_DEF0
    for k in keys:
        s = mix_int(s ^ mix_int(int(k) + GAMMA))
    return s


def uniforms(seed, count):
    """count gleichverteilte Zahlen in (0, 1), nie 0 oder 1."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = _mix_arr(np.uint64(seed) + idx * np.uint64(GAMMA))
    return ((z >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))


def normals(seed, count):
    """count Standardnormalzahlen (Box–Muller)."""
    n2 = (count + 1) // 2
    u = uniforms(seed, 2 * n2)
    r = np.sqrt(-2.0 * np.log(u[:n2]))
    z = np.concatenate([r * np.cos(2 * np.pi * u[n2:]), r * np.sin(2 * np.pi * u[n2:])])
    return z[:count]


# --- Normal- und t-Verteilung, Gammafunktion -----------------------------------------------------------------------------------------------

def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


_ACK_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_ACK_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
_ACK_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_ACK_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)


def ndtri(p: float) -> float:
    """Umkehrfunktion der Standardnormalverteilung (Acklam, mit einem Halley-Schritt über erf): relative Genauigkeit etwa 1e-15."""
    if not 0.0 < p < 1.0:
        return -math.inf if p <= 0 else math.inf
    a, b, c, d = _ACK_A, _ACK_B, _ACK_C, _ACK_D
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = Phi(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def t_cdf(t: float, df: int) -> float:
    """Verteilungsfunktion der t-Verteilung für ganzzahlige Freiheitsgrade (endliche Reihe, Abramowitz–Stegun 26.7.3 und 26.7.4)."""
    if t == 0:
        return 0.5
    theta = math.atan(abs(t) / math.sqrt(df))
    c, s = math.cos(theta), math.sin(theta)
    if df % 2 == 1:
        if df == 1:
            A = 2 * theta / math.pi
        else:
            term, tot = c, c
            for k in range((df - 3) // 2):
                term *= (2 * (k + 1)) / (2 * (k + 1) + 1) * c * c
                tot += term
            A = (2 / math.pi) * (theta + s * tot)
    else:
        term, tot = 1.0, 1.0
        for k in range((df - 2) // 2):
            term *= (2 * k + 1) / (2 * k + 2) * c * c
            tot += term
        A = s * tot
    return 0.5 + A / 2 if t > 0 else 0.5 - A / 2


def t_quantile(p: float, df: int) -> float:
    """Quantil der t-Verteilung (ganzzahlige Freiheitsgrade) durch Bisektion der Verteilungsfunktion."""
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_quantile(1 - p, df)
    hi = 1.0
    while t_cdf(hi, df) < p and hi < 1e12:
        hi *= 2
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


_LANCZOS = (0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)


def lgam(x):
    """ln Γ(x) für x ≥ ½ (Lanczos); Felder erlaubt."""
    x = np.asarray(x, dtype=float) - 1.0
    a = np.full(x.shape, _LANCZOS[0])
    for i in range(1, 9):
        a = a + _LANCZOS[i] / (x + i)
    t = x + 7.5
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * np.log(t) - t + np.log(a)


# --- Daten und Wahrheit ---------------------------------------------------------------------------------------------------------------------

def sample_lognormal(sigma: float, n: int, seed: int) -> np.ndarray:
    return np.exp(sigma * normals(seed, n))


def sample_bernoulli(pi: float, n: int, seed: int) -> np.ndarray:
    return (uniforms(seed, n) < pi).astype(float)


def sample_uniform(n: int, seed: int) -> np.ndarray:
    return uniforms(seed, n)


def sample_ar1(phi: float, n: int, seed: int, burn: int = 200) -> np.ndarray:
    """AR(1) x_t = φ x_(t−1) + ε_t mit Standardnormalrauschen und Einschwingzeit."""
    e = normals(seed, n + burn)
    x = np.empty(n + burn)
    x[0] = e[0]
    for i in range(1, n + burn):
        x[i] = phi * x[i - 1] + e[i]
    return x[burn:]


def lognormal_truth(stat: str, sigma: float) -> float:
    """Wahrheit der Statistik für log-normale Daten exp(σ Z)."""
    if stat == "mean":
        return math.exp(sigma**2 / 2)
    if stat == "median":
        return 1.0
    if stat == "q90":
        return math.exp(Z90 * sigma)
    if stat == "sd":
        return math.sqrt(math.exp(sigma**2) * (math.exp(sigma**2) - 1))
    raise ValueError(stat)


STATS = {
    "mean": lambda x, axis=-1: x.mean(axis=axis),
    "median": lambda x, axis=-1: np.median(x, axis=axis),
    "q90": lambda x, axis=-1: np.quantile(x, 0.9, axis=axis),
    "sd": lambda x, axis=-1: x.std(axis=axis, ddof=1),
    "max": lambda x, axis=-1: x.max(axis=axis),
}


# --- Bootstrap ------------------------------------------------------------------------------------------------------------------------------

def boot_indices(seed: int, B: int, n: int) -> np.ndarray:
    """B Ziehungen mit Zurücklegen: (B, n) Indizes aus festen Strömen."""
    return np.minimum((uniforms(seed, B * n) * n).astype(np.int64), n - 1).reshape(B, n)


def block_indices(seed: int, B: int, n: int, l: int) -> np.ndarray:
    """Gleitender Blockbootstrap: ⌈n/l⌉ Blöcke der Länge l mit zufälligem Start in 0 … n − l, aneinandergesetzt und auf n gekürzt: (B, n)."""
    l = max(1, min(l, n))
    nb = math.ceil(n / l)
    starts = np.minimum((uniforms(seed, B * nb) * (n - l + 1)).astype(np.int64), n - l).reshape(B, nb)
    return (starts[:, :, None] + np.arange(l)[None, None, :]).reshape(B, -1)[:, :n]


def jackknife(x: np.ndarray, stat: str) -> np.ndarray:
    """Statistik der n Stichproben ohne je eine Beobachtung (für den Mittelwert in geschlossener Form)."""
    n = len(x)
    if stat == "mean":
        return (x.sum() - x) / (n - 1)
    f = STATS[stat]
    return np.array([float(f(np.delete(x, i))) for i in range(n)])


def bca_interval(theta_star: np.ndarray, theta_hat: float, loo: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """BCa-Intervall: Verzerrungskorrektur z₀ aus dem Anteil θ* < θ̂, Beschleunigung a aus dem Jackknife; die Anteile werden auf [1/(B+1), 1 − 1/(B+1)] begrenzt."""
    B = len(theta_star)
    frac = float((theta_star < theta_hat).mean() + 0.5 * (theta_star == theta_hat).mean())
    frac = min(max(frac, 1.0 / (B + 1)), 1 - 1.0 / (B + 1))
    z0 = ndtri(frac)
    d = loo.mean() - loo
    denom = 6 * float((d**2).sum()) ** 1.5
    a = float((d**3).sum()) / denom if denom > 0 else 0.0
    out = []
    for za in (ndtri(alpha / 2), ndtri(1 - alpha / 2)):
        den = 1 - a * (z0 + za)
        adj = Phi(z0 + (z0 + za) / den) if den > 0 else (1.0 if z0 + za > 0 else 0.0)
        out.append(float(np.quantile(theta_star, min(max(adj, 0.0), 1.0))))
    return out[0], out[1]


def boot_intervals(x: np.ndarray, stat: str, B: int, seed: int, alpha: float = 0.05, methods=("perz", "basic", "bca", "boott", "t")) -> tuple[float, dict]:
    """Intervalle für `stat` aus der Stichprobe x mit B Bootstrap-Ziehungen: (θ̂, {Methode: (unten, oben)}). boott und t nur für den Mittelwert."""
    n = len(x)
    f = STATS[stat]
    th = float(f(x))
    xs = x[boot_indices(seed, B, n)]
    ts = f(xs, axis=1)
    out = {}
    if "perz" in methods or "basic" in methods:
        q = np.quantile(ts, [alpha / 2, 1 - alpha / 2])
        if "perz" in methods:
            out["perz"] = (float(q[0]), float(q[1]))
        if "basic" in methods:
            out["basic"] = (float(2 * th - q[1]), float(2 * th - q[0]))
    if "bca" in methods:
        out["bca"] = bca_interval(ts, th, jackknife(x, stat), alpha)
    if "boott" in methods and stat == "mean":
        se = float(x.std(ddof=1)) / math.sqrt(n)
        se_star = xs.std(axis=1, ddof=1) / math.sqrt(n)
        tt = (ts - th) / np.where(se_star > 0, se_star, np.nan)
        if np.isfinite(tt).any():
            q = np.nanquantile(tt, [alpha / 2, 1 - alpha / 2])
            out["boott"] = (float(th - q[1] * se), float(th - q[0] * se))
    if "t" in methods and stat == "mean" and n >= 2:
        h = t_quantile(1 - alpha / 2, n - 1) * float(x.std(ddof=1)) / math.sqrt(n)
        out["t"] = (th - h, th + h)
    return th, out


def block_interval(x: np.ndarray, l: int, B: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    """Perzentil-Intervall des Mittelwerts mit gleitendem Blockbootstrap der Blocklänge l."""
    ts = x[block_indices(seed, B, len(x), l)].mean(axis=1)
    q = np.quantile(ts, [alpha / 2, 1 - alpha / 2])
    return float(q[0]), float(q[1])


def exact_boot_mean(x) -> tuple[np.ndarray, np.ndarray]:
    """Exakte Bootstrap-Verteilung des Mittels für kleine n (n ≤ 8): alle Multimengen vom Umfang n mit Wahrscheinlichkeit n!/(Πk_i!)/n^n; Rückgabe (Werte aufsteigend, Wahrscheinlichkeiten)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n > 8:
        raise ValueError("exakte Bootstrap-Verteilung nur bis n = 8")
    vals, probs = {}, {}
    for combo in combinations_with_replacement(range(n), n):
        counts = np.bincount(combo, minlength=n)
        p = math.factorial(n) / math.prod(math.factorial(int(k)) for k in counts) / n**n
        v = round(float((x * counts).sum() / n), 12)
        probs[v] = probs.get(v, 0.0) + p
    keys = sorted(probs)
    return np.array(keys), np.array([probs[k] for k in keys])


def discrete_quantile(values: np.ndarray, probs: np.ndarray, q: float) -> float:
    """Kleinster Wert mit Verteilungsfunktion ≥ q (wie die Quantilfunktion einer diskreten Verteilung)."""
    cdf = np.cumsum(probs)
    return float(values[int(np.searchsorted(cdf, q - 1e-12, side="left"))])


# --- Bernoulli: exakte Abdeckung ------------------------------------------------------------------------------------------------------------

def wilson(phat, n, z=Z95):
    d = 1 + z * z / n
    c = (phat + z * z / (2 * n)) / d
    h = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def wald_half(phat, n, z=Z95):
    return z * np.sqrt(phat * (1 - phat) / n)


def binom_pmf(n: int, p: float) -> np.ndarray:
    """Zähldichte der Binomialverteilung (n, p) für k = 0 … n, über die Logarithmen mit der Lanczos-Gammafunktion."""
    k = np.arange(n + 1, dtype=float)
    if p <= 0:
        out = np.zeros(n + 1)
        out[0] = 1.0
        return out
    if p >= 1:
        out = np.zeros(n + 1)
        out[-1] = 1.0
        return out
    lg = lambda a: lgam(a + 1.0)          # ln a!
    lp = lg(np.full(n + 1, float(n))) - lg(k) - lg(n - k) + k * math.log(p) + (n - k) * math.log1p(-p)
    return np.exp(lp)


def bernoulli_exact(pi: float, n: int, alpha: float = 0.05) -> dict:
    """Exakte Abdeckung von Perzentil-Bootstrap (B → ∞: S* ~ Binomial(n, S/n)), Wilson und Wald für die Erfolgswahrscheinlichkeit π bei Stichprobenumfang n:
    Summe der Binomialwahrscheinlichkeiten P(S = s) über alle s, deren Intervall π enthält. Dazu P(S = 0) und der Anteil mit Breite 0 sowie die Fehltreffer des Perzentil-Intervalls."""
    S = np.arange(n + 1)
    pm = binom_pmf(n, pi)
    phat = S / n
    plo, phi_ = np.zeros(n + 1), np.zeros(n + 1)
    for s in range(1, n):                                           # s = 0 und s = n: degeneriert (Breite 0)
        cdf = np.cumsum(binom_pmf(n, s / n))
        plo[s] = np.searchsorted(cdf, alpha / 2 - 1e-12, side="left")
        phi_[s] = np.searchsorted(cdf, 1 - alpha / 2 - 1e-12, side="left")
    plo[0], phi_[0], plo[n], phi_[n] = 0, 0, n, n
    plo, phi_ = plo / n, phi_ / n
    wl, wu = wilson(phat, n)
    hw = wald_half(phat, n)
    return {"perz": float(pm[(plo <= pi) & (pi <= phi_)].sum()), "wilson": float(pm[(wl <= pi) & (pi <= wu)].sum()), "wald": float(pm[(phat - hw <= pi) & (pi <= phat + hw)].sum()),
            "p0": float(pm[0]), "degenerate": float(pm[phi_ - plo <= 1e-12].sum()), "left": float(pm[pi < plo].sum()), "right": float(pm[pi > phi_].sum())}
