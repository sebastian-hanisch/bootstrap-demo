"""Netze (seltene Ereignisse) und Eingabeprüfung der Spielwiese (eigene kleine Stichprobe)."""

from __future__ import annotations

import re

import numpy as np

import bst_constants as C


def _grid_edges(rows: int, cols: int):
    idx = lambda r, c: r * cols + c
    e = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                e.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                e.append((idx(r, c), idx(r + 1, c)))
    return rows * cols, tuple(e)


def named_netz(key: str):
    """(Name, Knotenzahl, Kanten); Hantel: zwei Dreiecke, verbunden durch die Brücke 2–3."""
    kind, r, c = C.NETZE[key]
    if kind == "hantel":
        return "Hantel", 6, ((0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3))
    n, edges = _grid_edges(r, c)
    return f"Gitter {r}×{c}", n, edges


def _connected_table(n: int, edges, chunk: int = 1 << 15) -> np.ndarray:
    """Zusammenhang für alle 2^m Kantenmengen (Kante e = Bit e), Kleinste-Marke-Ausbreitung."""
    m = len(edges)
    shifts = np.arange(m, dtype=np.uint32)
    out = np.empty(1 << m, dtype=bool)
    for start in range(0, 1 << m, chunk):
        masks = np.arange(start, min(start + chunk, 1 << m), dtype=np.uint32)
        present = ((masks[:, None] >> shifts) & 1).astype(bool)
        lab = np.tile(np.arange(n, dtype=np.int16), (len(masks), 1))
        while True:
            before = int(lab.sum())
            for e, (u, v) in enumerate(edges):
                low = np.minimum(lab[:, u], lab[:, v])
                pe = present[:, e]
                lab[:, u] = np.where(pe, low, lab[:, u])
                lab[:, v] = np.where(pe, low, lab[:, v])
            if int(lab.sum()) == before:
                break
        out[start:start + len(masks)] = lab.max(axis=1) == 0
    return out


_CACHE: dict = {}


def unreliability(key: str, p: float) -> float:
    """Ausfallwahrscheinlichkeit U = 1 − R(p) des Netzes (jede Kante hält unabhängig mit Wahrscheinlichkeit p), exakt aus der Aufzählung aller Kantenmengen."""
    name, n, edges = named_netz(key)
    if key not in _CACHE:
        tab = _connected_table(n, edges)
        k = np.zeros(1 << len(edges), dtype=np.int64)
        masks = np.arange(1 << len(edges), dtype=np.int64)
        for e in range(len(edges)):
            k += (masks >> e) & 1
        _CACHE[key] = (tab, k)
    tab, k = _CACHE[key]
    m = len(edges)
    w = p**k * (1 - p) ** (m - k)
    return float(1.0 - w[tab].sum())


def parse_numbers(text: str) -> np.ndarray:
    """Zahlen aus einem Text (getrennt durch Leerzeichen, Semikolon oder Zeilenumbruch; Dezimalkomma oder -punkt). ValueError mit lesbarer Meldung bei Fehlern."""
    tokens = [t for t in re.split(r"[\s;]+", text.strip()) if t]
    if not tokens:
        raise ValueError("Keine Werte eingegeben.")
    vals = []
    for t in tokens:
        try:
            v = float(t.replace(",", "."))
        except ValueError:
            raise ValueError(f"»{t}« ist keine Zahl.") from None
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError(f"»{t}« ist keine endliche Zahl.")
        vals.append(v)
    if len(vals) < C.CUSTOM_MIN_N:
        raise ValueError(f"Mindestens {C.CUSTOM_MIN_N} Werte nötig.")
    if len(vals) > C.CUSTOM_MAX_N:
        raise ValueError(f"Höchstens {C.CUSTOM_MAX_N} Werte erlaubt.")
    return np.array(vals, dtype=float)
