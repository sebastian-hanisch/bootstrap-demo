"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

STATS = {"mean": "Mittelwert", "median": "Median", "q90": "90-%-Quantil", "sd": "Standardabweichung"}
DEFAULT_STAT = "mean"
SIGMA_CHOICES = (0.5, 1.0, 1.5)                              # Schiefe der log-normalen Daten exp(σ Z)
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = SIGMA_CHOICES[0], SIGMA_CHOICES[-1], 1.0
N_CHOICES = (10, 20, 50, 100, 200)                          # Stichprobengröße der Experimente 2 und 4
N_MIN, N_MAX, DEFAULT_N = N_CHOICES[0], N_CHOICES[-1], 50
B_CHOICES = (99, 499, 999)                                  # Zahl der Bootstrap-Ziehungen in den Experimenten 1 und 2
B_MIN, B_MAX, DEFAULT_B = B_CHOICES[0], B_CHOICES[-1], 499
NETZE = {"hantel": ("hantel", 0, 0), "3x3": ("gitter", 3, 3), "3x4": ("gitter", 3, 4)}
DEFAULT_NETZ = "3x3"
P_CHOICES = (0.5, 0.7, 0.8, 0.9, 0.95)                      # Haltewahrscheinlichkeit je Kante (Ausfallwahrscheinlichkeit = seltenes Ereignis)
P_MIN, P_MAX, DEFAULT_P = P_CHOICES[0], P_CHOICES[-1], 0.9
PHI_CHOICES = (0.0, 0.5, 0.8, 0.95)                         # Abhängigkeit der AR(1)-Daten
PHI_MIN, PHI_MAX, DEFAULT_PHI = PHI_CHOICES[0], PHI_CHOICES[-1], 0.8
SEED_MAX, DEFAULT_SEED = 999999, 39
ALPHA = 0.05                                                # Niveau 95 %

COV_NS = (10, 20, 50, 100, 200)                             # Experiment 1: Stichprobengrößen
COV_R = 500                                                 # Wiederholungen je Stichprobengröße
STAT_R = 500
B_LIST = (19, 49, 99, 199, 499, 999)                        # Experiment 4
B_R = 500
B_ENDPOINT_DATASETS, B_ENDPOINT_RUNS = 40, 10               # Streuung der Endpunkte: Datensätze, Wiederholungen des Bootstraps je Datensatz
RARE_NS = (30, 100, 300, 1000, 3000)                        # Experiment 3 (exakt)
DEP_N, DEP_R, DEP_B = 500, 200, 299                         # Experiment 5
DEP_BLOCKS = (5, 10, 25, 50)
CUSTOM_MAX_N, CUSTOM_MIN_N, CUSTOM_B, CUSTOM_EXACT_N = 30, 3, 999, 8   # Spielwiese: Zahl der Werte, Ziehungen, exakte Verteilung bis n = 8

METHODS = ("t", "perz", "basic", "boott", "bca")
METHOD_LABELS = {"t": "t-Intervall", "perz": "Perzentil", "basic": "umgekehrtes Perzentil (basic)", "boott": "Bootstrap-t", "bca": "BCa", "iid": "iid-Bootstrap"}
METHOD_COLORS = {"t": "#222222", "perz": "#C0392B", "basic": "#E08E0B", "boott": "#2F6B65", "bca": "#4c78a8", "iid": "#C0392B"}
COLORS = {"perz": "#C0392B", "wilson": "#2F6B65", "wald": "#E08E0B", "ref": "#222222", "block": "#2F6B65"}
COLOR_BAD = "#C0392B"
COLOR_OK = "#2F6B65"

PRESETS = {
    "Standardfall (Voreinstellung)": {"stat": DEFAULT_STAT, "sigma": DEFAULT_SIGMA, "n": DEFAULT_N, "B": DEFAULT_B, "netz": DEFAULT_NETZ, "p": DEFAULT_P, "phi": DEFAULT_PHI, "seed": DEFAULT_SEED},
    "Schiefe Daten, kleines n": {"stat": "mean", "sigma": 1.5, "n": 10, "B": 999, "netz": DEFAULT_NETZ, "p": DEFAULT_P, "phi": DEFAULT_PHI, "seed": DEFAULT_SEED},
    "Standardabweichung": {"stat": "sd", "sigma": DEFAULT_SIGMA, "n": DEFAULT_N, "B": 999, "netz": DEFAULT_NETZ, "p": DEFAULT_P, "phi": DEFAULT_PHI, "seed": DEFAULT_SEED},
    "Seltene Ausfälle": {"stat": DEFAULT_STAT, "sigma": DEFAULT_SIGMA, "n": DEFAULT_N, "B": DEFAULT_B, "netz": DEFAULT_NETZ, "p": 0.95, "phi": DEFAULT_PHI, "seed": DEFAULT_SEED},
    "Daten mit Gedächtnis": {"stat": DEFAULT_STAT, "sigma": DEFAULT_SIGMA, "n": DEFAULT_N, "B": DEFAULT_B, "netz": DEFAULT_NETZ, "p": DEFAULT_P, "phi": 0.95, "seed": DEFAULT_SEED},
    "Zu wenige Ziehungen": {"stat": DEFAULT_STAT, "sigma": DEFAULT_SIGMA, "n": DEFAULT_N, "B": 99, "netz": DEFAULT_NETZ, "p": DEFAULT_P, "phi": DEFAULT_PHI, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Mittelwert log-normaler Daten (σ = 1), 499 Ziehungen: bei kleinem n deckt das Perzentil-Intervall nicht besser als das t-Intervall, und die Fehltreffer liegen fast alle auf einer Seite.",
    "Schiefe Daten, kleines n": "σ = 1,5 und n = 10: das Perzentil-Intervall deckt nur etwa 71 %; Bootstrap-t holt die Abdeckung zurück, wird aber viele Male so breit.",
    "Standardabweichung": "Die Standardabweichung (n = 50): der Bootstrap unterschätzt den Schwanz, das Perzentil-Intervall deckt nur rund 60 %.",
    "Seltene Ausfälle": "Gitter 3×3, p = 0,95: die Ausfallwahrscheinlichkeit liegt bei 0,012; bei n = 30 hat der Bootstrap in 70 % der Fälle null Ereignisse und die Breite null.",
    "Daten mit Gedächtnis": "AR(1) mit φ = 0,95: der iid-Bootstrap deckt wie das t-Intervall nur etwa ein Viertel; Blöcke helfen, brauchen aber lange Reihen.",
    "Zu wenige Ziehungen": "Mit nur 99 Ziehungen verliert das 95-%-Intervall rund zwei Abdeckungspunkte, und die Endpunkte sind um fünf bis acht Prozent der Breite unscharf.",
}
