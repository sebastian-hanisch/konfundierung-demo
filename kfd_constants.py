"""Konstanten: Regler-Grenzen, feste Annahmen, Experimente, Farben und Presets."""

# Regler. Rechenaufwand am Maximum: eine Stichprobe mit n = 5 000 Standorten, O(n), unter 10 ms.
N_MIN, N_MAX, N_STEP, DEFAULT_N = 20, 5000, 20, 400
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 0.0, 3.0, 0.25, 1.0     # Stärke der Auswahl nach der Schwankung
BETA_MIN, BETA_MAX, BETA_STEP, DEFAULT_BETA = 0.0, 8.0, 0.5, 4.0          # Servicegrad-Verlust je Einheit Schwankung (Punkte)
TAU_MIN, TAU_MAX, TAU_STEP, DEFAULT_TAU = 0.0, 8.0, 0.5, 3.5              # wahre Wirkung des hohen Faktors (Punkte)
SIGMA_MIN, SIGMA_MAX, SIGMA_STEP, DEFAULT_SIGMA = 1.0, 10.0, 0.5, 5.0     # Reststreuung des Servicegrads (Punkte)
SEED_MAX, DEFAULT_SEED = 999999, 35

ASSIGN_CHOICES = ("disponent", "muenzwurf")
ASSIGN_LABELS = {"disponent": "Der Disponent wählt (nach der Schwankung)", "muenzwurf": "Eine Münze entscheidet (Zufallszuteilung)"}

# Feste Annahmen der Welt (keine Regler): Anteil der Behandelten 50 % ohne Auswahl, Basiswert des Servicegrads unterhalb der Deckelung bei 100.
THRESHOLD_A = 0.0
BASE = 80.0
Z_CRIT = 1.959964                                  # 95-%-Intervall, Normalnäherung (n >= 20)

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 700000
EXP_REPS = 400                                     # Wiederholungen je Zelle; Binomialfehler der Deckung rund 1 Prozentpunkt
EXP_N_SWEEP = (50, 200, 1000, 5000)                # Experiment 1
EXP_MAP_N = 200                                    # Experiment 2: Stichprobe je Zelle
EXP_MAP_GAMMAS = (0.0, 0.25, 0.5, 1.0, 2.0)
EXP_MAP_BETAS = (0.0, 1.0, 2.0, 4.0, 6.0)
EXP_MECH_N, EXP_MECH_RUNS = 20000, 12              # Experiment 3: Stichproben je Diagramm und Anzahl Wiederholungen
EXP_PROXY_N = 100000                               # Experiment 4
EXP_PROXY_RELIABILITY = (1.0, 0.9, 0.7, 0.5, 0.2)

COLOR_TREATED = "#2F6B65"
COLOR_UNTREATED = "#8a8a8a"
COLOR_TRUTH = "#D68A2E"
COLOR_WARN = "#c0392b"
COLOR_LINE = "#4c78a8"

PRESETS = {
    "Standard (Disponent wählt nach Schwankung)": {"n": DEFAULT_N, "gamma": DEFAULT_GAMMA, "beta": DEFAULT_BETA, "tau": DEFAULT_TAU, "sigma": DEFAULT_SIGMA,
                                                  "assign": "disponent", "seed": DEFAULT_SEED},
    "Effekt fast unsichtbar": {"n": DEFAULT_N, "gamma": 0.5, "beta": DEFAULT_BETA, "tau": DEFAULT_TAU, "sigma": DEFAULT_SIGMA,
                               "assign": "disponent", "seed": DEFAULT_SEED},
    "Fünftausend Standorte": {"n": 5000, "gamma": DEFAULT_GAMMA, "beta": DEFAULT_BETA, "tau": DEFAULT_TAU, "sigma": DEFAULT_SIGMA,
                              "assign": "disponent", "seed": DEFAULT_SEED},
    "Münzwurf statt Auswahl": {"n": DEFAULT_N, "gamma": DEFAULT_GAMMA, "beta": DEFAULT_BETA, "tau": DEFAULT_TAU, "sigma": DEFAULT_SIGMA,
                               "assign": "muenzwurf", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard (Disponent wählt nach Schwankung)": "400 Standorte: wer stark schwankt, bekommt eher den hohen Faktor und ist ohnehin schlechter bedient. "
                                                  "Der Vergleich zeigt einen Schaden, wo die Wirkung positiv ist.",
    "Effekt fast unsichtbar": "Schwächere Auswahl (γ = 0,5): der naive Unterschied läuft gegen +0,65, rund ein Fünftel der wahren Wirkung von +3,5.",
    "Fünftausend Standorte": "Standardwelt mit 5 000 statt 400 Standorten: das Intervall wird eng, aber um −1,0 statt um die wahre Wirkung.",
    "Münzwurf statt Auswahl": "Dieselben Standorte, aber eine Münze verteilt den Faktor: der Vergleich trifft die Wirkung im Mittel. Dasselbe wie Auswahlstärke γ = 0.",
}
