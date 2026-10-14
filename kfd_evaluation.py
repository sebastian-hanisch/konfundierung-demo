"""Auswertung: Kennzahlen der Stichprobe, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import kfd_constants as C
import kfd_model as model
from kausal_generator import SplitMix64, total_effect


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und ausdrücklichem Vorzeichen (+ oder −)."""
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    gamma: float = C.DEFAULT_GAMMA
    beta: float = C.DEFAULT_BETA
    tau: float = C.DEFAULT_TAU
    sigma: float = C.DEFAULT_SIGMA
    assign: str = "disponent"
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    z: np.ndarray
    x: np.ndarray
    y: np.ndarray
    share_treated: float
    mean_treated: float
    mean_untreated: float
    naive: float
    se: float
    lo: float
    hi: float
    truth: float
    limit: float          # Grenzwert des naiven Unterschieds für n gegen unendlich
    covers: bool
    wrong_sign: bool


def analyse(settings: Settings) -> Analysis:
    s = settings
    scm = model.model_scm(s.gamma, s.beta, s.tau, s.sigma, s.assign)
    z, x, y = (v[0] for v in model.observe(scm, s.n, SplitMix64(s.seed)))
    n1 = int(x.sum())
    n0 = s.n - n1
    d, se = model.welch_difference(y[None, :], x[None, :]) if min(n1, n0) >= 2 else (np.array([np.nan]), np.array([np.inf]))
    naive, se = float(d[0]), float(se[0])
    truth = total_effect(scm, "Faktor", "Servicegrad")
    limit = s.tau if s.assign == "muenzwurf" else model.naive_limit(s.gamma, s.beta, s.tau)
    return Analysis(
        settings=s, z=z, x=x, y=y, share_treated=n1 / s.n,
        mean_treated=float(y[x == 1].mean()) if n1 else float("nan"), mean_untreated=float(y[x == 0].mean()) if n0 else float("nan"),
        naive=naive, se=se, lo=naive - C.Z_CRIT * se, hi=naive + C.Z_CRIT * se,
        truth=truth, limit=limit, covers=bool(abs(naive - truth) <= C.Z_CRIT * se),
        wrong_sign=bool(truth != 0.0 and naive != 0.0 and math.copysign(1.0, naive) != math.copysign(1.0, truth)),
    )


# ----------------------------------------------------------------------------- Experiment 1: mehr Daten, gleicher Fehler

def coverage_experiment(ns=C.EXP_N_SWEEP, reps: int = C.EXP_REPS, s: Settings | None = None) -> list[dict]:
    """Deckung des 95-%-Intervalls für die wahre Wirkung: naiver Vergleich unter Auswahl, unter Münzwurf, und mit der Schwankung als Kontrolle."""
    s = s or Settings()
    obs = model.model_scm(s.gamma, s.beta, s.tau, s.sigma, "disponent")
    coin = model.model_scm(s.gamma, s.beta, s.tau, s.sigma, "muenzwurf")
    rows = []
    for n in ns:
        z, x, y = model.observe(obs, n, SplitMix64(C.SEED_BASE + n), reps)
        d, se = model.welch_difference(y, x)
        b, seb = model.adjusted_difference(y, x, z)
        z2, x2, y2 = model.observe(coin, n, SplitMix64(C.SEED_BASE + 10_000 + n), reps)
        d2, se2 = model.welch_difference(y2, x2)
        rows.append({
            "n": n, "naive_mean": float(d.mean()), "naive_cover": float(model.covers(d, se, s.tau).mean()), "naive_wrong": float((d < 0).mean()),
            "coin_mean": float(d2.mean()), "coin_cover": float(model.covers(d2, se2, s.tau).mean()),
            "adj_mean": float(b.mean()), "adj_cover": float(model.covers(b, seb, s.tau).mean()),
            "naive_se": float(se.mean()),
        })
    return rows


# ----------------------------------------------------------------------------- Experiment 2: wie stark muss die Auswahl sein?

def sign_map(gammas=C.EXP_MAP_GAMMAS, betas=C.EXP_MAP_BETAS, n: int = C.EXP_MAP_N, reps: int = C.EXP_REPS, tau: float = C.DEFAULT_TAU,
             sigma: float = C.DEFAULT_SIGMA) -> list[dict]:
    rows = []
    for i, beta in enumerate(betas):
        for j, gamma in enumerate(gammas):
            scm = model.model_scm(gamma, beta, tau, sigma, "disponent")
            _, x, y = model.observe(scm, n, SplitMix64(C.SEED_BASE + 20_000 + 100 * i + j), reps)
            d, _ = model.welch_difference(y, x)
            rows.append({"gamma": gamma, "beta": beta, "limit": model.naive_limit(gamma, beta, tau), "wrong": float((d < 0).mean()), "mean": float(d.mean())})
    return rows


# ----------------------------------------------------------------------------- Experiment 3: zwei Diagramme, dieselben Daten

def mechanism_experiment(runs: int = C.EXP_MECH_RUNS, n: int = C.EXP_MECH_N, s: Settings | None = None) -> dict:
    """Diagramm A (Schwankung -> Faktor) und Diagramm B (Faktor -> Schwankung) mit gleicher gemeinsamer Verteilung: KS-Tests je Lauf und Wahrheiten."""
    s = s or Settings()
    scm = model.model_scm(s.gamma, s.beta, s.tau, s.sigma, "disponent")
    pvals, first = [], None
    for r in range(runs):
        za, xa, ya = (v[0] for v in model.observe(scm, n, SplitMix64(C.SEED_BASE + 30_000 + r)))
        zb, xb, yb = model.mediator_sample(s.gamma, s.beta, s.tau, s.sigma, n, SplitMix64(C.SEED_BASE + 31_000 + r))
        for xv in (0.0, 1.0):
            pvals.append(model.ks_two_sample(za[xa == xv], zb[xb == xv])[1])
            pvals.append(model.ks_two_sample(ya[xa == xv], yb[xb == xv])[1])
        if first is None:
            first = {"A": (za, xa, ya), "B": (zb, xb, yb)}
    out = {"runs": runs, "n": n, "tests": len(pvals), "p_values": pvals, "below_05": float(np.mean(np.array(pvals) < 0.05)), "min_p": float(min(pvals))}
    for name, (z, x, y) in first.items():
        d, _ = model.welch_difference(y[None, :], x[None, :])
        b, _ = model.adjusted_difference(y[None, :], x[None, :], z[None, :])
        out[name] = {"mean_z0": float(z[x == 0].mean()), "mean_z1": float(z[x == 1].mean()), "naive": float(d[0]), "adjusted": float(b[0]),
                     "share_treated": float(x.mean())}
    out["truth_A"] = s.tau
    out["truth_B"] = model.naive_limit(s.gamma, s.beta, s.tau)
    return out


# ----------------------------------------------------------------------------- Experiment 4: wenn die Kontrolle wackelt

def proxy_experiment(reliabilities=C.EXP_PROXY_RELIABILITY, n: int = C.EXP_PROXY_N, s: Settings | None = None) -> dict:
    """Regression mit einer Kontrollvariable Z* = Z + Rauschen der Reliabilität r = Var(Z) / Var(Z*). Rückgabe: Zeilen, naiver Unterschied, Wahrheit."""
    s = s or Settings()
    scm = model.model_scm(s.gamma, s.beta, s.tau, s.sigma, "disponent")
    z, x, y = (v[0] for v in model.observe(scm, n, SplitMix64(C.SEED_BASE + 40_000)))
    naive = float(model.welch_difference(y[None, :], x[None, :])[0][0])
    rows = []
    for i, r in enumerate(reliabilities):
        noise = SplitMix64(C.SEED_BASE + 41_000 + i).normal(n) * math.sqrt(1.0 / r - 1.0)
        b, _ = model.adjusted_difference(y[None, :], x[None, :], (z + noise)[None, :])
        est = float(b[0])
        rows.append({"r": r, "estimate": est, "left": (s.tau - est) / (s.tau - naive) if s.tau != naive else 0.0})
    return {"rows": rows, "naive": naive, "truth": s.tau}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    s = a.settings
    cover = f"Das 95-%-Intervall ({de(a.lo, 1)} bis {de(a.hi, 1)}) deckt die wahre Wirkung {de(a.truth, 1)} Punkte"
    if s.assign == "muenzwurf":
        return (("success", f"{cover}. Bei Zufallszuteilung trifft der Vergleich die Wirkung im Mittel; einzelne Stichproben verfehlen sie in etwa jedem zwanzigsten Fall.")
                if a.covers else
                ("info", f"Das Intervall ({de(a.lo, 1)} bis {de(a.hi, 1)}) verfehlt die wahre Wirkung {de(a.truth, 1)}: bei Zufallszuteilung in etwa jedem zwanzigsten Fall zu erwarten."))
    bias = a.limit - a.truth
    if not a.covers:
        sign = " und zeigt in die falsche Richtung" if a.wrong_sign else ""
        return "warning", (f"Das Intervall ({de(a.lo, 1)} bis {de(a.hi, 1)}) verfehlt die wahre Wirkung {de(a.truth, 1)} Punkte{sign}. Mit mehr Standorten wird es "
                           f"enger um {signed(a.limit, 1)}, nicht richtiger.")
    if abs(bias) > 0.5:
        return "info", (f"{cover}, aber nur, weil das Rauschen bei {s.n} Standorten den Fehler verdeckt: Der naive Vergleich läuft gegen {signed(a.limit, 1)}, nicht "
                        f"gegen {de(a.truth, 1)}. Mit mehr Standorten kippt es.")
    return "success", f"{cover}. Der naive Vergleich ist hier (fast) unverzerrt: kaum Auswahl oder kein Einfluss der Schwankung."


def verdict_coverage(rows: list[dict]) -> tuple[str, str]:
    big = rows[-1]
    return "warning", (f"Bei {big['n']} Standorten deckt das Intervall des naiven Vergleichs die wahre Wirkung in {de(100 * big['naive_cover'], 0)} % der Wiederholungen, "
                       f"bei Münzwurf in {de(100 * big['coin_cover'], 0)} %: Das Intervall wird mit n enger, aber um die falsche Zahl.")


def verdict_map(rows: list[dict]) -> tuple[str, str]:
    mixed = [r for r in rows if r["gamma"] > 0 and r["beta"] > 0]
    flipped = [r for r in mixed if r["limit"] < 0]
    return "info", (f"In {len(flipped)} von {len(mixed)} Zellen mit Auswahl und Einfluss der Schwankung ist der naive Vergleich im Grenzfall negativ, obwohl die "
                    f"Wirkung positiv ist; ohne Auswahl oder ohne Einfluss der Schwankung ist er in keiner Zelle verzerrt.")


def verdict_mechanism(res: dict) -> tuple[str, str]:
    below = int(round(res["below_05"] * res["tests"]))
    return "info", (f"Von {res['tests']} Tests auf gleiche Verteilung liegen {below} unter 0,05 (kleinster p-Wert {de(res['min_p'], 3)}); bei gleicher Verteilung "
                    f"sind rund 5 % zu erwarten, die Tests finden keinen belastbaren Unterschied. Die wahre Wirkung ist {signed(res['truth_A'], 1)} in Diagramm A und "
                    f"{signed(res['truth_B'], 1)} in Diagramm B: Die Daten allein sagen nicht, welches gilt.")


def verdict_proxy(res: dict) -> tuple[str, str]:
    r9 = next(r for r in res["rows"] if abs(r["r"] - 0.9) < 1e-9)
    return "warning", (f"Schon bei einer Reliabilität von 0,9 (die Kontrolle ist zu 90 % Signal) bleiben {de(100 * r9['left'], 0)} % des Fehlers übrig; "
                       "mit der Kontrolle gerechnet ist nicht dasselbe wie richtig gerechnet.")
