"""Kern: geschlossene Form des naiven Unterschieds, Schätzer (Welch, Regression mit Kontrolle), zweites Diagramm mit derselben Verteilung,
Kolmogorov-Smirnov-Test. Nur numpy und math, kein Streamlit, kein scipy. Jede Einheit eine reine Funktion; Zufall nur über übergebene Ströme."""

from __future__ import annotations

import math

import numpy as np

import kfd_constants as C
from kausal_generator import SplitMix64, normal_cdf, normal_pdf, normal_ppf, sample, standorte_confounded


# ----------------------------------------------------------------------------- Geschlossene Form (Diagramm A)

def p_treated(gamma: float, a: float = C.THRESHOLD_A) -> float:
    """Anteil der Standorte mit hohem Faktor: X = 1 genau dann, wenn a + gamma*Z + eta > 0 mit Z, eta standardnormal."""
    return float(normal_cdf(a / math.sqrt(1.0 + gamma ** 2)))


def mean_z_given_x(gamma: float, a: float = C.THRESHOLD_A) -> tuple[float, float]:
    """(E[Z | X=0], E[Z | X=1]) geschlossen. S = a + gamma*Z + eta ist N(a, 1+gamma^2), X = [S > 0], E[Z | S] = gamma*(S-a)/(1+gamma^2);
    der Mittelwert von S über der Schwelle ist der abgeschnittene Normalmittelwert (inverse Mills-Quote)."""
    v = 1.0 + gamma ** 2
    c = -a / math.sqrt(v)
    upper = float(normal_pdf(c) / (1.0 - normal_cdf(c)))
    lower = float(normal_pdf(c) / normal_cdf(c))
    k = gamma / math.sqrt(v)
    return -k * lower, k * upper


def naive_limit(gamma: float, beta: float, tau: float, a: float = C.THRESHOLD_A) -> float:
    """Wert, gegen den der naive Unterschied (Mittel mit minus Mittel ohne hohen Faktor) mit wachsendem n konvergiert:
    tau - beta * (E[Z|X=1] - E[Z|X=0]). Gleich tau nur ohne Auswahl (gamma = 0) oder ohne Wirkung der Schwankung (beta = 0)."""
    e0, e1 = mean_z_given_x(gamma, a)
    return tau - beta * (e1 - e0)


def model_scm(gamma: float, beta: float, tau: float, sigma: float, assign: str = "disponent"):
    """Strukturmodell der Standorte-Welt; bei Münzwurf mit demselben Anteil Behandelter wie unter der Auswahl."""
    coin = p_treated(gamma) if assign == "muenzwurf" else None
    return standorte_confounded(gamma, C.THRESHOLD_A, beta, tau, sigma, C.BASE, coin_p=coin)


def observe(scm, n: int, rng: SplitMix64, reps: int = 1):
    """(Schwankung, Faktor, Servicegrad), je Array (reps, n)."""
    d = sample(scm, n, rng, reps)
    return d["Schwankung"], d["Faktor"], d["Servicegrad"]


# ----------------------------------------------------------------------------- Schätzer, vektorisiert über Wiederholungen (Zeilen)

def welch_difference(y: np.ndarray, x: np.ndarray):
    """(Unterschied der Mittelwerte mit minus ohne x = 1, Standardfehler nach Welch) je Zeile."""
    n1 = x.sum(axis=1)
    n0 = x.shape[1] - n1
    m1 = (y * x).sum(axis=1) / n1
    m0 = (y * (1.0 - x)).sum(axis=1) / n0
    v1 = (((y - m1[:, None]) ** 2) * x).sum(axis=1) / (n1 - 1.0)
    v0 = (((y - m0[:, None]) ** 2) * (1.0 - x)).sum(axis=1) / (n0 - 1.0)
    return m1 - m0, np.sqrt(v1 / n1 + v0 / n0)


def adjusted_difference(y: np.ndarray, x: np.ndarray, z: np.ndarray):
    """Koeffizient von x in der Regression y ~ 1 + x + z (kleinste Quadrate) und sein Standardfehler, je Zeile. Normalgleichungen, ohne (r, n, 3)-Matrix."""
    r, n = y.shape
    one = np.ones(r)
    sx, sz = x.sum(1), z.sum(1)
    xtx = np.empty((r, 3, 3))
    xtx[:, 0, 0] = n
    xtx[:, 0, 1] = xtx[:, 1, 0] = sx
    xtx[:, 0, 2] = xtx[:, 2, 0] = sz
    xtx[:, 1, 1] = (x * x).sum(1)
    xtx[:, 1, 2] = xtx[:, 2, 1] = (x * z).sum(1)
    xtx[:, 2, 2] = (z * z).sum(1)
    xty = np.stack([y.sum(1) * one, (x * y).sum(1), (z * y).sum(1)], axis=1)
    beta = np.linalg.solve(xtx, xty[..., None])[..., 0]
    rss = (y * y).sum(1) - (beta * xty).sum(1)
    inv11 = np.linalg.inv(xtx)[:, 1, 1]
    return beta[:, 1], np.sqrt(np.maximum(rss, 0.0) / (n - 3) * inv11)


def covers(estimate: np.ndarray, se: np.ndarray, truth: float) -> np.ndarray:
    """Liegt die Wahrheit im 95-%-Intervall (Normalnäherung)? Je Zeile."""
    return np.abs(estimate - truth) <= C.Z_CRIT * se


# ----------------------------------------------------------------------------- Zweites Diagramm, dieselbe Verteilung

def mediator_sample(gamma: float, beta: float, tau: float, sigma: float, n: int, rng: SplitMix64, a: float = C.THRESHOLD_A):
    """Dieselbe gemeinsame Verteilung von (Schwankung, Faktor, Servicegrad) auf anderem Weg (Diagramm B: Faktor -> Schwankung -> Servicegrad):
    erst der Faktor, dann die Schwankung aus ihrem Gesetz gegeben den Faktor (abgeschnittene Normalverteilung über die Quantilfunktion),
    dann der Servicegrad aus derselben Gleichung. Rückgabe: (z, x, y), je Array (n,)."""
    v = 1.0 + gamma ** 2
    sv = math.sqrt(v)
    c = -a / sv
    p = p_treated(gamma, a)
    u_x = rng.uniform(n)
    u_s = rng.uniform(n)
    zn = rng.normal(n)
    eps = rng.normal(n)
    x = (u_x < p).astype(float)
    cdf_c = float(normal_cdf(c))
    q = np.where(x == 1.0, cdf_c + u_s * (1.0 - cdf_c), u_s * cdf_c)
    s = a + sv * normal_ppf(np.clip(q, 1e-15, 1.0 - 1e-15))
    z = gamma * (s - a) / v + zn / sv                    # Var(Z | S) = 1 / (1 + gamma^2)
    y = C.BASE - beta * z + tau * x + sigma * eps
    return z, x, y


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Zwei-Stichproben-Kolmogorov-Smirnov: (Abstand D, asymptotischer p-Wert)."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    grid = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b))))
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = math.sqrt(ne) * d
    if lam < 0.2:
        return d, 1.0
    k = np.arange(1, 101)
    p = 2.0 * float(np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k ** 2 * lam ** 2)))
    return d, min(1.0, max(0.0, p))


# ----------------------------------------------------------------------------- Mini-Instanz

def standardized_difference(effects, weights) -> float:
    """Gewichtetes Mittel der Unterschiede je Stratum (Standardisierung); Gewichte werden normiert."""
    w = np.asarray(weights, dtype=float)
    return float(np.dot(np.asarray(effects, dtype=float), w / w.sum()))
