"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """16 Standorte in zwei Gruppen (stabil: Schwankung 0, schwankend: Schwankung 1). Handrechnung:
    behandelt gesamt (2*95 + 6*85)/8 = 87,5; unbehandelt gesamt (6*91 + 2*80)/8 = 88,25; naiver Unterschied -0,75.
    Je Gruppe +4 (stabil: 95 gegen 91) und +5 (schwankend: 85 gegen 80); bei gleichen Gewichten standardisiert +4,5."""
    stable_treated = [94, 96]
    stable_untreated = [90, 91, 91, 91, 91, 92]
    volatile_treated = [84, 85, 85, 85, 85, 86]
    volatile_untreated = [79, 81]
    y = np.array(stable_treated + stable_untreated + volatile_treated + volatile_untreated, dtype=float)
    x = np.array([1] * 2 + [0] * 6 + [1] * 6 + [0] * 2, dtype=float)
    z = np.array([0] * 8 + [1] * 8, dtype=float)
    return y, x, z
