"""Unit tests for fitness evaluation module."""

import numpy as np
import pytest

from workforce_scheduler.fitness import calculate_schedule_rms_terms, evaluate_fitness
from workforce_scheduler.fuzzy_system import WorkforceFuzzySystem
from workforce_scheduler.preprocessing import PreprocessedData


def test_hand_calculated_rms_terms():
    # 2 workers, 2 slots on day 0
    worker_ids = ["W1", "W2"]
    pref_weekly = np.array([2.0, 2.0])
    pref_shift = np.array([2.0, 1.0])
    max_weekly = np.array([25.0, 25.0])
    avail = np.ones((2, 2), dtype=int)
    demand = np.array([1, 1])
    slot_days = np.array([0, 0])
    slot_hours = np.array([0, 1])
    labels = ["S0", "S1"]

    data = PreprocessedData(
        worker_ids=worker_ids,
        pref_weekly_hours=pref_weekly,
        pref_shift_lengths=pref_shift,
        max_weekly_hours=max_weekly,
        availability_matrix=avail,
        demand=demand,
        slot_days=slot_days,
        slot_hours_in_day=slot_hours,
        slot_labels=labels
    )

    # Schedule: W1 works slot 0, 1 (2 hours). W2 works 0 hours.
    sched = np.array([
        [1, 1],
        [0, 0]
    ])

    # Delta 1:
    # W1 works 2h on day 0, preferred shift is 2h -> diff = 0.
    # W2 not assigned on day 0 -> not calculated!
    # RMS(Delta 1) = sqrt(0^2 / 1) = 0.0!

    # Delta 2:
    # W1 assigned 2h, pref 2h -> diff = 0.
    # W2 assigned 0h, pref 2h -> diff = -2.
    # RMS(Delta 2) = sqrt((0^2 + (-2)^2) / 2) = sqrt(4 / 2) = sqrt(2) ~= 1.4142

    # Delta 3:
    # Neither exceeds 25h -> 0.0

    r1, r2, r3 = calculate_schedule_rms_terms(sched, data, slot_hours=1.0)
    assert pytest.approx(r1, abs=1e-4) == 0.0
    assert pytest.approx(r2, abs=1e-4) == np.sqrt(2.0)
    assert pytest.approx(r3, abs=1e-4) == 0.0
