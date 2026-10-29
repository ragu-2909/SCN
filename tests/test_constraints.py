"""Unit tests for hard constraint validation module."""

import numpy as np
import pytest

from workforce_scheduler.constraints import validate_schedule_constraints
from workforce_scheduler.preprocessing import PreprocessedData


@pytest.fixture
def sample_toy_data():
    # 3 workers, 4 slots (1 day)
    worker_ids = ["W1", "W2", "W3"]
    pref_weekly = np.array([4.0, 4.0, 4.0])
    pref_shift = np.array([2.0, 2.0, 2.0])
    max_weekly = np.array([4.0, 4.0, 4.0])
    # W1 available all slots, W2 available slots 0,1, W3 available slots 2,3
    avail = np.array([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    demand = np.array([2, 2, 2, 2])
    slot_days = np.array([0, 0, 0, 0])
    slot_hours = np.array([0, 1, 2, 3])
    labels = ["S0", "S1", "S2", "S3"]

    return PreprocessedData(
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


def test_valid_toy_schedule(sample_toy_data):
    # Feasible schedule meeting demand of 2 per slot:
    # W1 works all 4 slots (4h <= 4h max)
    # W2 works slots 0, 1 (2h <= 4h max)
    # W3 works slots 2, 3 (2h <= 4h max)
    sched = np.array([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    res = validate_schedule_constraints(sched, sample_toy_data)
    assert res.is_valid is True
    assert res.unavailable_assignments == 0
    assert res.uncovered_slots == 0
    assert res.max_hour_violations == 0
    assert res.coverage_rate == 1.0


def test_unavailable_assignment_violation(sample_toy_data):
    # W3 is assigned to slot 0 where W3 is unavailable (avail[2,0] == 0)
    sched = np.array([
        [1, 1, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 1]  # W3 working slot 0 is invalid
    ])
    res = validate_schedule_constraints(sched, sample_toy_data)
    assert res.is_valid is False
    assert res.unavailable_assignments >= 1


def test_weekly_cap_violation(sample_toy_data):
    # W2 max is 4.0, but works 5 slots if we extended slots
    sample_toy_data.max_weekly_hours[1] = 1.0  # Set cap to 1.0h
    sched = np.array([
        [1, 1, 1, 1],
        [1, 1, 0, 0],  # 2.0h > 1.0h cap!
        [0, 0, 1, 1]
    ])
    res = validate_schedule_constraints(sched, sample_toy_data)
    assert res.is_valid is False
    assert res.max_hour_violations == 1
