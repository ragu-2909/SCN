"""Unit tests for metrics calculation module."""

import numpy as np
import pytest

from workforce_scheduler.metrics import compute_schedule_metrics, aggregate_experiment_metrics
from workforce_scheduler.preprocessing import PreprocessedData


def test_metrics_calculation():
    worker_ids = ["W1", "W2"]
    pref_weekly = np.array([10.0, 10.0])
    pref_shift = np.array([5.0, 5.0])
    max_weekly = np.array([25.0, 25.0])
    avail = np.ones((2, 10), dtype=int)
    demand = np.full(10, 1, dtype=int)
    slot_days = np.zeros(10, dtype=int)
    slot_hours = np.arange(10, dtype=int)
    labels = [f"S{i}" for i in range(10)]

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

    # 5 slots to W1, 5 slots to W2 (equal 5 hours each)
    sched = np.zeros((2, 10), dtype=int)
    sched[0, :5] = 1
    sched[1, 5:] = 1

    mets = compute_schedule_metrics(sched, data, method_name="TestSched", runtime_seconds=0.1)

    assert mets.is_feasible is True
    assert mets.coverage_rate == 1.0
    assert mets.uncovered_slots == 0
    # Both preferred 10h, received 5h -> MAE = 5.0h
    assert pytest.approx(mets.weekly_preference_mae, abs=1e-3) == 5.0
    # Both received equal hours -> StdDev = 0.0, Jain's index = 1.0!
    assert pytest.approx(mets.weekly_hours_std, abs=1e-3) == 0.0
    assert pytest.approx(mets.jains_fairness_index, abs=1e-3) == 1.0


def test_aggregate_experiment_metrics():
    from workforce_scheduler.metrics import ScheduleMetrics
    m1 = ScheduleMetrics(fitness=5.0, weekly_preference_mae=2.0, is_feasible=True, runtime_seconds=1.0)
    m2 = ScheduleMetrics(fitness=7.0, weekly_preference_mae=4.0, is_feasible=True, runtime_seconds=2.0)

    agg = aggregate_experiment_metrics([m1, m2])
    assert agg["Runs"] == 2
    assert agg["Fitness Mean"] == 6.0
    assert agg["Fitness Min"] == 5.0
    assert agg["Fitness Max"] == 7.0
    assert agg["Feasibility Rate (%)"] == 100.0
