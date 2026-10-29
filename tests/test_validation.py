"""Unit tests for validation and feasibility checking module."""

import numpy as np
import pandas as pd
import pytest

from workforce_scheduler.validation import (
    validate_workers_table,
    validate_availability_table,
    check_pre_optimization_feasibility
)
from workforce_scheduler.schema_detection import detect_column_mappings


def test_validate_workers_table_valid():
    df = pd.DataFrame({
        "worker": ["W1", "W2"],
        "desired_hours": [10.0, 15.0],
        "preferred_shift": [4.0, 5.0],
        "hour_cap": [25.0, 25.0]
    })
    mapping = detect_column_mappings(df)
    cleaned, errors, warnings = validate_workers_table(df, mapping)
    assert len(errors) == 0
    assert len(cleaned) == 2
    assert "worker_id" in cleaned.columns


def test_validate_workers_table_duplicate_ids():
    df = pd.DataFrame({
        "worker_id": ["W1", "W1"],
        "preferred_weekly_hours": [10.0, 15.0],
        "preferred_shift_length": [4.0, 5.0]
    })
    mapping = {"worker_id": "worker_id", "preferred_weekly_hours": "preferred_weekly_hours", "preferred_shift_length": "preferred_shift_length"}
    cleaned, errors, warnings = validate_workers_table(df, mapping)
    assert len(errors) > 0
    assert any("Duplicate worker IDs" in e for e in errors)


def test_validate_workers_table_invalid_shift_length():
    df = pd.DataFrame({
        "worker_id": ["W1"],
        "preferred_weekly_hours": [10.0],
        "preferred_shift_length": [0.5]  # Less than min 1.0h
    })
    mapping = {"worker_id": "worker_id", "preferred_weekly_hours": "preferred_weekly_hours", "preferred_shift_length": "preferred_shift_length"}
    cleaned, errors, warnings = validate_workers_table(df, mapping, min_shift_length=1.0)
    assert len(errors) > 0
    assert any("at least 1.0 hours" in e for e in errors)


def test_validate_availability_unknown_worker():
    df_avail = pd.DataFrame({
        "worker_id": ["W999"],
        "available": [1],
        "slot_id": [0]
    })
    mapping = {"worker_id": "worker_id", "available": "available", "slot_id": "slot_id"}
    cleaned, errors, warnings = validate_availability_table(df_avail, mapping, known_workers=["W1", "W2"])
    assert len(errors) > 0
    assert any("unknown workers" in e for e in errors)


def test_feasibility_check_infeasible_coverage():
    workers_df = pd.DataFrame({
        "worker_id": ["W1", "W2"],
        "preferred_weekly_hours": [10, 10]
    })
    # Slot 0 has only 1 worker available, but demand is 4 -> Infeasible!
    avail_matrix = np.array([
        [1, 1],
        [0, 1]
    ])
    demand = np.array([4, 1])

    is_feas, metrics, errors, warnings = check_pre_optimization_feasibility(
        workers_df, avail_matrix, demand, ["W1", "W2"]
    )
    assert not is_feas
    assert 0 in metrics.infeasible_slots
    assert len(errors) > 0
