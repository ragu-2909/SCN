"""Unit tests for scheduler generation module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from workforce_scheduler.data_loader import load_tabular_data
from workforce_scheduler.fuzzy_system import WorkforceFuzzySystem
from workforce_scheduler.preprocessing import build_canonical_data
from workforce_scheduler.scheduler import generate_schedule, extract_shifts_dataframe
from workforce_scheduler.schema_detection import detect_column_mappings
from workforce_scheduler.validation import validate_workers_table, validate_availability_table

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


@pytest.fixture
def sample_preprocessed():
    w_df, _ = load_tabular_data(DATA_DIR / "sample_workers.csv", "sample_workers.csv")
    a_df, _ = load_tabular_data(DATA_DIR / "sample_availability.csv", "sample_availability.csv")
    d_df, _ = load_tabular_data(DATA_DIR / "sample_demand.csv", "sample_demand.csv")

    w_clean, _, _ = validate_workers_table(w_df, detect_column_mappings(w_df))
    a_clean, _, _ = validate_availability_table(a_df, detect_column_mappings(a_df), list(w_clean["worker_id"]))

    return build_canonical_data(w_clean, a_clean, d_df)


def test_generate_schedule_feasibility(sample_preprocessed):
    fuzzy_sys = WorkforceFuzzySystem()
    sched, check_res, meta = generate_schedule(sample_preprocessed, fuzzy_sys)

    assert sched.shape == (20, 51)
    # The sample scenario should produce a valid schedule meeting hard constraints
    assert check_res.unavailable_assignments == 0
    assert check_res.max_hour_violations == 0
    assert check_res.min_shift_violations == 0
    assert check_res.uncovered_slots == 0
    assert check_res.is_valid is True


def test_extract_shifts_dataframe(sample_preprocessed):
    fuzzy_sys = WorkforceFuzzySystem()
    sched, _, _ = generate_schedule(sample_preprocessed, fuzzy_sys)

    shifts_df = extract_shifts_dataframe(sched, sample_preprocessed)
    assert isinstance(shifts_df, pd.DataFrame)
    assert len(shifts_df) > 0
    assert "worker_id" in shifts_df.columns
    assert "duration_hours" in shifts_df.columns
    assert (shifts_df["duration_hours"] >= 1.0).all()
