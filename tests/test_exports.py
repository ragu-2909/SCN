"""Unit tests for exports module."""

from pathlib import Path
import numpy as np
import pytest

from workforce_scheduler.exports import (
    export_schedule_csv,
    export_worker_summary_csv,
    export_slot_coverage_csv,
    export_convergence_csv,
    export_run_metadata_json,
    export_figure_png
)
from workforce_scheduler.genetic_algorithm import GenerationLog
from workforce_scheduler.preprocessing import PreprocessedData
from workforce_scheduler.visualizations import plot_ga_convergence_matplotlib


def test_export_all_assets(tmp_path: Path):
    worker_ids = ["W1", "W2"]
    pref_weekly = np.array([5.0, 5.0])
    pref_shift = np.array([3.0, 3.0])
    max_weekly = np.array([25.0, 25.0])
    avail = np.ones((2, 4), dtype=int)
    demand = np.array([1, 1, 1, 1])
    slot_days = np.array([0, 0, 0, 0])
    slot_hours = np.array([0, 1, 2, 3])
    labels = ["Slot 1", "Slot 2", "Slot 3", "Slot 4"]

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
    sched = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])

    # 1. Schedule CSVs
    mat_p, shf_p = export_schedule_csv(sched, data, tmp_path / "schedules", "test")
    assert mat_p.exists()
    assert shf_p.exists()

    # 2. Worker summary CSV
    w_sum_p = export_worker_summary_csv(sched, data, tmp_path / "metrics" / "workers.csv")
    assert w_sum_p.exists()

    # 3. Slot coverage CSV
    cov_p = export_slot_coverage_csv(sched, data, tmp_path / "metrics" / "coverage.csv")
    assert cov_p.exists()

    # 4. Convergence CSV
    history = [GenerationLog(0, 5.0, 6.0, 6.0, 1.0, 0.1)]
    conv_p = export_convergence_csv(history, tmp_path / "metrics" / "conv.csv")
    assert conv_p.exists()

    # 5. Metadata JSON
    meta_p = export_run_metadata_json({"run_id": "test_123"}, tmp_path / "reports" / "meta.json")
    assert meta_p.exists()

    # 6. Figure PNG
    fig = plot_ga_convergence_matplotlib(history)
    fig_p = export_figure_png(fig, tmp_path / "figures" / "convergence.png")
    assert fig_p.exists()
    assert fig_p.stat().st_size > 0
