"""
Export Module.
Handles exporting schedules, metrics, convergence data, metadata, and figures
to standard formats (CSV, JSON, PNG) under outputs/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .genetic_algorithm import GenerationLog
from .preprocessing import PreprocessedData
from .scheduler import extract_shifts_dataframe


def export_schedule_csv(
    schedule: np.ndarray,
    data: PreprocessedData,
    output_dir: Path,
    filename_prefix: str = "schedule"
) -> Tuple[Path, Path]:
    """
    Export both the binary slot matrix CSV and the tabular shift roster CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Binary slot matrix CSV
    df_matrix = pd.DataFrame(
        schedule,
        index=data.worker_ids,
        columns=data.slot_labels
    )
    df_matrix.index.name = "worker_id"
    matrix_path = output_dir / f"{filename_prefix}_matrix.csv"
    df_matrix.to_csv(matrix_path)

    # 2. Shift-level roster CSV
    df_shifts = extract_shifts_dataframe(schedule, data)
    shifts_path = output_dir / f"{filename_prefix}_shifts.csv"
    df_shifts.to_csv(shifts_path, index=False)

    return matrix_path, shifts_path


def export_worker_summary_csv(
    schedule: np.ndarray,
    data: PreprocessedData,
    output_path: Path
) -> Path:
    """Export per-worker assignment breakdown and preference adherence."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assigned_weekly = np.sum(schedule, axis=1)
    avail_pct = np.mean(data.availability_matrix, axis=1) * 100

    df_summary = pd.DataFrame({
        "worker_id": data.worker_ids,
        "preferred_weekly_hours": data.pref_weekly_hours,
        "assigned_weekly_hours": assigned_weekly,
        "hours_difference": assigned_weekly - data.pref_weekly_hours,
        "absolute_difference": np.abs(assigned_weekly - data.pref_weekly_hours),
        "preferred_shift_length": data.pref_shift_lengths,
        "max_weekly_hours": data.max_weekly_hours,
        "availability_percentage": np.round(avail_pct, 1)
    })
    df_summary.to_csv(output_path, index=False)
    return output_path


def export_slot_coverage_csv(
    schedule: np.ndarray,
    data: PreprocessedData,
    output_path: Path
) -> Path:
    """Export time-slot staffing coverage CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assigned_per_slot = np.sum(schedule, axis=0)
    df_cov = pd.DataFrame({
        "slot_id": np.arange(data.num_slots),
        "slot_label": data.slot_labels,
        "day": [f"Day {d + 1}" for d in data.slot_days],
        "required_demand": data.demand,
        "assigned_workers": assigned_per_slot,
        "is_covered": assigned_per_slot >= data.demand
    })
    df_cov.to_csv(output_path, index=False)
    return output_path


def export_convergence_csv(history: List[GenerationLog], output_path: Path) -> Path:
    """Export GA convergence trajectory to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "generation": h.generation,
            "best_fitness": h.best_fitness,
            "mean_fitness": h.mean_fitness,
            "median_fitness": h.median_fitness,
            "feasible_ratio": h.feasible_ratio,
            "elapsed_seconds": round(h.elapsed_seconds, 4)
        }
        for h in history
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def export_run_metadata_json(meta: Dict[str, Any], output_path: Path) -> Path:
    """Export run configuration and provenance metadata to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return output_path


def export_figure_png(fig: plt.Figure, output_path: Path, dpi: int = 300) -> Path:
    """Save matplotlib figure to PNG file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
