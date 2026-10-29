"""
Evaluation Metrics Module.
Calculates performance, preference satisfaction, constraint compliance,
fairness, and runtime metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .constraints import validate_schedule_constraints, ConstraintCheckResult
from .fitness import calculate_schedule_rms_terms
from .preprocessing import PreprocessedData


@dataclass
class ScheduleMetrics:
    # 1. Paper Objective Metrics
    fitness: float = 0.0
    rms_delta1_shift: float = 0.0
    rms_delta2_weekly: float = 0.0
    rms_delta3_cap: float = 0.0

    # 2. Preference Satisfaction
    weekly_preference_mae: float = 0.0
    daily_shift_mae: float = 0.0
    preference_satisfaction_rate: float = 0.0  # % workers within tolerance of pref
    preference_tolerance_hours: float = 2.0

    # 3. Hard Constraints
    is_feasible: bool = True
    coverage_rate: float = 1.0
    uncovered_slots: int = 0
    unavailable_assignments: int = 0
    weekly_cap_violations: int = 0
    min_shift_violations: int = 0

    # 4. Fairness
    weekly_hours_std: float = 0.0
    weekly_hours_range: float = 0.0
    zero_hour_workers: int = 0
    total_under_assigned_hours: float = 0.0
    total_over_assigned_hours: float = 0.0
    jains_fairness_index: float = 0.0

    # 5. Efficiency
    runtime_seconds: float = 0.0
    method_name: str = "Fuzzy-GA"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Method": self.method_name,
            "Fitness": round(self.fitness, 4),
            "RMS(Δ1) Daily Shift": round(self.rms_delta1_shift, 4),
            "RMS(Δ2) Weekly Hours": round(self.rms_delta2_weekly, 4),
            "RMS(Δ3) Cap Penalty": round(self.rms_delta3_cap, 4),
            "Weekly MAE (hrs)": round(self.weekly_preference_mae, 2),
            "Shift MAE (hrs)": round(self.daily_shift_mae, 2),
            "Preference Satisfaction (%)": round(self.preference_satisfaction_rate * 100, 1),
            "Coverage Rate (%)": round(self.coverage_rate * 100, 1),
            "Uncovered Slots": self.uncovered_slots,
            "Cap Violations (>25h)": self.weekly_cap_violations,
            "Unavailable Assignments": self.unavailable_assignments,
            "Feasible": self.is_feasible,
            "Weekly Hours StdDev": round(self.weekly_hours_std, 2),
            "Weekly Hours Range": round(self.weekly_hours_range, 2),
            "Zero-Hour Workers": self.zero_hour_workers,
            "Jain's Fairness Index": round(self.jains_fairness_index, 3),
            "Runtime (s)": round(self.runtime_seconds, 4)
        }


def compute_schedule_metrics(
    schedule: np.ndarray,
    data: PreprocessedData,
    method_name: str = "Fuzzy-GA",
    runtime_seconds: float = 0.0,
    slot_hours: float = 1.0,
    pref_tolerance: float = 2.0
) -> ScheduleMetrics:
    """
    Exhaustively compute all evaluation metrics for a single schedule.
    """
    W, T = schedule.shape

    # Hard constraints
    check_res = validate_schedule_constraints(schedule, data, min_shift_length=1.0, slot_hours=slot_hours)

    # RMS terms
    r1, r2, r3 = calculate_schedule_rms_terms(schedule, data, slot_hours=slot_hours)
    fitness = r1 + r2 + r3
    if check_res.uncovered_slots > 0:
        fitness += 50.0 * check_res.uncovered_slots

    # Weekly hours & preference MAE
    assigned_weekly = np.sum(schedule, axis=1) * slot_hours
    diff_weekly = assigned_weekly - data.pref_weekly_hours
    weekly_mae = float(np.mean(np.abs(diff_weekly)))

    within_tol = np.abs(diff_weekly) <= pref_tolerance
    satisfaction_rate = float(np.sum(within_tol) / W) if W > 0 else 0.0

    # Daily shift length MAE over assigned shifts
    unique_days = np.unique(data.slot_days)
    shift_diffs = []
    for day in unique_days:
        day_indices = np.where(data.slot_days == day)[0]
        day_hours = np.sum(schedule[:, day_indices], axis=1) * slot_hours
        for w in range(W):
            if day_hours[w] > 0:
                shift_diffs.append(abs(day_hours[w] - data.pref_shift_lengths[w]))
    shift_mae = float(np.mean(shift_diffs)) if shift_diffs else 0.0

    # Fairness calculations
    hours_std = float(np.std(assigned_weekly))
    hours_range = float(np.max(assigned_weekly) - np.min(assigned_weekly)) if W > 0 else 0.0
    zero_workers = int(np.sum(assigned_weekly == 0))

    under_hours = float(np.sum(np.maximum(0.0, -diff_weekly)))
    over_hours = float(np.sum(np.maximum(0.0, diff_weekly)))

    # Jain's fairness index: (sum x)^2 / (n * sum x^2)
    sum_h = np.sum(assigned_weekly)
    sum_h_sq = np.sum(assigned_weekly ** 2)
    if sum_h_sq > 0:
        jains = float((sum_h ** 2) / (W * sum_h_sq))
    else:
        jains = 1.0

    return ScheduleMetrics(
        fitness=fitness,
        rms_delta1_shift=r1,
        rms_delta2_weekly=r2,
        rms_delta3_cap=r3,
        weekly_preference_mae=weekly_mae,
        daily_shift_mae=shift_mae,
        preference_satisfaction_rate=satisfaction_rate,
        preference_tolerance_hours=pref_tolerance,
        is_feasible=check_res.is_valid,
        coverage_rate=check_res.coverage_rate,
        uncovered_slots=check_res.uncovered_slots,
        unavailable_assignments=check_res.unavailable_assignments,
        weekly_cap_violations=check_res.max_hour_violations,
        min_shift_violations=check_res.min_shift_violations,
        weekly_hours_std=hours_std,
        weekly_hours_range=hours_range,
        zero_hour_workers=zero_workers,
        total_under_assigned_hours=under_hours,
        total_over_assigned_hours=over_hours,
        jains_fairness_index=jains,
        runtime_seconds=runtime_seconds,
        method_name=method_name
    )


def aggregate_experiment_metrics(metrics_list: List[ScheduleMetrics]) -> Dict[str, Any]:
    """Aggregate multiple repeated-run metrics into summary statistics."""
    if not metrics_list:
        return {}

    fits = [m.fitness for m in metrics_list]
    maes = [m.weekly_preference_mae for m in metrics_list]
    shift_maes = [m.daily_shift_mae for m in metrics_list]
    sat_rates = [m.preference_satisfaction_rate for m in metrics_list]
    feas_rates = [1.0 if m.is_feasible else 0.0 for m in metrics_list]
    runtimes = [m.runtime_seconds for m in metrics_list]

    return {
        "Method": metrics_list[0].method_name,
        "Runs": len(metrics_list),
        "Fitness Mean": float(np.mean(fits)),
        "Fitness Std": float(np.std(fits)),
        "Fitness Median": float(np.median(fits)),
        "Fitness Min": float(np.min(fits)),
        "Fitness Max": float(np.max(fits)),
        "Feasibility Rate (%)": float(np.mean(feas_rates) * 100),
        "Weekly MAE Mean (h)": float(np.mean(maes)),
        "Shift MAE Mean (h)": float(np.mean(shift_maes)),
        "Satisfaction Mean (%)": float(np.mean(sat_rates) * 100),
        "Avg Runtime (s)": float(np.mean(runtimes))
    }
