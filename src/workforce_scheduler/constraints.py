"""
Hard Constraint Checking and Validation Module.
Enforces and audits operational workforce rules:
1. Availability
2. Staffing Coverage
3. Weekly Working-Hour Cap
4. Single Assignment per Slot
5. Minimum Shift Length
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from .preprocessing import PreprocessedData


@dataclass
class ConstraintCheckResult:
    is_valid: bool = True
    uncovered_slots: int = 0
    coverage_rate: float = 1.0
    unavailable_assignments: int = 0
    max_hour_violations: int = 0
    min_shift_violations: int = 0
    overstaffed_slots: int = 0
    total_assigned_hours: float = 0.0
    worker_assigned_weekly_hours: np.ndarray = field(default_factory=lambda: np.zeros(0))
    violation_details: List[str] = field(default_factory=list)


def validate_schedule_constraints(
    schedule: np.ndarray,             # shape: (num_workers, num_slots), entries in {0, 1}
    data: PreprocessedData,
    min_shift_length: float = 1.0,
    slot_hours: float = 1.0
) -> ConstraintCheckResult:
    """
    Exhaustively evaluate all hard constraints on a candidate schedule.
    Returns ConstraintCheckResult.
    """
    W, T = schedule.shape
    violations = []

    # 1. Availability check: schedule[w, t] <= avail[w, t]
    unavail_mask = (schedule == 1) & (data.availability_matrix == 0)
    unavailable_assignments = int(np.sum(unavail_mask))
    if unavailable_assignments > 0:
        violations.append(f"{unavailable_assignments} assignment(s) made to unavailable workers.")

    # 2. Coverage check
    assigned_per_slot = np.sum(schedule, axis=0)
    understaffed = assigned_per_slot < data.demand
    uncovered_slots = int(np.sum(understaffed))
    overstaffed = assigned_per_slot > data.demand
    overstaffed_slots = int(np.sum(overstaffed))
    coverage_rate = float(np.sum(assigned_per_slot >= data.demand) / T) if T > 0 else 0.0
    if uncovered_slots > 0:
        violations.append(f"{uncovered_slots} slot(s) fail to meet required staffing demand.")

    # 3. Weekly hours maximum check
    worker_weekly_hours = np.sum(schedule, axis=1) * slot_hours
    cap_excess = worker_weekly_hours - data.max_weekly_hours
    max_hour_violations = int(np.sum(cap_excess > 1e-5))
    if max_hour_violations > 0:
        violating_workers = [data.worker_ids[i] for i in range(W) if cap_excess[i] > 1e-5]
        violations.append(
            f"{max_hour_violations} worker(s) exceed their maximum weekly hour limit: {violating_workers[:5]}"
        )

    # 4. Minimum shift duration check (continuous blocks within each day)
    min_shift_slots = int(np.ceil(min_shift_length / slot_hours))
    min_shift_violations = 0

    unique_days = np.unique(data.slot_days)
    for day in unique_days:
        day_slot_indices = np.where(data.slot_days == day)[0]
        day_sched = schedule[:, day_slot_indices]  # shape: (W, slots_in_day)
        for w in range(W):
            row = day_sched[w]
            # Identify consecutive runs of 1s
            runs = []
            curr_len = 0
            for val in row:
                if val == 1:
                    curr_len += 1
                else:
                    if curr_len > 0:
                        runs.append(curr_len)
                        curr_len = 0
            if curr_len > 0:
                runs.append(curr_len)

            # Check if any run < min_shift_slots
            short_shifts = [r for r in runs if r < min_shift_slots]
            if short_shifts:
                min_shift_violations += len(short_shifts)

    if min_shift_violations > 0:
        violations.append(
            f"{min_shift_violations} shift(s) are shorter than the minimum duration of {min_shift_length}h."
        )

    # Schedule is valid iff 0 unavailable assignments, 0 uncovered slots, 0 cap violations, 0 min shift violations
    is_valid = (
        unavailable_assignments == 0 and
        uncovered_slots == 0 and
        max_hour_violations == 0 and
        min_shift_violations == 0
    )

    return ConstraintCheckResult(
        is_valid=is_valid,
        uncovered_slots=uncovered_slots,
        coverage_rate=coverage_rate,
        unavailable_assignments=unavailable_assignments,
        max_hour_violations=max_hour_violations,
        min_shift_violations=min_shift_violations,
        overstaffed_slots=overstaffed_slots,
        total_assigned_hours=float(np.sum(schedule) * slot_hours),
        worker_assigned_weekly_hours=worker_weekly_hours,
        violation_details=violations
    )
