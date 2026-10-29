"""
Baseline Schedulers Module.
Provides comparison benchmarks:
- Baseline A: Non-GA Fixed-Rule Fuzzy Scheduler (Paper initial rule base & MFs)
- Baseline B: Preference-Aware Greedy Heuristic
- Baseline C: Random Eligible Allocation
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from .constraints import validate_schedule_constraints, ConstraintCheckResult
from .fuzzy_system import WorkforceFuzzySystem
from .preprocessing import PreprocessedData
from .scheduler import generate_schedule


def run_non_ga_fuzzy_baseline(
    data: PreprocessedData,
    slot_hours: float = 1.0,
    continuity_factor: float = 1.30
) -> Tuple[np.ndarray, ConstraintCheckResult, Dict[str, Any]]:
    """
    Baseline A: Non-GA Fuzzy Scheduler.
    Uses initial expert rule base from Figure 5 and default output MFs from Figure 6.
    """
    fuzzy_system = WorkforceFuzzySystem(continuity_scaling_factor=continuity_factor)
    return generate_schedule(data, fuzzy_system, slot_hours=slot_hours)


def run_heuristic_baseline(
    data: PreprocessedData,
    slot_hours: float = 1.0,
    continuity_bonus: float = 2.0
) -> Tuple[np.ndarray, ConstraintCheckResult, Dict[str, Any]]:
    """
    Baseline B: Preference-Aware Greedy Heuristic.
    Prioritizes workers with largest weekly preference deficit, daily shift deficit,
    and continuity bonus.
    """
    W = data.num_workers
    T = data.num_slots
    schedule = np.zeros((W, T), dtype=int)

    assigned_weekly = np.zeros(W, dtype=float)
    assigned_today = np.zeros(W, dtype=float)
    current_day = -1

    for t in range(T):
        day = data.slot_days[t]
        if day != current_day:
            assigned_today.fill(0.0)
            current_day = day

        required = int(data.demand[t])

        # Eligible workers
        eligible = []
        for w in range(W):
            if data.availability_matrix[w, t] == 1:
                if assigned_weekly[w] + slot_hours <= data.max_weekly_hours[w] + 1e-6:
                    eligible.append(w)

        # Compute heuristic score
        scores = []
        for w in eligible:
            weekly_deficit = data.pref_weekly_hours[w] - assigned_weekly[w]
            shift_deficit = data.pref_shift_lengths[w] - assigned_today[w]
            prev_assigned = bool(t > 0 and data.slot_days[t - 1] == day and schedule[w, t - 1] == 1)
            bonus = continuity_bonus if prev_assigned else 0.0

            # Priority: weekly deficit + shift deficit + continuity bonus
            priority = weekly_deficit + 0.5 * shift_deficit + bonus
            scores.append((w, priority))

        scores.sort(key=lambda x: x[1], reverse=True)

        assigned_in_slot = []
        num_to_assign = min(required, len(scores))
        for i in range(num_to_assign):
            w = scores[i][0]
            schedule[w, t] = 1
            assigned_weekly[w] += slot_hours
            assigned_today[w] += slot_hours
            assigned_in_slot.append(w)

        # Fallback if understaffed
        if len(assigned_in_slot) < required:
            rem = [w for w in range(W) if data.availability_matrix[w, t] == 1 and w not in assigned_in_slot]
            rem.sort(key=lambda w: assigned_weekly[w])
            while len(assigned_in_slot) < required and rem:
                pick = rem.pop(0)
                schedule[pick, t] = 1
                assigned_weekly[pick] += slot_hours
                assigned_today[pick] += slot_hours
                assigned_in_slot.append(pick)

    check_res = validate_schedule_constraints(
        schedule=schedule,
        data=data,
        min_shift_length=1.0,
        slot_hours=slot_hours
    )

    metadata = {
        "method": "Preference-Aware Heuristic",
        "num_assigned_slots": int(np.sum(schedule)),
        "is_feasible": check_res.is_valid
    }

    return schedule, check_res, metadata


def run_random_baseline(
    data: PreprocessedData,
    seed: int = 42,
    slot_hours: float = 1.0
) -> Tuple[np.ndarray, ConstraintCheckResult, Dict[str, Any]]:
    """
    Baseline C: Random eligible worker assignment (for stress-testing).
    """
    rng = np.random.default_rng(seed)
    W = data.num_workers
    T = data.num_slots
    schedule = np.zeros((W, T), dtype=int)
    assigned_weekly = np.zeros(W, dtype=float)

    for t in range(T):
        required = int(data.demand[t])
        eligible = [
            w for w in range(W)
            if data.availability_matrix[w, t] == 1 and assigned_weekly[w] + slot_hours <= data.max_weekly_hours[w] + 1e-6
        ]

        if len(eligible) > 0:
            num_pick = min(required, len(eligible))
            chosen = rng.choice(eligible, size=num_pick, replace=False)
            for w in chosen:
                schedule[w, t] = 1
                assigned_weekly[w] += slot_hours

    check_res = validate_schedule_constraints(
        schedule=schedule,
        data=data,
        min_shift_length=1.0,
        slot_hours=slot_hours
    )

    return schedule, check_res, {"method": "Random Baseline", "is_feasible": check_res.is_valid}
