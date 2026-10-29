"""
Schedule Generator Module.
Implements the paper's sequential slot assignment algorithm using fuzzy suitability ranking,
multiplicative aggregation, and hard constraint enforcement.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .constraints import validate_schedule_constraints, ConstraintCheckResult
from .fuzzy_system import WorkforceFuzzySystem
from .preprocessing import PreprocessedData


class ShiftRecord:
    def __init__(self, worker_id: str, day: str, start_time: str, end_time: str, duration_hours: float):
        self.worker_id = worker_id
        self.day = day
        self.start_time = start_time
        self.end_time = end_time
        self.duration_hours = duration_hours

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "day": self.day,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_hours": self.duration_hours
        }


def generate_schedule(
    data: PreprocessedData,
    fuzzy_system: WorkforceFuzzySystem,
    slot_hours: float = 1.0,
    enable_repair: bool = True
) -> Tuple[np.ndarray, ConstraintCheckResult, Dict[str, Any]]:
    """
    Generate schedule sequentially from slot 0 to T-1 using fuzzy logic ranking.
    Returns (schedule_matrix, constraint_check_result, execution_metadata).
    """
    W = data.num_workers
    T = data.num_slots
    schedule = np.zeros((W, T), dtype=int)

    assigned_weekly = np.zeros(W, dtype=float)
    assigned_today = np.zeros(W, dtype=float)

    current_day = -1
    slot_decisions = []

    for t in range(T):
        day = data.slot_days[t]
        if day != current_day:
            # New day: reset daily hours
            assigned_today.fill(0.0)
            current_day = day

        required_staff = int(data.demand[t])

        # Find eligible workers: available and within weekly cap
        eligible_workers = []
        for w in range(W):
            if data.availability_matrix[w, t] == 1:
                # Check weekly cap
                if assigned_weekly[w] + slot_hours <= data.max_weekly_hours[w] + 1e-6:
                    eligible_workers.append(w)

        # Calculate scores for candidate workers
        worker_scores = []
        for w in eligible_workers:
            prev_assigned = bool(t > 0 and data.slot_days[t - 1] == day and schedule[w, t - 1] == 1)
            score, s1, s2, cont = fuzzy_system.calculate_suitability(
                pref_weekly=float(data.pref_weekly_hours[w]),
                curr_weekly=float(assigned_weekly[w]),
                pref_shift=float(data.pref_shift_lengths[w]),
                curr_daily=float(assigned_today[w]),
                is_available=1,
                prev_hour_assigned=prev_assigned
            )
            # Add small tie-breaker based on weekly preference deficit to ensure determinism
            deficit = data.pref_weekly_hours[w] - assigned_weekly[w]
            adjusted_score = score + 1e-5 * max(0.0, deficit)
            worker_scores.append((w, adjusted_score, score, s1, s2, cont, prev_assigned))

        # Rank workers descending by score
        worker_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top required_staff workers
        assigned_in_slot = []
        num_to_assign = min(required_staff, len(worker_scores))
        for i in range(num_to_assign):
            w = worker_scores[i][0]
            schedule[w, t] = 1
            assigned_weekly[w] += slot_hours
            assigned_today[w] += slot_hours
            assigned_in_slot.append(w)

        # If eligible count was insufficient and repair is enabled, try emergency fallback
        # to available workers with lowest weekly hours (understaffing mitigation)
        if len(assigned_in_slot) < required_staff:
            rem_avail = [
                w for w in range(W)
                if data.availability_matrix[w, t] == 1 and w not in assigned_in_slot
            ]
            rem_avail.sort(key=lambda w: assigned_weekly[w])
            while len(assigned_in_slot) < required_staff and rem_avail:
                pick = rem_avail.pop(0)
                schedule[pick, t] = 1
                assigned_weekly[pick] += slot_hours
                assigned_today[pick] += slot_hours
                assigned_in_slot.append(pick)

        slot_decisions.append({
            "slot": t,
            "day": day,
            "label": data.slot_labels[t] if t < len(data.slot_labels) else f"Slot {t}",
            "assigned_workers": [data.worker_ids[w] for w in assigned_in_slot],
            "scores": [
                {
                    "worker_id": data.worker_ids[ws[0]],
                    "score": ws[2],
                    "s1": ws[3],
                    "s2": ws[4],
                    "continuity": ws[5],
                    "prev_assigned": ws[6],
                    "selected": ws[0] in assigned_in_slot
                }
                for ws in worker_scores
            ]
        })

    # Validate constraints
    constraint_res = validate_schedule_constraints(
        schedule=schedule,
        data=data,
        min_shift_length=1.0,
        slot_hours=slot_hours
    )

    metadata = {
        "num_assigned_slots": int(np.sum(schedule)),
        "is_feasible": constraint_res.is_valid,
        "slot_decisions": slot_decisions
    }

    return schedule, constraint_res, metadata


def extract_shifts_dataframe(schedule: np.ndarray, data: PreprocessedData) -> pd.DataFrame:
    """
    Convert (W, T) binary schedule matrix into human-readable tabular shift roster.
    Columns: worker_id, day, start_time, end_time, duration_hours.
    """
    W, T = schedule.shape
    shifts = []

    unique_days = np.unique(data.slot_days)
    for day in unique_days:
        day_name = f"Day {day + 1}"
        slot_indices = np.where(data.slot_days == day)[0]
        day_sched = schedule[:, slot_indices]

        for w in range(W):
            row = day_sched[w]
            start_idx = None
            for s_idx, val in enumerate(row):
                if val == 1 and start_idx is None:
                    start_idx = s_idx
                elif val == 0 and start_idx is not None:
                    # End of shift
                    dur = (s_idx - start_idx)
                    start_h = 8 + start_idx
                    end_h = 8 + s_idx
                    shifts.append(ShiftRecord(
                        worker_id=data.worker_ids[w],
                        day=day_name,
                        start_time=f"{start_h:02d}:30",
                        end_time=f"{end_h:02d}:30",
                        duration_hours=float(dur)
                    ).to_dict())
                    start_idx = None

            if start_idx is not None:
                dur = len(row) - start_idx
                start_h = 8 + start_idx
                end_h = 8 + len(row)
                shifts.append(ShiftRecord(
                    worker_id=data.worker_ids[w],
                    day=day_name,
                    start_time=f"{start_h:02d}:30",
                    end_time=f"{end_h:02d}:30",
                    duration_hours=float(dur)
                ).to_dict())

    return pd.DataFrame(shifts)
