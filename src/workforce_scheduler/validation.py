"""
Validation and Pre-Optimization Feasibility Checking Module.
Ensures data integrity, enforces business constraints, and audits inputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class FeasibilityMetrics:
    total_slots: int = 0
    total_workers: int = 0
    total_required_worker_hours: float = 0.0
    total_preferred_worker_hours: float = 0.0
    total_available_worker_hours: float = 0.0
    min_available_workers_per_slot: int = 0
    max_available_workers_per_slot: int = 0
    avg_available_workers_per_slot: float = 0.0
    infeasible_slots: List[int] = field(default_factory=list)
    zero_availability_workers: List[str] = field(default_factory=list)
    is_feasible: bool = True


@dataclass
class ValidationReport:
    is_valid: bool = True
    is_feasible: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    feasibility: FeasibilityMetrics = field(default_factory=FeasibilityMetrics)
    worker_count: int = 0
    slot_count: int = 0


def validate_workers_table(
    df: pd.DataFrame,
    col_mapping: Dict[str, str],
    min_shift_length: float = 1.0,
    default_max_hours: float = 25.0
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Validate and clean workers DataFrame according to mapping."""
    errors = []
    warnings = []

    worker_col = col_mapping.get("worker_id")
    pref_week_col = col_mapping.get("preferred_weekly_hours")
    pref_shift_col = col_mapping.get("preferred_shift_length")
    max_week_col = col_mapping.get("max_weekly_hours")

    if not worker_col or worker_col not in df.columns:
        errors.append("Workers table missing required 'worker_id' column.")
        return df, errors, warnings
    if not pref_week_col or pref_week_col not in df.columns:
        errors.append("Workers table missing required 'preferred_weekly_hours' column.")
        return df, errors, warnings
    if not pref_shift_col or pref_shift_col not in df.columns:
        errors.append("Workers table missing required 'preferred_shift_length' column.")
        return df, errors, warnings

    cleaned = pd.DataFrame()
    cleaned["worker_id"] = df[worker_col].astype(str).str.strip()

    # Check uniqueness
    if cleaned["worker_id"].duplicated().any():
        dupes = cleaned[cleaned["worker_id"].duplicated()]["worker_id"].unique()
        errors.append(f"Duplicate worker IDs found: {list(dupes)}")

    # Check non-empty
    if (cleaned["worker_id"] == "").any() or cleaned["worker_id"].isna().any():
        errors.append("Worker IDs must not contain empty or missing values.")

    # Validate weekly preferred hours
    cleaned["preferred_weekly_hours"] = pd.to_numeric(df[pref_week_col], errors="coerce")
    if cleaned["preferred_weekly_hours"].isna().any():
        errors.append(f"Non-numeric values found in '{pref_week_col}'.")
    elif (cleaned["preferred_weekly_hours"] <= 0).any():
        errors.append("Preferred weekly hours must be strictly positive.")

    # Validate preferred shift length
    cleaned["preferred_shift_length"] = pd.to_numeric(df[pref_shift_col], errors="coerce")
    if cleaned["preferred_shift_length"].isna().any():
        errors.append(f"Non-numeric values found in '{pref_shift_col}'.")
    elif (cleaned["preferred_shift_length"] < min_shift_length).any():
        errors.append(f"Preferred shift length must be at least {min_shift_length} hours.")

    # Validate max weekly hours
    if max_week_col and max_week_col in df.columns:
        cleaned["max_weekly_hours"] = pd.to_numeric(df[max_week_col], errors="coerce").fillna(default_max_hours)
    else:
        cleaned["max_weekly_hours"] = default_max_hours
        warnings.append(f"No max_weekly_hours specified; defaulting to {default_max_hours}h for all workers.")

    # Check preferred <= max
    if (cleaned["preferred_weekly_hours"] > cleaned["max_weekly_hours"]).any():
        over = cleaned[cleaned["preferred_weekly_hours"] > cleaned["max_weekly_hours"]]
        warnings.append(
            f"{len(over)} workers have preferred hours exceeding max limit. "
            f"Weekly assignments will be capped at max_weekly_hours."
        )

    # Optional role column
    role_col = col_mapping.get("role")
    if role_col and role_col in df.columns:
        cleaned["role"] = df[role_col].astype(str).str.strip()
    else:
        cleaned["role"] = "Part-Time Worker"

    return cleaned, errors, warnings


def validate_availability_table(
    df: pd.DataFrame,
    col_mapping: Dict[str, str],
    known_workers: List[str]
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Validate and clean long-format availability table."""
    errors = []
    warnings = []

    worker_col = col_mapping.get("worker_id")
    avail_col = col_mapping.get("available")
    slot_col = col_mapping.get("slot_id")

    if not worker_col or worker_col not in df.columns:
        errors.append("Availability table missing 'worker_id' column.")
        return df, errors, warnings
    if not avail_col or avail_col not in df.columns:
        errors.append("Availability table missing 'available' column.")
        return df, errors, warnings

    cleaned = pd.DataFrame()
    cleaned["worker_id"] = df[worker_col].astype(str).str.strip()

    # Check known workers
    unknown = set(cleaned["worker_id"]) - set(known_workers)
    if unknown:
        errors.append(f"Availability contains unknown workers: {list(unknown)[:5]}...")

    # Convert availability to 0/1
    raw_vals = df[avail_col]
    if raw_vals.dtype == bool:
        cleaned["available"] = raw_vals.astype(int)
    else:
        converted = pd.to_numeric(raw_vals, errors="coerce")
        if converted.isna().any():
            # Check for strings like 'yes', 'no', 'true', 'false', 'y', 'n'
            str_vals = raw_vals.astype(str).str.strip().str.lower()
            mapping = {"true": 1, "yes": 1, "y": 1, "1": 1, "1.0": 1, "false": 0, "no": 0, "n": 0, "0": 0, "0.0": 0}
            mapped = str_vals.map(mapping)
            if mapped.isna().any():
                errors.append(f"Cannot parse all availability values in '{avail_col}' as binary 0/1.")
            cleaned["available"] = mapped.fillna(0).astype(int)
        else:
            if not converted.isin([0, 1]).all():
                warnings.append("Availability values other than 0/1 detected; positive values will be treated as 1.")
            cleaned["available"] = (converted > 0).astype(int)

    # Slot id
    if slot_col and slot_col in df.columns:
        cleaned["slot_id"] = pd.to_numeric(df[slot_col], errors="coerce")
        if cleaned["slot_id"].isna().any():
            errors.append(f"Invalid slot IDs in '{slot_col}'.")
    else:
        cleaned["slot_id"] = df.groupby("worker_id").cumcount()
        warnings.append("No slot_id specified; inferred sequential slot indices for each worker.")

    # Day and time if present
    day_col = col_mapping.get("day")
    if day_col and day_col in df.columns:
        cleaned["day"] = df[day_col].astype(str).str.strip()
    else:
        cleaned["day"] = "Day 1"

    start_col = col_mapping.get("start_time")
    if start_col and start_col in df.columns:
        cleaned["start_time"] = df[start_col].astype(str).str.strip()
    else:
        cleaned["start_time"] = ""

    # Check for duplicate worker x slot records
    if cleaned.duplicated(subset=["worker_id", "slot_id"]).any():
        errors.append("Duplicate (worker_id, slot_id) entries found in availability table.")

    return cleaned, errors, warnings


def check_pre_optimization_feasibility(
    workers_df: pd.DataFrame,
    avail_matrix: np.ndarray,  # shape: (num_workers, num_slots)
    demand_array: np.ndarray,  # shape: (num_slots,)
    worker_ids: List[str],
    slot_hours: float = 1.0
) -> Tuple[bool, FeasibilityMetrics, List[str], List[str]]:
    """
    Run pre-optimization feasibility checks.
    Returns (is_feasible, metrics, errors, warnings).
    """
    errors = []
    warnings = []
    num_workers, num_slots = avail_matrix.shape

    tot_req = float(np.sum(demand_array) * slot_hours)
    tot_pref = float(workers_df["preferred_weekly_hours"].sum())
    tot_avail = float(np.sum(avail_matrix) * slot_hours)

    per_slot_avail = np.sum(avail_matrix, axis=0)
    min_avail = int(np.min(per_slot_avail))
    max_avail = int(np.max(per_slot_avail))
    avg_avail = float(np.mean(per_slot_avail))

    infeasible_slots = []
    for s in range(num_slots):
        if per_slot_avail[s] < demand_array[s]:
            infeasible_slots.append(int(s))

    worker_avail_sums = np.sum(avail_matrix, axis=1)
    zero_workers = [worker_ids[i] for i in range(num_workers) if worker_avail_sums[i] == 0]

    is_feasible = len(infeasible_slots) == 0

    if infeasible_slots:
        errors.append(
            f"Infeasible scenario: {len(infeasible_slots)} slot(s) have fewer available workers "
            f"than required demand. Slot indices: {infeasible_slots[:10]}..."
        )

    if zero_workers:
        warnings.append(f"{len(zero_workers)} worker(s) have 0 availability: {zero_workers}")

    if tot_avail < tot_req:
        errors.append(
            f"Total available worker-hours ({tot_avail:.1f}h) is less than required ({tot_req:.1f}h)."
        )
    elif tot_avail < tot_req * 1.1:
        warnings.append(
            f"Available worker-hours ({tot_avail:.1f}h) is tightly constrained relative to demand ({tot_req:.1f}h)."
        )

    metrics = FeasibilityMetrics(
        total_slots=num_slots,
        total_workers=num_workers,
        total_required_worker_hours=tot_req,
        total_preferred_worker_hours=tot_pref,
        total_available_worker_hours=tot_avail,
        min_available_workers_per_slot=min_avail,
        max_available_workers_per_slot=max_avail,
        avg_available_workers_per_slot=avg_avail,
        infeasible_slots=infeasible_slots,
        zero_availability_workers=zero_workers,
        is_feasible=is_feasible
    )

    return is_feasible, metrics, errors, warnings
