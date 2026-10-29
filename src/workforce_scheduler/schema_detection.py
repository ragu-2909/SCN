"""
Schema detection and column mapping module.
Infers likely canonical workforce scheduling fields from user column names.
"""

import re
from typing import Dict, List, Optional, Tuple
import pandas as pd

CANONICAL_FIELDS = {
    # Workers table
    "worker_id": [
        "worker_id", "worker", "employee_id", "employee", "staff_id", "staff",
        "person_id", "id", "student_id", "assistant_id"
    ],
    "preferred_weekly_hours": [
        "preferred_weekly_hours", "pref_weekly_hours", "desired_hours",
        "preferred_hours", "weekly_preference", "requested_weekly_hours",
        "requested_hours", "weekly_hours", "target_weekly_hours"
    ],
    "preferred_shift_length": [
        "preferred_shift_length", "pref_shift_length", "desired_shift_length",
        "preferred_shift", "shift_preference", "shift_length", "shift_hours",
        "target_shift_length"
    ],
    "max_weekly_hours": [
        "max_weekly_hours", "weekly_max", "max_hours", "weekly_limit",
        "hour_cap", "max_allowed_hours"
    ],
    "role": [
        "role", "skill", "position", "department", "title"
    ],
    # Availability table
    "day": [
        "day", "weekday", "day_of_week", "work_day", "date"
    ],
    "slot_id": [
        "slot_id", "slot", "time_slot", "slot_index", "slot_num"
    ],
    "start_time": [
        "start_time", "time", "hour", "slot_time", "begin_time"
    ],
    "available": [
        "available", "availability", "is_available", "can_work", "status"
    ],
    # Demand table
    "required_workers": [
        "required_workers", "staff_demand", "coverage", "staff_needed",
        "demand", "workers_needed", "min_workers", "headcount"
    ]
}


def normalize_string(s: str) -> str:
    """Normalize column header: lowercase, alphanumeric and underscores only."""
    s = s.strip().lower()
    s = re.sub(r"[\s\-\.\/\\]+", "_", s)
    s = re.sub(r"[^\w]", "", s)
    return s


def detect_column_mappings(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Suggest canonical field mappings for a given DataFrame.
    Returns dict mapping canonical_field -> detected_df_column_name (or None).
    """
    detected: Dict[str, Optional[str]] = {}
    normalized_cols = {col: normalize_string(col) for col in df.columns}

    for canonical, aliases in CANONICAL_FIELDS.items():
        best_col: Optional[str] = None
        for orig_col, norm_col in normalized_cols.items():
            if norm_col == canonical:
                best_col = orig_col
                break
            if norm_col in aliases:
                best_col = orig_col
                break
            # Substring match if no exact alias found
            if any(alias in norm_col for alias in aliases):
                if best_col is None:
                    best_col = orig_col
        detected[canonical] = best_col

    return detected


def identify_table_type(df: pd.DataFrame) -> str:
    """
    Heuristically determine if a table is 'workers', 'availability', 'demand', or 'wide_availability'.
    """
    norm_cols = [normalize_string(c) for c in df.columns]

    if any(k in norm_cols for k in ["required_workers", "staff_demand", "coverage", "demand"]):
        return "demand"
    if any(k in norm_cols for k in ["preferred_weekly_hours", "preferred_shift_length", "desired_hours"]):
        return "workers"
    if any(k in norm_cols for k in ["available", "availability", "is_available"]):
        return "availability"

    # Check for wide availability matrix (worker_id column + multiple time slot columns)
    worker_col = any(k in norm_cols for k in ["worker_id", "worker", "employee_id", "employee"])
    num_numeric_cols = df.select_dtypes(include=["number", "bool"]).shape[1]
    if worker_col and num_numeric_cols >= 5:
        return "wide_availability"

    return "unknown"
