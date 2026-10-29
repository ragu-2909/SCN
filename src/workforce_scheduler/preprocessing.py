"""
Preprocessing module.
Converts raw validated tables into canonical scheduling inputs:
- 2D binary numpy availability matrix (workers x slots)
- 1D integer demand vector
- Slot-to-day and slot-to-hour mappings
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class PreprocessedData:
    def __init__(
        self,
        worker_ids: List[str],
        pref_weekly_hours: np.ndarray,
        pref_shift_lengths: np.ndarray,
        max_weekly_hours: np.ndarray,
        availability_matrix: np.ndarray,  # shape: (W, T)
        demand: np.ndarray,                # shape: (T,)
        slot_days: np.ndarray,             # shape: (T,) day index 0..D-1
        slot_hours_in_day: np.ndarray,     # shape: (T,) hour index in that day
        slot_labels: List[str]             # descriptive string per slot
    ):
        self.worker_ids = worker_ids
        self.pref_weekly_hours = pref_weekly_hours
        self.pref_shift_lengths = pref_shift_lengths
        self.max_weekly_hours = max_weekly_hours
        self.availability_matrix = availability_matrix
        self.demand = demand
        self.slot_days = slot_days
        self.slot_hours_in_day = slot_hours_in_day
        self.slot_labels = slot_labels

    @property
    def num_workers(self) -> int:
        return len(self.worker_ids)

    @property
    def num_slots(self) -> int:
        return len(self.demand)


def build_canonical_data(
    workers_df: pd.DataFrame,
    avail_df_or_matrix: Union[pd.DataFrame, np.ndarray],
    demand_df_or_array: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    default_demand: int = 4,
    hours_per_day: Optional[List[int]] = None
) -> PreprocessedData:
    """
    Construct canonical PreprocessedData object from validated tables.
    """
    if hours_per_day is None:
        hours_per_day = [11, 11, 11, 11, 7]  # Paper reference: 51 slots

    num_slots = sum(hours_per_day)
    worker_ids = list(workers_df["worker_id"].values)
    num_workers = len(worker_ids)
    worker_index = {w: i for i, w in enumerate(worker_ids)}

    # Build slot-to-day mapping
    slot_days = []
    slot_hours_in_day = []
    slot_labels = []
    curr_slot = 0
    for day_idx, h_count in enumerate(hours_per_day):
        day_num = day_idx + 1
        for h in range(h_count):
            slot_days.append(day_idx)
            slot_hours_in_day.append(h)
            start_h = 8 + h
            slot_labels.append(f"Day {day_num} {start_h:02d}:30")
            curr_slot += 1

    slot_days = np.array(slot_days, dtype=int)
    slot_hours_in_day = np.array(slot_hours_in_day, dtype=int)

    # Process availability
    if isinstance(avail_df_or_matrix, np.ndarray):
        avail_matrix = avail_df_or_matrix.astype(int)
    elif "slot_id" in avail_df_or_matrix.columns and "worker_id" in avail_df_or_matrix.columns:
        # Long format
        avail_matrix = np.zeros((num_workers, num_slots), dtype=int)
        for _, row in avail_df_or_matrix.iterrows():
            w = str(row["worker_id"]).strip()
            s = int(row["slot_id"])
            if w in worker_index and 0 <= s < num_slots:
                avail_matrix[worker_index[w], s] = int(row["available"])
    else:
        # Wide matrix: worker_id + slot columns
        cols = [c for c in avail_df_or_matrix.columns if c != "worker_id"]
        avail_matrix = np.zeros((num_workers, num_slots), dtype=int)
        for _, row in avail_df_or_matrix.iterrows():
            w = str(row["worker_id"]).strip()
            if w in worker_index:
                w_i = worker_index[w]
                for s_i, col in enumerate(cols[:num_slots]):
                    avail_matrix[w_i, s_i] = int(row[col])

    # Process demand
    if demand_df_or_array is None:
        demand = np.full(num_slots, default_demand, dtype=int)
    elif isinstance(demand_df_or_array, np.ndarray):
        demand = demand_df_or_array.astype(int)
    elif isinstance(demand_df_or_array, pd.DataFrame):
        if "required_workers" in demand_df_or_array.columns:
            demand = demand_df_or_array["required_workers"].values.astype(int)
        else:
            demand = demand_df_or_array.iloc[:, -1].values.astype(int)
    else:
        demand = np.full(num_slots, default_demand, dtype=int)

    if len(demand) < num_slots:
        # Pad with default demand
        demand = np.pad(demand, (0, num_slots - len(demand)), constant_values=default_demand)
    elif len(demand) > num_slots:
        demand = demand[:num_slots]

    pref_weekly = workers_df["preferred_weekly_hours"].values.astype(float)
    pref_shift = workers_df["preferred_shift_length"].values.astype(float)
    max_weekly = workers_df["max_weekly_hours"].values.astype(float)

    return PreprocessedData(
        worker_ids=worker_ids,
        pref_weekly_hours=pref_weekly,
        pref_shift_lengths=pref_shift,
        max_weekly_hours=max_weekly,
        availability_matrix=avail_matrix,
        demand=demand,
        slot_days=slot_days,
        slot_hours_in_day=slot_hours_in_day,
        slot_labels=slot_labels
    )
