"""
Scenario Generator Module.
Generates training and testing workforce scenarios following the Nguyen & Cohen paper:
- 70-student pool (40 train, 30 test)
- 20-worker scenarios (and 16-worker understaffed scenarios)
- Weekly hours ~ U(5, 15), shift lengths ~ U(3, 8), max 25h
- Realistic slot-level availability matching student profiles
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from .preprocessing import PreprocessedData


class SyntheticPool:
    """Represents the global pool of students (40 train, 30 test)."""
    def __init__(self, seed: int = 42, total_students: int = 70, train_count: int = 40):
        self.rng = np.random.default_rng(seed)
        self.total_students = total_students
        self.train_count = train_count
        self.test_count = total_students - train_count

        self.student_ids = [f"Student_{i+1:02d}" for i in range(total_students)]

        # Sample preferences for all 70 students
        self.pref_weekly = self.rng.integers(5, 16, size=total_students).astype(float)
        self.pref_shift = self.rng.integers(3, 9, size=total_students).astype(float)
        self.max_weekly = np.full(total_students, 25.0)

        # Baseline availability probability for each student (between 0.60 and 0.90)
        self.avail_probabilities = self.rng.uniform(0.60, 0.92, size=total_students)

        # Train and test partition
        all_indices = np.arange(total_students)
        self.train_indices = all_indices[:train_count]
        self.test_indices = all_indices[train_count:]


def generate_scenario(
    pool: SyntheticPool,
    selected_indices: np.ndarray,
    hours_per_day: Optional[List[int]] = None,
    demand_per_slot: int = 4,
    seed: Optional[int] = None
) -> PreprocessedData:
    """
    Generate a single PreprocessedData scenario from a subset of students.
    """
    if hours_per_day is None:
        hours_per_day = [11, 11, 11, 11, 7]  # 51 slots

    num_slots = sum(hours_per_day)
    num_workers = len(selected_indices)
    rng = np.random.default_rng(seed)

    worker_ids = [pool.student_ids[i] for i in selected_indices]
    pref_weekly = pool.pref_weekly[selected_indices].copy()
    pref_shift = pool.pref_shift[selected_indices].copy()
    max_weekly = pool.max_weekly[selected_indices].copy()
    probs = pool.avail_probabilities[selected_indices]

    # Generate availability matrix (num_workers, num_slots)
    avail_matrix = np.zeros((num_workers, num_slots), dtype=int)
    for w in range(num_workers):
        p = probs[w]
        # Simulate realistic daily availability blocks
        for day_idx in range(len(hours_per_day)):
            # Probability student can work this day
            if rng.random() < p + 0.1:
                # Continuous available window of 4-8 hours
                win_len = rng.integers(4, 9)
                start_h = rng.integers(0, max(1, hours_per_day[day_idx] - win_len + 1))
                # Map to global slot index
                day_start_slot = sum(hours_per_day[:day_idx])
                for h in range(start_h, min(hours_per_day[day_idx], start_h + win_len)):
                    avail_matrix[w, day_start_slot + h] = 1

    # Ensure coverage is feasible: every slot has >= demand_per_slot available workers
    for s in range(num_slots):
        while np.sum(avail_matrix[:, s]) < demand_per_slot:
            unavail = np.where(avail_matrix[:, s] == 0)[0]
            if len(unavail) > 0:
                pick = rng.choice(unavail)
                avail_matrix[pick, s] = 1
            else:
                break

    demand = np.full(num_slots, demand_per_slot, dtype=int)

    slot_days = []
    slot_hours = []
    slot_labels = []
    curr = 0
    for day_idx, h_count in enumerate(hours_per_day):
        day_num = day_idx + 1
        for h in range(h_count):
            slot_days.append(day_idx)
            slot_hours.append(h)
            start_h = 8 + h
            slot_labels.append(f"Day {day_num} {start_h:02d}:30")
            curr += 1

    return PreprocessedData(
        worker_ids=worker_ids,
        pref_weekly_hours=pref_weekly,
        pref_shift_lengths=pref_shift,
        max_weekly_hours=max_weekly,
        availability_matrix=avail_matrix,
        demand=demand,
        slot_days=np.array(slot_days, dtype=int),
        slot_hours_in_day=np.array(slot_hours, dtype=int),
        slot_labels=slot_labels
    )


def generate_training_scenarios(
    pool: SyntheticPool,
    num_scenarios: int = 30,
    workers_per_scenario: int = 20,
    seed: int = 100
) -> List[PreprocessedData]:
    """Generate n training scenarios sampled from the 40-student training pool."""
    rng = np.random.default_rng(seed)
    scenarios = []
    for i in range(num_scenarios):
        chosen = rng.choice(pool.train_indices, size=workers_per_scenario, replace=False)
        sc = generate_scenario(pool, chosen, seed=seed + i * 13)
        scenarios.append(sc)
    return scenarios


def generate_testing_scenarios(
    pool: SyntheticPool,
    num_scenarios: int = 200,
    workers_per_scenario: int = 20,
    seed: int = 200
) -> List[PreprocessedData]:
    """Generate testing scenarios sampled from the 30-student testing pool."""
    rng = np.random.default_rng(seed)
    scenarios = []
    for i in range(num_scenarios):
        chosen = rng.choice(pool.test_indices, size=workers_per_scenario, replace=False)
        sc = generate_scenario(pool, chosen, seed=seed + i * 17)
        scenarios.append(sc)
    return scenarios
