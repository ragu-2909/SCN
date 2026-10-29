"""
Fitness Evaluation Module.
Implements the paper-described fitness function:
Fitness = (1 / n) * sum_{i=1..n} [ RMS(Delta_1) + RMS(Delta_2) + RMS(Delta_3) ]
Where:
- Delta_1: Daily shift length difference for assigned worker-days
- Delta_2: Weekly assigned hours difference from preferred weekly hours
- Delta_3: Penalty for weekly hours exceeding weekly limit (25h)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .fuzzy_system import WorkforceFuzzySystem
from .preprocessing import PreprocessedData
from .scheduler import generate_schedule


@dataclass
class FitnessBreakdown:
    total_fitness: float = 0.0
    mean_rms_delta1: float = 0.0
    mean_rms_delta2: float = 0.0
    mean_rms_delta3: float = 0.0
    feasibility_rate: float = 0.0
    all_feasible: bool = True
    num_scenarios: int = 0
    scenario_fitnesses: List[float] = field(default_factory=list)


def calculate_schedule_rms_terms(
    schedule: np.ndarray,
    data: PreprocessedData,
    slot_hours: float = 1.0
) -> Tuple[float, float, float]:
    """
    Compute RMS(Delta_1), RMS(Delta_2), RMS(Delta_3) for a single schedule.
    """
    W, T = schedule.shape

    # 1. Delta_1: daily shift length difference
    unique_days = np.unique(data.slot_days)
    delta1_sq = []

    for day in unique_days:
        day_slot_indices = np.where(data.slot_days == day)[0]
        day_sched = schedule[:, day_slot_indices]  # shape (W, slots_in_day)
        day_hours = np.sum(day_sched, axis=1) * slot_hours  # shape (W,)

        for w in range(W):
            if day_hours[w] > 0:  # Only calculated if worker is assigned on this day
                diff = day_hours[w] - data.pref_shift_lengths[w]
                delta1_sq.append(diff ** 2)

    rms_delta1 = float(np.sqrt(np.mean(delta1_sq))) if len(delta1_sq) > 0 else 0.0

    # 2. Delta_2: weekly hours difference from preference
    worker_weekly_hours = np.sum(schedule, axis=1) * slot_hours
    delta2 = worker_weekly_hours - data.pref_weekly_hours
    rms_delta2 = float(np.sqrt(np.mean(delta2 ** 2)))

    # 3. Delta_3: weekly hours exceeding maximum limit (25h)
    delta3 = np.maximum(0.0, worker_weekly_hours - data.max_weekly_hours)
    rms_delta3 = float(np.sqrt(np.mean(delta3 ** 2)))

    return rms_delta1, rms_delta2, rms_delta3


def evaluate_fitness(
    fuzzy_system: WorkforceFuzzySystem,
    scenarios: List[PreprocessedData],
    slot_hours: float = 1.0,
    uncovered_demand_penalty: float = 50.0
) -> FitnessBreakdown:
    """
    Evaluate fitness across n scenarios as in Eq. (1) of the paper.
    """
    n = len(scenarios)
    if n == 0:
        return FitnessBreakdown()

    rms_delta1_list = []
    rms_delta2_list = []
    rms_delta3_list = []
    fitness_list = []
    feasible_count = 0

    for sc in scenarios:
        sched, check_res, _ = generate_schedule(sc, fuzzy_system, slot_hours=slot_hours)

        r1, r2, r3 = calculate_schedule_rms_terms(sched, sc, slot_hours=slot_hours)
        base_fitness = r1 + r2 + r3

        # Add penalty for uncovered demand slots if any
        if check_res.uncovered_slots > 0:
            scenario_score = base_fitness + uncovered_demand_penalty * check_res.uncovered_slots
        else:
            scenario_score = base_fitness

        if check_res.is_valid:
            feasible_count += 1

        rms_delta1_list.append(r1)
        rms_delta2_list.append(r2)
        rms_delta3_list.append(r3)
        fitness_list.append(scenario_score)

    mean_r1 = float(np.mean(rms_delta1_list))
    mean_r2 = float(np.mean(rms_delta2_list))
    mean_r3 = float(np.mean(rms_delta3_list))
    mean_fit = float(np.mean(fitness_list))
    feas_rate = float(feasible_count / n)

    return FitnessBreakdown(
        total_fitness=mean_fit,
        mean_rms_delta1=mean_r1,
        mean_rms_delta2=mean_r2,
        mean_rms_delta3=mean_r3,
        feasibility_rate=feas_rate,
        all_feasible=(feasible_count == n),
        num_scenarios=n,
        scenario_fitnesses=fitness_list
    )
