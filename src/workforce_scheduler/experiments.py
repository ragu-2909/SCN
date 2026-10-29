"""
Experiment Management and Comparative Evaluation Module.
Executes reproducible comparative experiments across:
1. GA-Optimized Fuzzy Scheduler
2. Non-GA Baseline (Fixed Expert Fuzzy)
3. Preference-Aware Heuristic
Across both standard (20 workers) and understaffed (16 workers) test scenarios.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .baselines import run_non_ga_fuzzy_baseline, run_heuristic_baseline
from .genetic_algorithm import GeneticAlgorithm, GAResult
from .metrics import compute_schedule_metrics, aggregate_experiment_metrics, ScheduleMetrics
from .preprocessing import PreprocessedData
from .scenario_generator import SyntheticPool, generate_training_scenarios, generate_testing_scenarios
from .scheduler import generate_schedule


@dataclass
class ComparisonRunResult:
    method_name: str
    schedules: List[np.ndarray]
    metrics_list: List[ScheduleMetrics]
    summary_stats: Dict[str, Any]
    total_runtime_seconds: float


@dataclass
class ExperimentSuiteResult:
    timestamp: str
    run_id: str
    ga_result: Optional[GAResult]
    comparison_results: Dict[str, ComparisonRunResult]
    standard_summary_df: pd.DataFrame
    understaffed_summary_df: Optional[pd.DataFrame] = None
    config_dict: Dict[str, Any] = field(default_factory=dict)


def run_comparative_experiment(
    train_scenarios: List[PreprocessedData],
    test_scenarios_standard: List[PreprocessedData],
    test_scenarios_understaffed: Optional[List[PreprocessedData]] = None,
    ga_params: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> ExperimentSuiteResult:
    """
    Execute full comparative benchmark between GA-Fuzzy, Non-GA Fuzzy, and Heuristic.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"exp_{timestamp}"

    if ga_params is None:
        ga_params = {
            "population_size": 30,
            "max_generations": 15,
            "stall_generations": 5,
            "crossover_rate": 0.8,
            "mutation_rate": 0.2,
            "elite_count": 5,
            "random_seed": 42
        }

    # 1. Train GA
    if progress_callback:
        progress_callback("Training Genetic Algorithm...", 0.1)

    ga = GeneticAlgorithm(**ga_params)
    ga_result = ga.optimize(train_scenarios)

    methods = [
        ("GA-Optimized Fuzzy", ga_result.best_fuzzy_system),
        ("Non-GA Fuzzy Baseline", None),
        ("Preference Heuristic", "heuristic")
    ]

    standard_results: Dict[str, ComparisonRunResult] = {}

    for idx, (method_name, engine) in enumerate(methods):
        if progress_callback:
            progress_callback(f"Evaluating {method_name} on test set...", 0.4 + 0.2 * idx)

        start_t = time.time()
        scheds = []
        mets = []

        for sc in test_scenarios_standard:
            sc_start = time.time()
            if engine == "heuristic":
                sched, _, _ = run_heuristic_baseline(sc)
            elif engine is None:
                sched, _, _ = run_non_ga_fuzzy_baseline(sc)
            else:
                sched, _, _ = generate_schedule(sc, engine)
            sc_dur = time.time() - sc_start

            metric = compute_schedule_metrics(
                schedule=sched,
                data=sc,
                method_name=method_name,
                runtime_seconds=sc_dur
            )
            scheds.append(sched)
            mets.append(metric)

        total_t = time.time() - start_t
        stats = aggregate_experiment_metrics(mets)
        stats["Total Runtime (s)"] = total_t

        standard_results[method_name] = ComparisonRunResult(
            method_name=method_name,
            schedules=scheds,
            metrics_list=mets,
            summary_stats=stats,
            total_runtime_seconds=total_t
        )

    # Build summary DataFrame
    summary_rows = [res.summary_stats for res in standard_results.values()]
    summary_df = pd.DataFrame(summary_rows)

    # Understaffed test if requested
    understaffed_df = None
    if test_scenarios_understaffed:
        if progress_callback:
            progress_callback("Running understaffed robustness evaluations...", 0.9)

        under_rows = []
        for method_name, engine in methods:
            start_t = time.time()
            mets = []
            for sc in test_scenarios_understaffed:
                sc_start = time.time()
                if engine == "heuristic":
                    sched, _, _ = run_heuristic_baseline(sc)
                elif engine is None:
                    sched, _, _ = run_non_ga_fuzzy_baseline(sc)
                else:
                    sched, _, _ = generate_schedule(sc, engine)
                sc_dur = time.time() - sc_start
                mets.append(compute_schedule_metrics(sched, sc, method_name, sc_dur))
            st = aggregate_experiment_metrics(mets)
            st["Condition"] = "Understaffed (16 workers)"
            under_rows.append(st)
        understaffed_df = pd.DataFrame(under_rows)

    if progress_callback:
        progress_callback("Completed all experiment evaluations.", 1.0)

    return ExperimentSuiteResult(
        timestamp=timestamp,
        run_id=run_id,
        ga_result=ga_result,
        comparison_results=standard_results,
        standard_summary_df=summary_df,
        understaffed_summary_df=understaffed_df,
        config_dict=ga_params
    )
