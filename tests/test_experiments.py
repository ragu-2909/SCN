"""Unit tests for experiment runner module."""

import pytest
from workforce_scheduler.experiments import run_comparative_experiment
from workforce_scheduler.scenario_generator import SyntheticPool, generate_training_scenarios, generate_testing_scenarios


def test_quick_comparative_experiment():
    pool = SyntheticPool(seed=42, total_students=30, train_count=20)
    train_scenarios = generate_training_scenarios(pool, num_scenarios=2, workers_per_scenario=10, seed=10)
    test_scenarios = generate_testing_scenarios(pool, num_scenarios=2, workers_per_scenario=10, seed=20)

    # Fast test parameters
    ga_params = {
        "population_size": 6,
        "max_generations": 2,
        "stall_generations": 2,
        "crossover_rate": 0.8,
        "mutation_rate": 0.2,
        "elite_count": 2,
        "random_seed": 42
    }

    result = run_comparative_experiment(
        train_scenarios=train_scenarios,
        test_scenarios_standard=test_scenarios,
        ga_params=ga_params
    )

    assert result.ga_result is not None
    assert len(result.comparison_results) == 3
    assert "GA-Optimized Fuzzy" in result.comparison_results
    assert "Non-GA Fuzzy Baseline" in result.comparison_results
    assert "Preference Heuristic" in result.comparison_results
    assert not result.standard_summary_df.empty
    assert len(result.standard_summary_df) == 3
