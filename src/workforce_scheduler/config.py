"""
Configuration module for the workforce scheduler.
Supports loading YAML configuration files and validating parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ScenarioConfig:
    name: str = "default_workforce"
    num_workers: int = 20
    operating_days: int = 5
    hours_per_day: List[int] = field(default_factory=lambda: [11, 11, 11, 11, 7])
    demand_per_slot: int = 4
    max_weekly_hours: float = 25.0
    min_shift_length: float = 1.0
    slot_duration_hours: float = 1.0

    @property
    def total_slots(self) -> int:
        return sum(self.hours_per_day)

    @property
    def total_required_worker_hours(self) -> float:
        return self.total_slots * self.demand_per_slot * self.slot_duration_hours


@dataclass
class PreferencesConfig:
    pref_weekly_hours_min: float = 5.0
    pref_weekly_hours_max: float = 15.0
    pref_shift_length_min: float = 3.0
    pref_shift_length_max: float = 8.0


@dataclass
class FuzzyConfig:
    universe_weekly_hours: List[float] = field(default_factory=lambda: [0.0, 40.0])
    universe_daily_hours: List[float] = field(default_factory=lambda: [0.0, 12.0])
    universe_output: List[float] = field(default_factory=lambda: [0.0, 1.0])
    defuzzification: str = "centroid"
    continuity_scaling_factor: float = 1.30


@dataclass
class GAConfig:
    population_size: int = 50
    max_generations: int = 20
    stall_generations: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elite_count: int = 5
    scenarios_per_candidate: int = 5
    chromosome_size: int = 34
    random_seed: int = 42
    selection_type: str = "roulette_wheel"


@dataclass
class SchedulerConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    preferences: PreferencesConfig = field(default_factory=PreferencesConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)
    ga: GAConfig = field(default_factory=GAConfig)

    def validate(self) -> List[str]:
        errors = []
        if self.ga.elite_count >= self.ga.population_size:
            errors.append(f"Elite count ({self.ga.elite_count}) must be less than population size ({self.ga.population_size}).")
        if not (0.0 <= self.ga.crossover_rate <= 1.0):
            errors.append(f"Crossover rate ({self.ga.crossover_rate}) must be between 0 and 1.")
        if not (0.0 <= self.ga.mutation_rate <= 1.0):
            errors.append(f"Mutation rate ({self.ga.mutation_rate}) must be between 0 and 1.")
        if self.ga.population_size <= 0:
            errors.append("Population size must be positive.")
        if self.ga.max_generations <= 0:
            errors.append("Max generations must be positive.")
        if self.scenario.max_weekly_hours <= 0:
            errors.append("Max weekly hours must be positive.")
        if self.scenario.demand_per_slot <= 0:
            errors.append("Demand per slot must be positive.")
        if len(self.scenario.hours_per_day) != self.scenario.operating_days:
            errors.append(f"Hours per day list length ({len(self.scenario.hours_per_day)}) must equal operating days ({self.scenario.operating_days}).")
        return errors


def load_config(path: Optional[Path] = None) -> SchedulerConfig:
    """Load configuration from a YAML file or return default configuration."""
    if path is None or not Path(path).exists():
        return SchedulerConfig()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    scenario_data = data.get("scenario", {})
    pref_data = data.get("preferences", {})
    fuzzy_data = data.get("fuzzy", {})
    ga_data = data.get("genetic_algorithm", {})

    cfg = SchedulerConfig(
        scenario=ScenarioConfig(**{k: v for k, v in scenario_data.items() if hasattr(ScenarioConfig, k)}),
        preferences=PreferencesConfig(**{k: v for k, v in pref_data.items() if hasattr(PreferencesConfig, k)}),
        fuzzy=FuzzyConfig(**{k: v for k, v in fuzzy_data.items() if hasattr(FuzzyConfig, k)}),
        ga=GAConfig(**{k: v for k, v in ga_data.items() if hasattr(GAConfig, k)}),
    )
    errors = cfg.validate()
    if errors:
        raise ValueError(f"Invalid configuration in {path}: {'; '.join(errors)}")
    return cfg
