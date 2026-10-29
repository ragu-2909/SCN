"""
Genetic Algorithm (GA) Optimization Module.
Optimizes the 34-gene chromosome (FIS1 rules, FIS2 rules, and shared output MFs)
using Roulette Wheel Selection, Elitism, Mixed Crossover, and Adaptive Mutation.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .chromosome import (
    Chromosome,
    create_initial_chromosome,
    create_random_chromosome,
    decode_chromosome,
    CHROMOSOME_LENGTH,
    NUM_FIS1_RULES,
    NUM_FIS2_RULES,
    NUM_MF_GENES
)
from .fitness import evaluate_fitness, FitnessBreakdown
from .fuzzy_system import WorkforceFuzzySystem
from .preprocessing import PreprocessedData


@dataclass
class GenerationLog:
    generation: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    feasible_ratio: float
    elapsed_seconds: float


@dataclass
class GAResult:
    best_chromosome: Chromosome
    best_fitness: float
    best_fuzzy_system: WorkforceFuzzySystem
    history: List[GenerationLog] = field(default_factory=list)
    total_runtime_seconds: float = 0.0
    stalled_generations: int = 0
    converged: bool = False


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 20,
        stall_generations: int = 5,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elite_count: int = 5,
        random_seed: int = 42,
        selection_type: str = "roulette_wheel"
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.stall_generations = stall_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = min(elite_count, population_size - 1)
        self.random_seed = random_seed
        self.selection_type = selection_type
        self.rng = np.random.default_rng(random_seed)

    def _evaluate_individual(
        self,
        chrom: Chromosome,
        scenarios: List[PreprocessedData]
    ) -> Tuple[float, bool]:
        """Decode chromosome and evaluate its fitness across scenarios."""
        fis1_rules, fis2_rules, output_mfs = decode_chromosome(chrom)
        fuzzy_sys = WorkforceFuzzySystem(
            fis1_rules=fis1_rules,
            fis2_rules=fis2_rules,
            output_mfs=output_mfs
        )
        breakdown = evaluate_fitness(fuzzy_sys, scenarios)
        chrom.fitness = breakdown.total_fitness
        chrom.is_feasible = breakdown.all_feasible
        return breakdown.total_fitness, breakdown.all_feasible

    def _roulette_wheel_selection(self, population: List[Chromosome]) -> Chromosome:
        """Roulette wheel selection converted safely for minimization."""
        fitnesses = np.array([c.fitness if c.fitness is not None else 1e6 for c in population])
        max_f = np.max(fitnesses) + 1.0
        # Invert so lower fitness gets higher weight
        weights = (max_f - fitnesses)
        weights = np.maximum(weights, 1e-4)
        probs = weights / np.sum(weights)

        chosen_idx = self.rng.choice(len(population), p=probs)
        return population[chosen_idx]

    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Mixed crossover for discrete rule genes and continuous MF genes."""
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1_genes = list(parent1.genes)
        child2_genes = list(parent2.genes)

        # 1. Rule crossover: two-point crossover across the 25 rule genes
        num_rules = NUM_FIS1_RULES + NUM_FIS2_RULES
        pt1, pt2 = sorted(self.rng.choice(num_rules, size=2, replace=False))
        child1_genes[pt1:pt2] = parent2.genes[pt1:pt2]
        child2_genes[pt1:pt2] = parent1.genes[pt1:pt2]

        # 2. MF crossover: arithmetic blend
        alpha = self.rng.uniform(0.2, 0.8)
        for i in range(num_rules, CHROMOSOME_LENGTH):
            v1 = float(parent1.genes[i])
            v2 = float(parent2.genes[i])
            child1_genes[i] = alpha * v1 + (1 - alpha) * v2
            child2_genes[i] = (1 - alpha) * v1 + alpha * v2

        c1 = Chromosome(genes=child1_genes).repair()
        c2 = Chromosome(genes=child2_genes).repair()
        return c1, c2

    def _mutate(self, chrom: Chromosome) -> Chromosome:
        """Mutate discrete rules and continuous MF genes."""
        num_rules = NUM_FIS1_RULES + NUM_FIS2_RULES
        genes = list(chrom.genes)

        # Discrete rule mutation
        for i in range(num_rules):
            if self.rng.random() < self.mutation_rate * 0.5:
                genes[i] = int(self.rng.integers(1, 6))

        # Continuous MF mutation
        for i in range(num_rules, CHROMOSOME_LENGTH):
            if self.rng.random() < self.mutation_rate:
                noise = self.rng.normal(0.0, 0.05)
                genes[i] = float(np.clip(float(genes[i]) + noise, 0.01, 0.99))

        chrom.genes = genes
        return chrom.repair()

    def optimize(
        self,
        training_scenarios: List[PreprocessedData],
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None
    ) -> GAResult:
        """
        Execute Genetic Algorithm optimization loop.
        """
        start_time = time.time()

        # 1. Initialize population: seed with paper initial baseline + random
        population: List[Chromosome] = [create_initial_chromosome()]
        for _ in range(self.population_size - 1):
            population.append(create_random_chromosome(self.rng))

        # Evaluate initial population
        for ind in population:
            self._evaluate_individual(ind, training_scenarios)

        population.sort(key=lambda c: c.fitness if c.fitness is not None else 1e9)
        best_candidate = population[0].copy()
        best_fitness = best_candidate.fitness

        history: List[GenerationLog] = []
        stall_counter = 0

        # Log generation 0
        fits = [c.fitness for c in population if c.fitness is not None]
        feas = [1 for c in population if c.is_feasible]
        history.append(GenerationLog(
            generation=0,
            best_fitness=float(np.min(fits)),
            mean_fitness=float(np.mean(fits)),
            median_fitness=float(np.median(fits)),
            feasible_ratio=float(len(feas) / len(population)),
            elapsed_seconds=time.time() - start_time
        ))

        # Evolution loop
        for gen in range(1, self.max_generations + 1):
            gen_start = time.time()

            # Elitism: preserve top elite_count individuals
            next_pop = [ind.copy() for ind in population[:self.elite_count]]

            # Reproduction: Selection, Crossover, Mutation
            while len(next_pop) < self.population_size:
                p1 = self._roulette_wheel_selection(population)
                p2 = self._roulette_wheel_selection(population)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)

                next_pop.append(c1)
                if len(next_pop) < self.population_size:
                    next_pop.append(c2)

            # Evaluate next generation
            for ind in next_pop:
                self._evaluate_individual(ind, training_scenarios)

            next_pop.sort(key=lambda c: c.fitness if c.fitness is not None else 1e9)
            population = next_pop

            current_best = population[0]
            current_fitness = current_best.fitness

            if current_fitness is not None and (best_fitness is None or current_fitness < best_fitness - 1e-4):
                best_fitness = current_fitness
                best_candidate = current_best.copy()
                stall_counter = 0
            else:
                stall_counter += 1

            fits = [c.fitness for c in population if c.fitness is not None]
            feas = [1 for c in population if c.is_feasible]
            history.append(GenerationLog(
                generation=gen,
                best_fitness=float(np.min(fits)),
                mean_fitness=float(np.mean(fits)),
                median_fitness=float(np.median(fits)),
                feasible_ratio=float(len(feas) / len(population)),
                elapsed_seconds=time.time() - start_time
            ))

            if progress_callback:
                progress_callback(gen, self.max_generations, float(best_fitness), float(np.mean(fits)))

            # Stall condition check
            if stall_counter >= self.stall_generations:
                break

        total_time = time.time() - start_time
        fis1, fis2, mfs = decode_chromosome(best_candidate)
        best_fuzzy = WorkforceFuzzySystem(fis1_rules=fis1, fis2_rules=fis2, output_mfs=mfs)

        return GAResult(
            best_chromosome=best_candidate,
            best_fitness=float(best_fitness) if best_fitness is not None else 0.0,
            best_fuzzy_system=best_fuzzy,
            history=history,
            total_runtime_seconds=total_time,
            stalled_generations=stall_counter,
            converged=(stall_counter >= self.stall_generations)
        )
