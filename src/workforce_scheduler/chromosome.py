"""
Chromosome Representation and Encoding Module.
Chromosome length = 34:
- 16 genes: FIS1 Rule Table (4x4 integers in 1..5)
-  9 genes: FIS2 Rule Table (3x3 integers in 1..5)
-  9 genes: Output Triangular MFs shape parameters (continuous floats in 0..1)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .fuzzy_system import INITIAL_FIS1_RULES, INITIAL_FIS2_RULES
from .membership_functions import (
    DEFAULT_OUTPUT_MFS,
    decode_output_mfs_from_genes,
    encode_output_mfs_to_genes
)

CHROMOSOME_LENGTH = 34
NUM_FIS1_RULES = 16
NUM_FIS2_RULES = 9
NUM_MF_GENES = 9


@dataclass
class Chromosome:
    genes: List[Union[int, float]]
    fitness: Optional[float] = None
    is_feasible: bool = False

    def copy(self) -> "Chromosome":
        return Chromosome(genes=list(self.genes), fitness=self.fitness, is_feasible=self.is_feasible)

    def validate(self) -> bool:
        if len(self.genes) != CHROMOSOME_LENGTH:
            return False
        # Validate FIS1 rule genes
        for g in self.genes[:NUM_FIS1_RULES]:
            if not isinstance(g, (int, np.integer)) or not (1 <= g <= 5):
                return False
        # Validate FIS2 rule genes
        for g in self.genes[NUM_FIS1_RULES:NUM_FIS1_RULES + NUM_FIS2_RULES]:
            if not isinstance(g, (int, np.integer)) or not (1 <= g <= 5):
                return False
        # Validate MF genes
        for g in self.genes[NUM_FIS1_RULES + NUM_FIS2_RULES:]:
            if not (0.0 <= float(g) <= 1.0):
                return False
        return True

    def repair(self) -> "Chromosome":
        """Repair any genes outside valid boundaries."""
        new_genes = []
        # Repair FIS1 rules
        for g in self.genes[:NUM_FIS1_RULES]:
            val = int(np.clip(round(float(g)), 1, 5))
            new_genes.append(val)
        # Repair FIS2 rules
        for g in self.genes[NUM_FIS1_RULES:NUM_FIS1_RULES + NUM_FIS2_RULES]:
            val = int(np.clip(round(float(g)), 1, 5))
            new_genes.append(val)
        # Repair MF genes
        for g in self.genes[NUM_FIS1_RULES + NUM_FIS2_RULES:]:
            val = float(np.clip(float(g), 0.01, 0.99))
            new_genes.append(val)
        self.genes = new_genes
        return self


def create_initial_chromosome() -> Chromosome:
    """Create chromosome corresponding to paper's initial baseline rules & MFs."""
    genes: List[Union[int, float]] = []
    # FIS1: 16 genes
    genes.extend(list(INITIAL_FIS1_RULES.flatten()))
    # FIS2: 9 genes
    genes.extend(list(INITIAL_FIS2_RULES.flatten()))
    # Output MFs: 9 genes
    mf_genes = encode_output_mfs_to_genes(DEFAULT_OUTPUT_MFS)
    genes.extend(mf_genes)
    chrom = Chromosome(genes=genes)
    chrom.repair()
    return chrom


def create_random_chromosome(rng: np.random.Generator) -> Chromosome:
    """Create a valid randomly initialized chromosome."""
    genes: List[Union[int, float]] = []
    # FIS1: 16 discrete integers in 1..5
    genes.extend(list(rng.integers(1, 6, size=NUM_FIS1_RULES)))
    # FIS2: 9 discrete integers in 1..5
    genes.extend(list(rng.integers(1, 6, size=NUM_FIS2_RULES)))
    # Output MFs: 9 sorted random floats in (0.05, 0.95)
    mf_genes = list(np.sort(rng.uniform(0.05, 0.95, size=NUM_MF_GENES)))
    genes.extend(mf_genes)
    chrom = Chromosome(genes=genes)
    chrom.repair()
    return chrom


def decode_chromosome(
    chrom: Chromosome
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[float, float, float]]]:
    """
    Decode chromosome into:
    - fis1_rules: 4x4 matrix of ints in 1..5
    - fis2_rules: 3x3 matrix of ints in 1..5
    - output_mfs: dict of 5 triangular membership functions
    """
    chrom.repair()
    fis1_arr = np.array(chrom.genes[:NUM_FIS1_RULES], dtype=int).reshape((4, 4))
    fis2_arr = np.array(chrom.genes[NUM_FIS1_RULES:NUM_FIS1_RULES + NUM_FIS2_RULES], dtype=int).reshape((3, 3))
    mf_genes = [float(g) for g in chrom.genes[NUM_FIS1_RULES + NUM_FIS2_RULES:]]
    output_mfs = decode_output_mfs_from_genes(mf_genes)
    return fis1_arr, fis2_arr, output_mfs
