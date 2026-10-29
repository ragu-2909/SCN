"""Unit tests for Chromosome encoding and decoding module."""

import numpy as np
import pytest

from workforce_scheduler.chromosome import (
    Chromosome,
    create_initial_chromosome,
    create_random_chromosome,
    decode_chromosome,
    CHROMOSOME_LENGTH,
    NUM_FIS1_RULES,
    NUM_FIS2_RULES,
    NUM_MF_GENES
)


def test_chromosome_length():
    chrom = create_initial_chromosome()
    assert len(chrom.genes) == 34
    assert CHROMOSOME_LENGTH == 34
    assert chrom.validate() is True


def test_decode_initial_chromosome():
    chrom = create_initial_chromosome()
    fis1, fis2, mfs = decode_chromosome(chrom)

    assert fis1.shape == (4, 4)
    assert fis2.shape == (3, 3)
    assert len(mfs) == 5
    assert "Very Low" in mfs
    assert "Very High" in mfs

    # Check valid triangular ordering
    for label, (left, center, right) in mfs.items():
        assert left <= center <= right


def test_random_chromosome_reproducibility():
    rng1 = np.random.default_rng(123)
    c1 = create_random_chromosome(rng1)

    rng2 = np.random.default_rng(123)
    c2 = create_random_chromosome(rng2)

    assert c1.genes == c2.genes


def test_chromosome_repair():
    # Intentionally corrupt some genes
    corrupted_genes = [99] * NUM_FIS1_RULES + [-5] * NUM_FIS2_RULES + [2.5] * NUM_MF_GENES
    chrom = Chromosome(genes=corrupted_genes)
    assert chrom.validate() is False

    chrom.repair()
    assert chrom.validate() is True
    # Values should be clipped
    assert all(1 <= g <= 5 for g in chrom.genes[:NUM_FIS1_RULES + NUM_FIS2_RULES])
    assert all(0.0 <= g <= 1.0 for g in chrom.genes[NUM_FIS1_RULES + NUM_FIS2_RULES:])
