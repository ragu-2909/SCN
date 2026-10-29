"""Unit tests for Mamdani Fuzzy Inference System."""

import numpy as np
import pytest

from workforce_scheduler.fuzzy_system import (
    MamdaniFIS,
    WorkforceFuzzySystem,
    INITIAL_FIS1_RULES,
    INITIAL_FIS2_RULES
)
from workforce_scheduler.membership_functions import (
    DEFAULT_FIS1_INPUT_MFS,
    DEFAULT_OUTPUT_MFS,
    trimf,
    trapmf
)


def test_trimf_and_trapmf():
    # Test triangular MF at peak
    assert trimf(5.0, (0.0, 5.0, 10.0)) == 1.0
    # Test outside range
    assert trimf(12.0, (0.0, 5.0, 10.0)) == 0.0
    # Test midpoint
    assert trimf(2.5, (0.0, 5.0, 10.0)) == 0.5

    # Test trapezoidal MF at plateau
    assert trapmf(5.0, (0.0, 2.0, 8.0, 10.0)) == 1.0
    assert trapmf(12.0, (0.0, 2.0, 8.0, 10.0)) == 0.0


def test_mamdani_fis_evaluation_bounds():
    fis = MamdaniFIS(
        name="TestFIS",
        input1_name="in1",
        input2_name="in2",
        input1_mfs=DEFAULT_FIS1_INPUT_MFS,
        input2_mfs=DEFAULT_FIS1_INPUT_MFS,
        output_mfs=DEFAULT_OUTPUT_MFS,
        rule_matrix=INITIAL_FIS1_RULES
    )

    # Valid inputs must produce score in [0, 1]
    score, details = fis.evaluate(10.0, 5.0)
    assert 0.0 <= score <= 1.0
    assert "centroid" in details
    assert "output_weights" in details


def test_workforce_fuzzy_system_unavailable_worker():
    system = WorkforceFuzzySystem()
    score, s1, s2, cont = system.calculate_suitability(
        pref_weekly=10.0,
        curr_weekly=0.0,
        pref_shift=4.0,
        curr_daily=0.0,
        is_available=0,  # Unavailable
        prev_hour_assigned=False
    )
    # Unavailable worker must receive score of 0.0
    assert score == 0.0


def test_continuity_bonus():
    system = WorkforceFuzzySystem(continuity_scaling_factor=1.35)
    score_no_cont, _, _, cont1 = system.calculate_suitability(12.0, 0.0, 6.0, 0.0, 1, False)
    score_with_cont, _, _, cont2 = system.calculate_suitability(12.0, 0.0, 6.0, 0.0, 1, True)

    assert cont1 == 1.0
    assert cont2 == 1.35
    assert score_with_cont > score_no_cont


def test_decision_explainability():
    system = WorkforceFuzzySystem()
    explanation = system.explain_decision(
        worker_id="W01",
        pref_weekly=12.0,
        curr_weekly=4.0,
        pref_shift=6.0,
        curr_daily=2.0,
        is_available=1,
        prev_hour_assigned=True,
        is_assigned=True,
        rank=1
    )
    assert explanation["worker_id"] == "W01"
    assert explanation["is_available"] is True
    assert explanation["is_assigned"] is True
    assert "reason" in explanation
    assert len(explanation["reason"]) > 20
