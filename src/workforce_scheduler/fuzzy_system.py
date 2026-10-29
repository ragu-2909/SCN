"""
Fuzzy Inference System (FIS) Module.
Implements FIS1 (Weekly workload suitability) and FIS2 (Daily shift suitability)
using pure NumPy vectorized Mamdani inference with centroid defuzzification,
multiplicative aggregation, and decision explainability.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .membership_functions import (
    DEFAULT_FIS1_INPUT_MFS,
    DEFAULT_FIS2_INPUT_MFS,
    DEFAULT_OUTPUT_MFS,
    evaluate_mf,
    trimf
)

OUTPUT_LABELS = ["Very Low", "Low", "Medium", "High", "Very High"]
OUTPUT_LABEL_TO_INT = {label: i + 1 for i, label in enumerate(OUTPUT_LABELS)}
INT_TO_OUTPUT_LABEL = {i + 1: label for i, label in enumerate(OUTPUT_LABELS)}

# Initial paper rule table from Figure 5
INITIAL_FIS1_RULES = np.array([
    [2, 1, 1, 1],  # Preferred: Very Low -> Assigned: [Low, Very Low, Very Low, Very Low]
    [4, 1, 1, 1],  # Preferred: Low      -> Assigned: [High, Very Low, Very Low, Very Low]
    [5, 2, 1, 1],  # Preferred: High     -> Assigned: [Very High, Low, Very Low, Very Low]
    [5, 4, 3, 1]   # Preferred: Very High-> Assigned: [Very High, High, Medium, Very Low]
], dtype=int)

INITIAL_FIS2_RULES = np.array([
    [5, 1, 3],  # Preferred: Low    -> Assigned: [Very High, Very Low, Medium]
    [5, 4, 5],  # Preferred: Medium -> Assigned: [Very High, High, Very High]
    [5, 4, 3]   # Preferred: High   -> Assigned: [Very High, High, Medium]
], dtype=int)


class MamdaniFIS:
    """
    Two-input, single-output Mamdani Fuzzy Inference System.
    """
    def __init__(
        self,
        name: str,
        input1_name: str,
        input2_name: str,
        input1_mfs: Dict[str, Tuple[float, ...]],
        input2_mfs: Dict[str, Tuple[float, ...]],
        output_mfs: Dict[str, Tuple[float, float, float]],
        rule_matrix: np.ndarray,  # shape: (len(input1_mfs), len(input2_mfs)), entries in 1..5
        universe_discretization: int = 101
    ):
        self.name = name
        self.input1_name = input1_name
        self.input2_name = input2_name
        self.input1_mfs = input1_mfs
        self.input2_mfs = input2_mfs
        self.output_mfs = output_mfs
        self.rule_matrix = np.asarray(rule_matrix, dtype=int)

        self.input1_labels = list(input1_mfs.keys())
        self.input2_labels = list(input2_mfs.keys())

        # Pre-discretize output universe [0, 1]
        self.output_universe = np.linspace(0.0, 1.0, universe_discretization)
        # Precompute output MF values over discretized universe
        self.precomputed_output_mfs = np.zeros((5, universe_discretization))
        for idx, label in enumerate(OUTPUT_LABELS):
            self.precomputed_output_mfs[idx] = trimf(self.output_universe, self.output_mfs[label])

    def evaluate(self, val1: float, val2: float) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate Mamdani FIS for scalar inputs (val1, val2).
        Returns (centroid_score, details_dict).
        """
        # 1. Fuzzification
        mu1 = np.array([evaluate_mf(val1, self.input1_mfs[lbl]) for lbl in self.input1_labels])
        mu2 = np.array([evaluate_mf(val2, self.input2_mfs[lbl]) for lbl in self.input2_labels])

        # 2. Rule evaluation (AND = min)
        # firing_strengths matrix shape: (len(input1), len(input2))
        firing = np.minimum(mu1[:, None], mu2[None, :])

        # 3. Aggregate for each of the 5 output linguistic levels (1..5)
        out_weights = np.zeros(5)
        for i in range(len(self.input1_labels)):
            for j in range(len(self.input2_labels)):
                out_level = self.rule_matrix[i, j] - 1  # 0..4
                if 0 <= out_level < 5:
                    if firing[i, j] > out_weights[out_level]:
                        out_weights[out_level] = firing[i, j]

        # 4. Implication & Aggregation over discretized universe
        # shape: (5, universe_discretization)
        clipped_mfs = np.minimum(out_weights[:, None], self.precomputed_output_mfs)
        aggregated = np.max(clipped_mfs, axis=0)

        # 5. Centroid Defuzzification
        total_area = np.sum(aggregated)
        if total_area > 1e-9:
            centroid = float(np.sum(self.output_universe * aggregated) / total_area)
        else:
            centroid = 0.5  # Neutral fallback

        details = {
            "val1": val1,
            "val2": val2,
            "mu1": {lbl: float(mu1[i]) for i, lbl in enumerate(self.input1_labels)},
            "mu2": {lbl: float(mu2[j]) for j, lbl in enumerate(self.input2_labels)},
            "output_weights": {lbl: float(out_weights[k]) for k, lbl in enumerate(OUTPUT_LABELS)},
            "centroid": centroid
        }

        return centroid, details

    def evaluate_batch(self, vals1: np.ndarray, vals2: np.ndarray) -> np.ndarray:
        """
        Fast evaluation for array of workers.
        vals1, vals2 shape: (N,)
        Returns array of scores of shape: (N,)
        """
        N = len(vals1)
        scores = np.empty(N, dtype=float)
        for i in range(N):
            scores[i], _ = self.evaluate(float(vals1[i]), float(vals2[i]))
        return scores


class WorkforceFuzzySystem:
    """
    Combined scheduling fuzzy logic system:
    - FIS1: Weekly workload suitability
    - FIS2: Daily shift suitability
    - Multiplicative aggregation operator (Pi) with availability and continuity factor
    """
    def __init__(
        self,
        fis1_rules: Optional[np.ndarray] = None,
        fis2_rules: Optional[np.ndarray] = None,
        output_mfs: Optional[Dict[str, Tuple[float, float, float]]] = None,
        continuity_scaling_factor: float = 1.30
    ):
        if fis1_rules is None:
            fis1_rules = INITIAL_FIS1_RULES.copy()
        if fis2_rules is None:
            fis2_rules = INITIAL_FIS2_RULES.copy()
        if output_mfs is None:
            output_mfs = DEFAULT_OUTPUT_MFS.copy()

        self.continuity_scaling_factor = continuity_scaling_factor
        self.output_mfs = output_mfs

        self.fis1 = MamdaniFIS(
            name="FIS1_Weekly",
            input1_name="preferred_weekly_hours",
            input2_name="assigned_weekly_hours",
            input1_mfs=DEFAULT_FIS1_INPUT_MFS,
            input2_mfs=DEFAULT_FIS1_INPUT_MFS,
            output_mfs=output_mfs,
            rule_matrix=fis1_rules
        )

        self.fis2 = MamdaniFIS(
            name="FIS2_Daily",
            input1_name="preferred_shift_length",
            input2_name="assigned_daily_hours",
            input1_mfs=DEFAULT_FIS2_INPUT_MFS,
            input2_mfs=DEFAULT_FIS2_INPUT_MFS,
            output_mfs=output_mfs,
            rule_matrix=fis2_rules
        )

    def calculate_suitability(
        self,
        pref_weekly: float,
        curr_weekly: float,
        pref_shift: float,
        curr_daily: float,
        is_available: int,
        prev_hour_assigned: bool
    ) -> Tuple[float, float, float, float]:
        """
        Calculate combined assignment likelihood for a single worker in a single slot.
        Returns (final_score, fis1_score, fis2_score, continuity_mult).
        """
        if is_available == 0:
            return 0.0, 0.0, 0.0, 1.0

        score1, _ = self.fis1.evaluate(pref_weekly, curr_weekly)
        score2, _ = self.fis2.evaluate(pref_shift, curr_daily)

        min_score = min(score1, score2)
        continuity_mult = self.continuity_scaling_factor if prev_hour_assigned else 1.0

        final_score = min_score * continuity_mult
        return final_score, score1, score2, continuity_mult

    def explain_decision(
        self,
        worker_id: str,
        pref_weekly: float,
        curr_weekly: float,
        pref_shift: float,
        curr_daily: float,
        is_available: int,
        prev_hour_assigned: bool,
        is_assigned: bool,
        rank: int
    ) -> Dict[str, Any]:
        """Generate human-readable decision trace for explainable AI."""
        if is_available == 0:
            return {
                "worker_id": worker_id,
                "is_available": False,
                "reason": "Worker is marked unavailable for this slot (Availability = 0).",
                "final_score": 0.0,
                "is_assigned": False,
                "rank": rank
            }

        score1, details1 = self.fis1.evaluate(pref_weekly, curr_weekly)
        score2, details2 = self.fis2.evaluate(pref_shift, curr_daily)
        min_score = min(score1, score2)
        continuity_mult = self.continuity_scaling_factor if prev_hour_assigned else 1.0
        final_score = min_score * continuity_mult

        reason = (
            f"FIS1 score {score1:.3f} (Pref: {pref_weekly}h, Assigned: {curr_weekly}h), "
            f"FIS2 score {score2:.3f} (Pref Shift: {pref_shift}h, Today: {curr_daily}h). "
            f"Min likelihood = {min_score:.3f}. "
            f"Continuity multiplier = {continuity_mult:.2f} (prev assigned: {prev_hour_assigned}). "
            f"Final score = {final_score:.4f}. Rank: #{rank}."
        )

        return {
            "worker_id": worker_id,
            "is_available": True,
            "pref_weekly_hours": pref_weekly,
            "curr_weekly_hours": curr_weekly,
            "pref_shift_length": pref_shift,
            "curr_daily_hours": curr_daily,
            "prev_hour_assigned": prev_hour_assigned,
            "fis1_score": score1,
            "fis2_score": score2,
            "fis1_details": details1,
            "fis2_details": details2,
            "continuity_multiplier": continuity_mult,
            "final_score": final_score,
            "is_assigned": is_assigned,
            "rank": rank,
            "reason": reason
        }
