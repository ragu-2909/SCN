"""
Fuzzy Membership Functions Module.
Implements triangular and trapezoidal membership functions,
evaluation methods, and visualization helpers.
"""

from typing import Dict, List, Tuple, Union
import numpy as np


def trimf(x: Union[float, np.ndarray], params: Tuple[float, float, float]) -> Union[float, np.ndarray]:
    """
    Triangular membership function.
    params = (a, b, c) where a <= b <= c.
    """
    a, b, c = params
    x_arr = np.asarray(x, dtype=float)

    # Avoid division by zero when a == b or b == c
    term1 = np.zeros_like(x_arr)
    if b > a:
        cond1 = (x_arr >= a) & (x_arr <= b)
        term1[cond1] = (x_arr[cond1] - a) / (b - a)
    elif a == b:
        term1[x_arr <= b] = 1.0

    term2 = np.zeros_like(x_arr)
    if c > b:
        cond2 = (x_arr > b) & (x_arr <= c)
        term2[cond2] = (c - x_arr[cond2]) / (c - b)
    elif b == c:
        term2[x_arr >= b] = 1.0

    res = np.maximum(0.0, np.minimum(1.0, term1 + term2))
    return float(res) if np.isscalar(x) else res


def trapmf(x: Union[float, np.ndarray], params: Tuple[float, float, float, float]) -> Union[float, np.ndarray]:
    """
    Trapezoidal membership function.
    params = (a, b, c, d) where a <= b <= c <= d.
    """
    a, b, c, d = params
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)

    # Left slope
    if b > a:
        c1 = (x_arr >= a) & (x_arr < b)
        y[c1] = (x_arr[c1] - a) / (b - a)
    else:
        y[x_arr < b] = 1.0

    # Plateau
    c2 = (x_arr >= b) & (x_arr <= c)
    y[c2] = 1.0

    # Right slope
    if d > c:
        c3 = (x_arr > c) & (x_arr <= d)
        y[c3] = (d - x_arr[c3]) / (d - c)
    else:
        y[x_arr > c] = 1.0

    res = np.maximum(0.0, np.minimum(1.0, y))
    return float(res) if np.isscalar(x) else res


# Default input membership functions derived from paper Figure 2
DEFAULT_FIS1_INPUT_MFS = {
    "Very Low": (0.0, 0.0, 4.0, 8.0),      # trapmf
    "Low": (5.0, 10.0, 15.0),               # trimf
    "High": (12.0, 18.0, 25.0),             # trimf
    "Very High": (20.0, 30.0, 60.0, 60.0)   # trapmf
}

DEFAULT_FIS2_INPUT_MFS = {
    "Low": (0.0, 0.0, 2.0, 3.5),           # trapmf
    "Medium": (2.5, 4.5, 6.5),             # trimf
    "High": (5.0, 7.5, 12.0, 12.0)         # trapmf
}

# Initial paper-inspired output membership functions (Figure 6)
DEFAULT_OUTPUT_MFS = {
    "Very Low": (0.0, 0.05, 0.18),
    "Low": (0.10, 0.22, 0.38),
    "Medium": (0.28, 0.45, 0.62),
    "High": (0.50, 0.68, 0.85),
    "Very High": (0.75, 0.90, 1.00)
}


def evaluate_mf(x: Union[float, np.ndarray], params: Union[Tuple[float, ...], List[float]]) -> Union[float, np.ndarray]:
    """Evaluate either triangular (3 params) or trapezoidal (4 params) MF."""
    if len(params) == 3:
        return trimf(x, (params[0], params[1], params[2]))
    elif len(params) == 4:
        return trapmf(x, (params[0], params[1], params[2], params[3]))
    else:
        raise ValueError(f"Invalid MF parameter count: {len(params)}")


def decode_output_mfs_from_genes(genes: List[float]) -> Dict[str, Tuple[float, float, float]]:
    """
    Decode 9 continuous genes in [0, 1] into 5 valid triangular output MFs.
    Guarantees valid ordering: left <= center <= right and proper universe coverage.
    """
    sorted_genes = sorted([float(np.clip(g, 0.01, 0.99)) for g in genes[:9]])
    # 9 points p0 < p1 < p2 < p3 < p4 < p5 < p6 < p7 < p8
    p = sorted_genes

    # 5 triangular MFs across [0, 1]:
    # mf1 (Very Low): center near 0, right extends to p1
    # mf2 (Low): left p0, center p2, right p4
    # mf3 (Medium): left p3, center p4, right p6
    # mf4 (High): left p5, center p6, right p8
    # mf5 (Very High): left p7, center near 1.0
    return {
        "Very Low": (0.0, p[0], p[1]),
        "Low": (p[0], p[2], p[3]),
        "Medium": (p[2], p[4], p[5]),
        "High": (p[4], p[6], p[7]),
        "Very High": (p[6], p[8], 1.0)
    }


def encode_output_mfs_to_genes(mfs: Dict[str, Tuple[float, float, float]]) -> List[float]:
    """Extract 9 genes from 5 triangular MFs."""
    return [
        mfs["Very Low"][1], mfs["Very Low"][2],
        mfs["Low"][1], mfs["Low"][2],
        mfs["Medium"][1], mfs["Medium"][2],
        mfs["High"][1], mfs["High"][2],
        mfs["Very High"][1]
    ]
