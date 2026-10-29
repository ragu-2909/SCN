"""
Visualization Module.
Generates academic and interactive visualizations using Matplotlib and Plotly:
- GA Convergence
- Baseline Comparisons
- Schedule Heatmap
- Demand Coverage
- Preferred vs Assigned Hours
- Fairness Distribution
- Membership Functions (Before/After GA)
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .genetic_algorithm import GenerationLog
from .membership_functions import trimf, trapmf, evaluate_mf
from .preprocessing import PreprocessedData


# -------------------------------------------------------------
# Figure A: GA Convergence
# -------------------------------------------------------------
def plot_ga_convergence_plotly(history: List[GenerationLog]) -> go.Figure:
    gens = [h.generation for h in history]
    best_f = [h.best_fitness for h in history]
    mean_f = [h.mean_fitness for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=best_f, mode="lines+markers", name="Best Fitness", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=gens, y=mean_f, mode="lines", name="Mean Fitness", line=dict(color="#ff7f0e", width=2, dash="dash")))

    fig.update_layout(
        title="Genetic Algorithm Convergence Curve (Lower is Better)",
        xaxis_title="Generation",
        yaxis_title="Fitness Score (Sum of RMS Penalties)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig


def plot_ga_convergence_matplotlib(history: List[GenerationLog]) -> plt.Figure:
    gens = [h.generation for h in history]
    best_f = [h.best_fitness for h in history]
    mean_f = [h.mean_fitness for h in history]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(gens, best_f, "o-", color="#1f77b4", label="Best Fitness", linewidth=2)
    ax.plot(gens, mean_f, "--", color="#ff7f0e", label="Mean Fitness", linewidth=1.5)
    ax.set_title("Genetic Algorithm Convergence", fontsize=14, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness Score (RMS Sum)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=11)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------
# Figure B: Baseline Comparison
# -------------------------------------------------------------
def plot_baseline_comparison_plotly(summary_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        summary_df,
        x="Method",
        y="Fitness Mean",
        error_y="Fitness Std" if "Fitness Std" in summary_df.columns else None,
        color="Method",
        title="Comparison of Objective Fitness Across Schedulers (Lower is Better)",
        labels={"Fitness Mean": "Mean Fitness"},
        template="plotly_white"
    )
    return fig


def plot_baseline_comparison_matplotlib(summary_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(summary_df))
    bars = ax.bar(x, summary_df["Fitness Mean"], color=["#2ca02c", "#1f77b4", "#d62728"], alpha=0.85, width=0.5)

    if "Fitness Std" in summary_df.columns:
        ax.errorbar(x, summary_df["Fitness Mean"], yerr=summary_df["Fitness Std"], fmt="none", ecolor="black", capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Method"], fontsize=11, fontweight="medium")
    ax.set_title("Method Comparison: Objective Fitness", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Fitness (Lower is Better)", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    return fig


# -------------------------------------------------------------
# Figure C: Weekly Schedule Heatmap
# -------------------------------------------------------------
def plot_schedule_heatmap_plotly(schedule: np.ndarray, data: PreprocessedData) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=schedule,
        x=data.slot_labels,
        y=data.worker_ids,
        colorscale=[[0, "#f8f9fa"], [1, "#2b5c8f"]],
        showscale=False
    ))
    fig.update_layout(
        title="Assigned Shifts Heatmap Across Week (Dark = Assigned)",
        xaxis_title="Time Slot",
        yaxis_title="Worker ID",
        xaxis=dict(tickangle=-45),
        template="plotly_white",
        height=max(450, len(data.worker_ids) * 22)
    )
    return fig


def plot_schedule_heatmap_matplotlib(schedule: np.ndarray, data: PreprocessedData) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, max(6, len(data.worker_ids) * 0.35)), dpi=300)
    cax = ax.imshow(schedule, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_title("Weekly Roster Heatmap", fontsize=14, fontweight="bold")
    ax.set_yticks(np.arange(len(data.worker_ids)))
    ax.set_yticklabels(data.worker_ids, fontsize=9)

    # Subsample x-ticks to avoid clutter
    step = max(1, len(data.slot_labels) // 10)
    tick_indices = list(range(0, len(data.slot_labels), step))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([data.slot_labels[i] for i in tick_indices], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Time Slot", fontsize=11)
    ax.set_ylabel("Worker", fontsize=11)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------
# Figure D: Staffing Coverage vs Demand
# -------------------------------------------------------------
def plot_coverage_plotly(schedule: np.ndarray, data: PreprocessedData) -> go.Figure:
    assigned_per_slot = np.sum(schedule, axis=0)
    slots = list(range(len(data.demand)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=slots, y=data.demand, mode="lines", name="Required Demand", line=dict(color="#d62728", width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=slots, y=assigned_per_slot, mode="lines+markers", name="Assigned Workers", line=dict(color="#2ca02c", width=2)))

    fig.update_layout(
        title="Staffing Coverage per Slot vs. Required Demand",
        xaxis_title="Slot Index",
        yaxis_title="Worker Headcount",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig


def plot_coverage_matplotlib(schedule: np.ndarray, data: PreprocessedData) -> plt.Figure:
    assigned_per_slot = np.sum(schedule, axis=0)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.step(range(len(data.demand)), data.demand, where="mid", color="#d62728", linestyle="--", linewidth=2, label="Required Demand")
    ax.plot(range(len(assigned_per_slot)), assigned_per_slot, "o-", color="#2ca02c", linewidth=1.5, markersize=4, label="Assigned Workers")
    ax.set_title("Operational Coverage vs Staffing Demand", fontsize=14, fontweight="bold")
    ax.set_xlabel("Slot Index", fontsize=11)
    ax.set_ylabel("Workers on Duty", fontsize=11)
    ax.set_ylim(0, max(np.max(assigned_per_slot) + 2, np.max(data.demand) + 2))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------
# Figure E: Preferred vs Assigned Weekly Hours
# -------------------------------------------------------------
def plot_hours_comparison_plotly(schedule: np.ndarray, data: PreprocessedData) -> go.Figure:
    assigned_hours = np.sum(schedule, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.worker_ids, y=data.pref_weekly_hours, name="Preferred Hours", marker_color="#4575b4"))
    fig.add_trace(go.Bar(x=data.worker_ids, y=assigned_hours, name="Assigned Hours", marker_color="#fdae61"))

    fig.update_layout(
        title="Worker Preferred vs. Assigned Weekly Hours",
        barmode="group",
        xaxis_title="Worker ID",
        yaxis_title="Hours per Week",
        template="plotly_white"
    )
    return fig


def plot_hours_comparison_matplotlib(schedule: np.ndarray, data: PreprocessedData) -> plt.Figure:
    assigned_hours = np.sum(schedule, axis=1)
    x = np.arange(len(data.worker_ids))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.bar(x - width/2, data.pref_weekly_hours, width, label="Preferred Hours", color="#4575b4", alpha=0.9)
    ax.bar(x + width/2, assigned_hours, width, label="Assigned Hours", color="#fdae61", alpha=0.9)
    ax.axhline(25.0, color="#d62728", linestyle=":", label="Weekly Cap (25h)")

    ax.set_title("Weekly Hours: Preferred vs Assigned", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(data.worker_ids, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Worker", fontsize=11)
    ax.set_ylabel("Hours", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------
# Figure H: Membership Functions (Before & After Optimization)
# -------------------------------------------------------------
def plot_output_mfs_comparison_matplotlib(
    initial_mfs: Dict[str, Tuple[float, float, float]],
    optimized_mfs: Optional[Dict[str, Tuple[float, float, float]]] = None
) -> plt.Figure:
    u = np.linspace(0.0, 1.0, 200)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    num_plots = 2 if optimized_mfs else 1
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots), dpi=300)
    if num_plots == 1:
        axes = [axes]

    # Initial
    for i, lbl in enumerate(labels):
        y = trimf(u, initial_mfs[lbl])
        axes[0].plot(u, y, label=lbl, color=colors[i], linewidth=2)
    axes[0].set_title("Initial Output Membership Functions", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Assignment Likelihood", fontsize=10)
    axes[0].set_ylabel("Degree of Membership", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", fontsize=9)

    # Optimized
    if optimized_mfs:
        for i, lbl in enumerate(labels):
            y = trimf(u, optimized_mfs[lbl])
            axes[1].plot(u, y, label=lbl, color=colors[i], linewidth=2)
        axes[1].set_title("GA-Optimized Output Membership Functions", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Assignment Likelihood", fontsize=10)
        axes[1].set_ylabel("Degree of Membership", fontsize=10)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig
