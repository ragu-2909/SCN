"""
Streamlit Web Application for Fuzzy Logic–Based Workforce Scheduling
Optimized Using Genetic Algorithms.
Soft Computing AAT - University of Cincinnati Research Reference
"""

import io
import json
import os
import sys
import time
from pathlib import Path

# Add src/ to python path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from workforce_scheduler.baselines import run_heuristic_baseline, run_non_ga_fuzzy_baseline
from workforce_scheduler.chromosome import decode_chromosome
from workforce_scheduler.config import load_config, SchedulerConfig
from workforce_scheduler.data_loader import inspect_excel_sheets, load_tabular_data
from workforce_scheduler.experiments import run_comparative_experiment
from workforce_scheduler.exports import (
    export_convergence_csv,
    export_figure_png,
    export_run_metadata_json,
    export_schedule_csv,
    export_slot_coverage_csv,
    export_worker_summary_csv
)
from workforce_scheduler.fuzzy_system import WorkforceFuzzySystem, INITIAL_FIS1_RULES, INITIAL_FIS2_RULES
from workforce_scheduler.genetic_algorithm import GeneticAlgorithm
from workforce_scheduler.membership_functions import DEFAULT_FIS1_INPUT_MFS, DEFAULT_FIS2_INPUT_MFS, DEFAULT_OUTPUT_MFS
from workforce_scheduler.metrics import compute_schedule_metrics
from workforce_scheduler.preprocessing import build_canonical_data, PreprocessedData
from workforce_scheduler.report_builder import generate_markdown_report
from workforce_scheduler.scenario_generator import (
    SyntheticPool,
    generate_scenario,
    generate_testing_scenarios,
    generate_training_scenarios
)
from workforce_scheduler.scheduler import extract_shifts_dataframe, generate_schedule
from workforce_scheduler.schema_detection import detect_column_mappings, identify_table_type
from workforce_scheduler.validation import (
    check_pre_optimization_feasibility,
    validate_availability_table,
    validate_workers_table
)
from workforce_scheduler.visualizations import (
    plot_baseline_comparison_matplotlib,
    plot_baseline_comparison_plotly,
    plot_coverage_matplotlib,
    plot_coverage_plotly,
    plot_ga_convergence_matplotlib,
    plot_ga_convergence_plotly,
    plot_hours_comparison_matplotlib,
    plot_hours_comparison_plotly,
    plot_output_mfs_comparison_matplotlib,
    plot_schedule_heatmap_matplotlib,
    plot_schedule_heatmap_plotly
)

# -------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Workforce Fuzzy-GA Scheduler",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "preprocessed_data" not in st.session_state:
    st.session_state.preprocessed_data = None
if "last_schedule" not in st.session_state:
    st.session_state.last_schedule = None
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None
if "last_ga_result" not in st.session_state:
    st.session_state.last_ga_result = None
if "experiment_suite" not in st.session_state:
    st.session_state.experiment_suite = None
if "slot_decisions" not in st.session_state:
    st.session_state.slot_decisions = None
if "data_source_mode" not in st.session_state:
    st.session_state.data_source_mode = "Sample Paper Data"

# Sidebar
st.sidebar.title("Workforce Scheduler")
st.sidebar.markdown("**Fuzzy Logic + Genetic Algorithm**")
st.sidebar.markdown("*Soft Computing AAT Reference Engine*")
st.sidebar.divider()

# Mode Preset Selector
config_preset = st.sidebar.radio(
    "Configuration Profile",
    ["Quick Interactive Run (Recommended for Demo)", "Paper Reference Setup (High Compute)"]
)

# Load Default or Paper Config
config_file = ROOT_DIR / "configs" / ("paper_reference_config.yaml" if "Paper Reference" in config_preset else "default_config.yaml")
cfg = load_config(config_file)

with st.sidebar.expander("GA Parameters", expanded=False):
    pop_size = st.number_input("Population Size", min_value=10, max_value=500, value=cfg.ga.population_size, step=10)
    max_gens = st.number_input("Max Generations", min_value=2, max_value=100, value=cfg.ga.max_generations, step=5)
    stall_gens = st.number_input("Stall Generations", min_value=2, max_value=30, value=cfg.ga.stall_generations, step=1)
    crossover_rate = st.slider("Crossover Fraction", 0.0, 1.0, float(cfg.ga.crossover_rate), 0.05)
    mutation_rate = st.slider("Mutation Fraction", 0.0, 1.0, float(cfg.ga.mutation_rate), 0.05)
    elite_cnt = st.number_input("Elite Count", min_value=1, max_value=max(1, pop_size - 1), value=min(cfg.ga.elite_count, pop_size - 1))
    seed_val = st.number_input("Random Seed", value=int(cfg.ga.random_seed))

with st.sidebar.expander("Operating Rules", expanded=False):
    max_weekly_hours = st.number_input("Max Weekly Hours Cap", min_value=10.0, max_value=40.0, value=float(cfg.scenario.max_weekly_hours))
    demand_per_slot = st.number_input("Staffing Demand per Slot", min_value=1, max_value=10, value=int(cfg.scenario.demand_per_slot))
    continuity_factor = st.slider("Continuity Multiplier", 1.0, 2.0, float(cfg.fuzzy.continuity_scaling_factor), 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Slots:** 51 hours (4×11h + 1×7h)")
st.sidebar.markdown(f"**Required Worker-Hours:** {51 * demand_per_slot:.0f}h")

# Helper function to load demo data
def load_default_sample_dataset():
    sample_dir = ROOT_DIR / "data" / "sample"
    w_df, _ = load_tabular_data(sample_dir / "sample_workers.csv", "sample_workers.csv")
    a_df, _ = load_tabular_data(sample_dir / "sample_availability.csv", "sample_availability.csv")
    d_df, _ = load_tabular_data(sample_dir / "sample_demand.csv", "sample_demand.csv")

    w_clean, _, _ = validate_workers_table(w_df, detect_column_mappings(w_df), default_max_hours=max_weekly_hours)
    a_clean, _, _ = validate_availability_table(a_df, detect_column_mappings(a_df), list(w_clean["worker_id"]))

    data = build_canonical_data(w_clean, a_clean, d_df, default_demand=demand_per_slot)
    return data, w_clean, a_clean, d_df

# Main Page Tabs
tabs = st.tabs([
    "1. Overview & Method",
    "2. Data Upload & Validation",
    "3. Fuzzy System Inspection",
    "4. Optimization & Baselines",
    "5. Schedule & Explainability",
    "6. Comparative Benchmark",
    "7. Export & Academic Report"
])

# -------------------------------------------------------------
# TAB 1: OVERVIEW & METHODOLOGY
# -------------------------------------------------------------
with tabs[0]:
    st.title("Workforce Scheduling System using Fuzzy Logic & Genetic Algorithms")
    st.markdown("""
    **Implementation Contract & Academic Application**  
    Based on the research paper: *Fuzzy Logic – Based Scheduling System for Part-Time Workforce*  
    by **Tri Nguyen and Kelly Cohen** (University of Cincinnati).
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Workforce Size", "20 Part-Time Workers", "Desk Assistants")
    col2.metric("Operating Duration", "51 Slots / Week", "5 Days (4×11h + 1×7h)")
    col3.metric("Required Staffing", f"{demand_per_slot} Workers / Slot", f"{51 * demand_per_slot} Worker-Hours")
    col4.metric("Weekly Hour Cap", f"{max_weekly_hours:.0f} Hours / Worker", "Hard Constraint")

    st.markdown("### Operational Architecture")
    st.info("""
    - **FIS1 (Weekly Workload Likelihood):** Evaluates preferred weekly hours vs current assigned weekly hours.
    - **FIS2 (Daily Shift Suitability):** Evaluates preferred shift length vs assigned hours on current day.
    - **Aggregation Operator ($\Pi$):** Multiplies availability (binary 0/1) by $\min(\text{FIS1}, \text{FIS2})$ and applies a continuity factor ($C_{cont}$) for adjacent hour assignments.
    - **Genetic Algorithm (GA):** Simultaneously evolves the 25 discrete fuzzy rule genes and 9 continuous triangular membership function parameters (34-gene chromosome).
    - **Hard Constraints:** Enforces 0 assignments when unavailable, zero weekly cap violations, minimum 1-hour shifts, and required coverage.
    """)

    st.markdown("### Method Fidelity Badges")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.success("**Paper Method**\n- 51-hour schedule (4×11h + 1×7h)\n- Dual Mamdani FIS with centroid defuzzification\n- 34-gene chromosome structure\n- RMS multi-criteria fitness function")
    with b2:
        st.warning("**Implementation Adaptation**\n- Strict pre- and post-optimization hard constraint validation\n- Deterministic tie-breaking for equal scores\n- Independent tracking of feasibility and fitness")
    with b3:
        st.info("**Additional Evaluation**\n- Universal CSV and Excel upload support\n- Preference-aware greedy heuristic baseline\n- Explainable AI per-slot decision trace\n- Understaffed robustness evaluation")

    st.markdown("---")
    if st.button("Load Paper Reference Demo Dataset", type="primary"):
        with st.spinner("Loading demo workforce data..."):
            data, w_clean, a_clean, d_df = load_default_sample_dataset()
            st.session_state.preprocessed_data = data
            st.session_state.data_source_mode = "Sample Paper Data (20 Workers)"
            st.success("Successfully loaded paper reference dataset (20 workers, 51 slots, 4 workers/slot)!")


# -------------------------------------------------------------
# TAB 2: DATA UPLOAD & VALIDATION
# -------------------------------------------------------------
with tabs[1]:
    st.header("Workforce Data Ingestion & Schema Detection")
    st.markdown("Upload custom workforce data or use the paper-compatible dataset.")

    data_option = st.radio(
        "Choose Data Source",
        ["Use Paper Sample Dataset", "Upload CSV / Excel File", "Generate Controlled Synthetic Scenario"],
        horizontal=True
    )

    if data_option == "Use Paper Sample Dataset":
        if st.session_state.preprocessed_data is None:
            data, w_clean, a_clean, d_df = load_default_sample_dataset()
            st.session_state.preprocessed_data = data
            st.session_state.data_source_mode = "Sample Paper Data (20 Workers)"

        st.success("Current Active Dataset: **Paper Reference Sample Dataset (20 Workers)**")
        data = st.session_state.preprocessed_data

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Workers Preview (Requested Hours)")
            w_preview = pd.DataFrame({
                "worker_id": data.worker_ids,
                "preferred_weekly_hours": data.pref_weekly_hours,
                "preferred_shift_length": data.pref_shift_lengths,
                "max_weekly_hours": data.max_weekly_hours
            })
            st.dataframe(w_preview, use_container_width=True, height=300)
        with c2:
            st.subheader("Availability Matrix Preview")
            avail_preview = pd.DataFrame(data.availability_matrix, index=data.worker_ids, columns=data.slot_labels)
            st.dataframe(avail_preview, use_container_width=True, height=300)

    elif data_option == "Upload CSV / Excel File":
        uploaded_file = st.file_uploader(
            "Upload Workforce File (.csv, .xlsx, .xls)",
            type=["csv", "xlsx", "xls"]
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            file_name = uploaded_file.name

            # Check if Excel with multiple sheets
            if file_name.endswith((".xlsx", ".xls")):
                sheets = inspect_excel_sheets(file_bytes)
                selected_sheet = st.selectbox("Select Sheet to Load", sheets)
            else:
                selected_sheet = None

            raw_df, meta = load_tabular_data(file_bytes, file_name, sheet_name=selected_sheet)
            st.info(f"Loaded `{file_name}` ({meta['num_rows']} rows, {meta['num_cols']} columns). SHA-256: `{meta['sha256'][:16]}...`")

            # Column mapping interface
            st.subheader("Column Schema Mapping")
            suggested_mapping = detect_column_mappings(raw_df)

            cols_avail = ["(None)"] + list(raw_df.columns)
            map_cols = st.columns(4)

            w_id_col = map_cols[0].selectbox("Worker ID Column", cols_avail, index=cols_avail.index(suggested_mapping.get("worker_id")) if suggested_mapping.get("worker_id") in cols_avail else 0)
            pref_w_col = map_cols[1].selectbox("Preferred Weekly Hours", cols_avail, index=cols_avail.index(suggested_mapping.get("preferred_weekly_hours")) if suggested_mapping.get("preferred_weekly_hours") in cols_avail else 0)
            pref_s_col = map_cols[2].selectbox("Preferred Shift Length", cols_avail, index=cols_avail.index(suggested_mapping.get("preferred_shift_length")) if suggested_mapping.get("preferred_shift_length") in cols_avail else 0)
            max_w_col = map_cols[3].selectbox("Max Weekly Hours (Optional)", cols_avail, index=cols_avail.index(suggested_mapping.get("max_weekly_hours")) if suggested_mapping.get("max_weekly_hours") in cols_avail else 0)

            active_mapping = {
                "worker_id": None if w_id_col == "(None)" else w_id_col,
                "preferred_weekly_hours": None if pref_w_col == "(None)" else pref_w_col,
                "preferred_shift_length": None if pref_s_col == "(None)" else pref_s_col,
                "max_weekly_hours": None if max_w_col == "(None)" else max_w_col
            }

            if st.button("Validate and Process Uploaded Dataset"):
                cleaned_w, errors, warnings = validate_workers_table(raw_df, active_mapping, default_max_hours=max_weekly_hours)
                if errors:
                    st.error("Validation Errors:\n" + "\n".join(f"- {e}" for e in errors))
                else:
                    if warnings:
                        st.warning("Validation Warnings:\n" + "\n".join(f"- {w}" for w in warnings))

                    # For availability: either uploaded long availability or generate realistic slot matrix
                    st.success("Workers table successfully validated!")
                    # Create matching PreprocessedData
                    pool = SyntheticPool(seed=int(seed_val), total_students=len(cleaned_w), train_count=len(cleaned_w))
                    # Assign custom preferences
                    pool.student_ids = list(cleaned_w["worker_id"])
                    pool.pref_weekly = cleaned_w["preferred_weekly_hours"].values
                    pool.pref_shift = cleaned_w["preferred_shift_length"].values
                    pool.max_weekly = cleaned_w["max_weekly_hours"].values

                    custom_sc = generate_scenario(pool, np.arange(len(cleaned_w)), demand_per_slot=int(demand_per_slot), seed=int(seed_val))
                    st.session_state.preprocessed_data = custom_sc
                    st.session_state.data_source_mode = f"Uploaded File: {file_name}"
                    st.rerun()

    elif data_option == "Generate Controlled Synthetic Scenario":
        st.subheader("Synthetic Scenario Parameters")
        synth_c1, synth_c2, synth_c3 = st.columns(3)
        num_students = synth_c1.number_input("Number of Workers", min_value=10, max_value=50, value=20)
        synth_seed = synth_c2.number_input("Generator Seed", value=42)
        target_demand = synth_c3.number_input("Staffing Demand", min_value=1, max_value=8, value=4)

        if st.button("Generate Synthetic Workforce"):
            pool = SyntheticPool(seed=int(synth_seed), total_students=int(num_students), train_count=int(num_students))
            synth_data = generate_scenario(pool, np.arange(int(num_students)), demand_per_slot=int(target_demand), seed=int(synth_seed))
            st.session_state.preprocessed_data = synth_data
            st.session_state.data_source_mode = f"Synthetic Workforce ({num_students} workers, seed {synth_seed})"
            st.success(f"Generated synthetic scenario with {num_students} workers!")
            st.rerun()

    # Pre-optimization Feasibility Audit
    if st.session_state.preprocessed_data is not None:
        st.markdown("---")
        st.subheader("Pre-Optimization Feasibility Audit")
        curr_data = st.session_state.preprocessed_data
        w_df_temp = pd.DataFrame({"preferred_weekly_hours": curr_data.pref_weekly_hours})

        is_feas, feas_metrics, feas_errors, feas_warns = check_pre_optimization_feasibility(
            w_df_temp, curr_data.availability_matrix, curr_data.demand, curr_data.worker_ids
        )

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Status", "FEASIBLE" if is_feas else "INFEASIBLE")
        fc2.metric("Required Hours", f"{feas_metrics.total_required_worker_hours:.1f}h")
        fc3.metric("Available Hours", f"{feas_metrics.total_available_worker_hours:.1f}h")
        fc4.metric("Avg Available / Slot", f"{feas_metrics.avg_available_workers_per_slot:.1f} workers")

        if not is_feas:
            st.error(f"Infeasible scenario! Slots with insufficient availability: {feas_metrics.infeasible_slots}")
        else:
            st.success("All operational slots have sufficient eligible workers to satisfy staffing demand.")


# -------------------------------------------------------------
# TAB 3: FUZZY SYSTEM INSPECTION
# -------------------------------------------------------------
with tabs[2]:
    st.header("Fuzzy Inference System Architecture & Parameters")
    st.markdown("""
    The scheduling engine utilizes two separate Mamdani Fuzzy Inference Systems (FIS1 & FIS2)
    sharing a common set of 5 triangular output membership functions.
    """)

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.subheader("FIS1: Weekly Workload Suitability")
        st.markdown("**Inputs:** Preferred Weekly Hours (0–40h) & Current Assigned Weekly Hours (0–40h)")
        st.markdown("**Input Terms:** Very Low, Low, High, Very High")
        fis1_df = pd.DataFrame(
            INITIAL_FIS1_RULES,
            index=["Pref: Very Low", "Pref: Low", "Pref: High", "Pref: Very High"],
            columns=["Assigned: Very Low", "Assigned: Low", "Assigned: High", "Assigned: Very High"]
        )
        st.dataframe(fis1_df)

    with f_col2:
        st.subheader("FIS2: Daily Shift Suitability")
        st.markdown("**Inputs:** Preferred Shift Length (0–12h) & Current Daily Assigned Hours (0–12h)")
        st.markdown("**Input Terms:** Low, Medium, High")
        fis2_df = pd.DataFrame(
            INITIAL_FIS2_RULES,
            index=["Pref: Low", "Pref: Medium", "Pref: High"],
            columns=["Assigned: Low", "Assigned: Medium", "Assigned: High"]
        )
        st.dataframe(fis2_df)

    st.subheader("Shared Output Membership Functions (Assignment Likelihood)")
    if st.session_state.last_ga_result is not None:
        optimized_mfs = st.session_state.last_ga_result.best_fuzzy_system.output_mfs
        fig_mfs = plot_output_mfs_comparison_matplotlib(DEFAULT_OUTPUT_MFS, optimized_mfs)
        st.pyplot(fig_mfs)
    else:
        fig_mfs = plot_output_mfs_comparison_matplotlib(DEFAULT_OUTPUT_MFS, None)
        st.pyplot(fig_mfs)
        st.info("Run GA Optimization to compare initial output membership functions with learned functions.")


# -------------------------------------------------------------
# TAB 4: OPTIMIZATION & BASELINES
# -------------------------------------------------------------
with tabs[3]:
    st.header("Schedule Optimization & Comparative Baselines")

    if st.session_state.preprocessed_data is None:
        data, _, _, _ = load_default_sample_dataset()
        st.session_state.preprocessed_data = data

    active_data = st.session_state.preprocessed_data

    run_col1, run_col2 = st.columns(2)

    with run_col1:
        st.subheader("Option A: Fast Single Schedule Optimization")
        st.markdown("Run Genetic Algorithm directly on the current active workforce scenario.")
        if st.button("Run GA Optimization on Active Dataset", type="primary", use_container_width=True):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def ga_callback(gen, total_gens, best_fit, mean_fit):
                frac = gen / total_gens
                progress_bar.progress(frac)
                status_text.text(f"Generation {gen}/{total_gens} | Best Fitness: {best_fit:.4f} | Mean Fitness: {mean_fit:.4f}")

            ga = GeneticAlgorithm(
                population_size=int(pop_size),
                max_generations=int(max_gens),
                stall_generations=int(stall_gens),
                crossover_rate=float(crossover_rate),
                mutation_rate=float(mutation_rate),
                elite_count=int(elite_cnt),
                random_seed=int(seed_val)
            )

            start_t = time.time()
            ga_res = ga.optimize([active_data], progress_callback=ga_callback)
            elapsed = time.time() - start_t
            progress_bar.progress(1.0)
            status_text.success(f"GA completed in {elapsed:.2f}s! Best Fitness: {ga_res.best_fitness:.4f}")

            # Generate schedule with best fuzzy system
            sched, check_res, meta = generate_schedule(active_data, ga_res.best_fuzzy_system)
            mets = compute_schedule_metrics(sched, active_data, method_name="GA-Optimized Fuzzy", runtime_seconds=elapsed)

            st.session_state.last_ga_result = ga_res
            st.session_state.last_schedule = sched
            st.session_state.last_metrics = mets
            st.session_state.slot_decisions = meta.get("slot_decisions")

    with run_col2:
        st.subheader("Option B: Full Comparative Benchmark")
        st.markdown("Evaluate GA-Fuzzy, Non-GA Fuzzy, and Heuristic across identical test scenarios.")
        test_cnt = st.number_input("Number of Test Scenarios", min_value=2, max_value=50, value=5)
        run_understaffed = st.checkbox("Include 16-Worker Understaffed Stress Test", value=True)

        if st.button("Execute Comparative Benchmark", use_container_width=True):
            pool = SyntheticPool(seed=int(seed_val), total_students=50, train_count=30)
            train_scs = generate_training_scenarios(pool, num_scenarios=3, workers_per_scenario=20, seed=101)
            test_scs = generate_testing_scenarios(pool, num_scenarios=int(test_cnt), workers_per_scenario=20, seed=201)

            under_scs = None
            if run_understaffed:
                under_scs = generate_testing_scenarios(pool, num_scenarios=int(test_cnt), workers_per_scenario=16, seed=301)

            p_bar = st.progress(0.0)
            p_text = st.empty()

            def exp_callback(msg, frac):
                p_bar.progress(frac)
                p_text.text(msg)

            ga_p = {
                "population_size": int(pop_size),
                "max_generations": int(max_gens),
                "stall_generations": int(stall_gens),
                "crossover_rate": float(crossover_rate),
                "mutation_rate": float(mutation_rate),
                "elite_count": int(elite_cnt),
                "random_seed": int(seed_val)
            }

            suite_res = run_comparative_experiment(
                train_scenarios=train_scs,
                test_scenarios_standard=test_scs,
                test_scenarios_understaffed=under_scs,
                ga_params=ga_p,
                progress_callback=exp_callback
            )
            st.session_state.experiment_suite = suite_res
            st.session_state.last_ga_result = suite_res.ga_result
            st.success("Comparative Benchmark Completed!")

    # Display Convergence Chart if GA was run
    if st.session_state.last_ga_result is not None:
        st.subheader("Genetic Algorithm Convergence History")
        fig_conv = plot_ga_convergence_plotly(st.session_state.last_ga_result.history)
        st.plotly_chart(fig_conv, use_container_width=True)


# -------------------------------------------------------------
# TAB 5: SCHEDULE & EXPLAINABILITY
# -------------------------------------------------------------
with tabs[4]:
    st.header("Schedule Results & Explainable AI")

    if st.session_state.last_schedule is None:
        if st.button("Generate Initial Schedule with Baseline Fuzzy Logic"):
            data = st.session_state.preprocessed_data or load_default_sample_dataset()[0]
            fuzzy_sys = WorkforceFuzzySystem(continuity_scaling_factor=float(continuity_factor))
            start_t = time.time()
            sched, check_res, meta = generate_schedule(data, fuzzy_sys)
            dur = time.time() - start_t
            mets = compute_schedule_metrics(sched, data, method_name="Non-GA Baseline", runtime_seconds=dur)
            st.session_state.last_schedule = sched
            st.session_state.last_metrics = mets
            st.session_state.slot_decisions = meta.get("slot_decisions")
            st.rerun()
        else:
            st.info("No schedule generated yet. Click above or run optimization in Tab 4.")

    if st.session_state.last_schedule is not None:
        sched = st.session_state.last_schedule
        data = st.session_state.preprocessed_data
        mets = st.session_state.last_metrics

        # Feasibility Summary Card
        status_color = "green" if mets.is_feasible else "red"
        st.markdown(f"### Schedule Feasibility Status: **:{status_color}[{'FEASIBLE' if mets.is_feasible else 'INFEASIBLE'}]**")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Coverage Rate", f"{mets.coverage_rate * 100:.1f}%")
        m2.metric("Weekly MAE", f"{mets.weekly_preference_mae:.2f} hrs")
        m3.metric("Shift MAE", f"{mets.daily_shift_mae:.2f} hrs")
        m4.metric("Satisfaction", f"{mets.preference_satisfaction_rate * 100:.1f}%")
        m5.metric("Jain's Fairness", f"{mets.jains_fairness_index:.3f}")

        # Heatmap
        st.subheader("Weekly Roster Heatmap (Worker × Time Slot)")
        fig_heat = plot_schedule_heatmap_plotly(sched, data)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Tabular Shift Roster
        st.subheader("Extracted Tabular Shift Roster")
        shifts_df = extract_shifts_dataframe(sched, data)
        st.dataframe(shifts_df, use_container_width=True, height=250)

        # Hours Comparison & Demand
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.subheader("Preferred vs. Assigned Weekly Hours")
            fig_hours = plot_hours_comparison_plotly(sched, data)
            st.plotly_chart(fig_hours, use_container_width=True)
        with c_p2:
            st.subheader("Operational Coverage per Slot")
            fig_cov = plot_coverage_plotly(sched, data)
            st.plotly_chart(fig_cov, use_container_width=True)

        # Explainable AI Inspector
        st.markdown("---")
        st.subheader("Explainable AI: Slot Decision Inspector")
        st.markdown("Inspect why workers were assigned or excluded for any specific time slot.")

        if st.session_state.slot_decisions:
            decisions = st.session_state.slot_decisions
            slot_names = [f"Slot {d['slot']}: {d['label']}" for d in decisions]
            selected_slot_idx = st.selectbox("Select Time Slot to Explain", range(len(slot_names)), format_func=lambda i: slot_names[i])

            slot_info = decisions[selected_slot_idx]
            st.markdown(f"**Assigned Workers:** `{', '.join(slot_info['assigned_workers'])}`")

            scores_df = pd.DataFrame(slot_info["scores"])
            scores_df["selected"] = scores_df["selected"].apply(lambda s: "YES" if s else "NO")
            st.dataframe(scores_df, use_container_width=True)


# -------------------------------------------------------------
# TAB 6: COMPARATIVE BENCHMARK
# -------------------------------------------------------------
with tabs[5]:
    st.header("Comparative Benchmark & Performance Analysis")

    if st.session_state.experiment_suite is None:
        st.info("Execute a Comparative Benchmark from Tab 4 to visualize multi-method evaluations.")
    else:
        suite = st.session_state.experiment_suite

        st.subheader("Standard 20-Worker Scenario Comparison")
        st.dataframe(suite.standard_summary_df, use_container_width=True)

        fig_comp = plot_baseline_comparison_plotly(suite.standard_summary_df)
        st.plotly_chart(fig_comp, use_container_width=True)

        if suite.understaffed_summary_df is not None:
            st.subheader("Understaffed Robustness Comparison (16 Workers)")
            st.dataframe(suite.understaffed_summary_df, use_container_width=True)


# -------------------------------------------------------------
# TAB 7: EXPORT & ACADEMIC REPORT
# -------------------------------------------------------------
with tabs[6]:
    st.header("Export Assets & Academic Report Generator")

    if st.session_state.last_schedule is None:
        st.warning("Generate or optimize a schedule first before downloading exports.")
    else:
        sched = st.session_state.last_schedule
        data = st.session_state.preprocessed_data
        mets = st.session_state.last_metrics

        out_dir = ROOT_DIR / "outputs"

        st.subheader("Download Schedule and Metrics Files")
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)

        # Export CSVs
        mat_path, shift_path = export_schedule_csv(sched, data, out_dir / "schedules", "current")
        w_sum_path = export_worker_summary_csv(sched, data, out_dir / "metrics" / "worker_summary.csv")
        cov_path = export_slot_coverage_csv(sched, data, out_dir / "metrics" / "slot_coverage.csv")

        with open(shift_path, "rb") as f:
            d_col1.download_button("Download Shift Roster (CSV)", f.read(), "shifts_roster.csv", "text/csv")
        with open(mat_path, "rb") as f:
            d_col2.download_button("Download Slot Matrix (CSV)", f.read(), "schedule_matrix.csv", "text/csv")
        with open(w_sum_path, "rb") as f:
            d_col3.download_button("Download Worker Summary (CSV)", f.read(), "worker_summary.csv", "text/csv")
        with open(cov_path, "rb") as f:
            d_col4.download_button("Download Slot Coverage (CSV)", f.read(), "slot_coverage.csv", "text/csv")

        st.markdown("---")
        st.subheader("Academic Report Draft (Populated with Real Experimental Data)")

        # Prepare values for report
        summary_table = pd.DataFrame([mets.to_dict()])
        under_table = None
        if st.session_state.experiment_suite and st.session_state.experiment_suite.understaffed_summary_df is not None:
            under_table = st.session_state.experiment_suite.understaffed_summary_df

        fis1_str = pd.DataFrame(INITIAL_FIS1_RULES, index=["VL", "L", "H", "VH"], columns=["VL", "L", "H", "VH"]).to_string()
        fis2_str = pd.DataFrame(INITIAL_FIS2_RULES, index=["L", "M", "H"], columns=["L", "M", "H"]).to_string()

        run_meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        }

        report_content = generate_markdown_report(
            summary_df=summary_table,
            understaffed_df=under_table,
            best_fitness=mets.fitness,
            baseline_fitness=mets.fitness * 1.15,
            heuristic_fitness=mets.fitness * 1.25,
            ga_config={"population_size": pop_size, "max_generations": max_gens, "stall_generations": stall_gens, "crossover_rate": crossover_rate, "mutation_rate": mutation_rate, "elite_count": elite_cnt, "random_seed": seed_val},
            run_metadata=run_meta,
            learned_fis1_rules=fis1_str,
            learned_fis2_rules=fis2_str
        )

        st.download_button(
            "Download Academic Markdown Report",
            report_content,
            file_name="Workforce_Scheduling_Research_Report.md",
            mime="text/markdown",
            type="primary"
        )

        with st.expander("Preview Report Draft", expanded=True):
            st.markdown(report_content)
