# Fuzzy Logic–Based Workforce Scheduling System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-28%20passed-brightgreen.svg)](https://pytest.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io/)

A complete, locally runnable Streamlit application and research testbench that implements and evaluates a fuzzy-logic workforce scheduler optimized with a Genetic Algorithm (GA), based on the published research paper:
> **Tri Nguyen and Kelly Cohen**, *"Fuzzy Logic – Based Scheduling System for Part-Time Workforce"*, Department of Aerospace Engineering, University of Cincinnati.

---

## 1. System Overview

Part-time student workforce scheduling involves reconciling irregular student availability and individual working-hour preferences with hard operational coverage constraints.

This system provides:
1. **Two Mamdani Fuzzy Inference Systems:**
   - **FIS1 (Weekly Workload Likelihood):** Evaluates preferred weekly hours against current assigned weekly hours.
   - **FIS2 (Daily Shift Suitability):** Evaluates preferred shift length against hours already assigned on the current day.
2. **Multiplicative Aggregation Operator ($\Pi$):**
   - Incorporates worker availability (binary 0/1) to ensure zero score when unavailable.
   - Computes $\min(\text{FIS1}, \text{FIS2})$.
   - Applies a continuity scaling factor ($C_{cont}$) when assigned to the preceding adjacent slot on the same day.
3. **Genetic Algorithm Optimization:**
   - 34-gene hybrid chromosome: 16 discrete rule genes for FIS1, 9 discrete rule genes for FIS2, and 9 continuous membership function shape parameters.
   - Roulette Wheel Selection, Elitism (10 elites), Crossover (0.8), Mutation (0.2), and 10-generation stall stopping limit.
4. **Hard Constraint Enforcement:**
   - $100\%$ zero-tolerance checks for availability, weekly caps (25h default), minimum shift lengths (1h), and required coverage headcount.
5. **Ablation & Benchmarking:**
   - Comparative testbench against Non-GA Fixed-Rule Fuzzy Baseline and Preference-Aware Heuristic Baseline across identical scenario seeds.
6. **Universal Data Ingestion:**
   - CSV and multi-sheet Excel (`.xlsx`, `.xls`) upload with automatic column alias detection, preview, and validation.
7. **Explainable AI (XAI):**
   - Interactive decision inspector revealing exact rule activations, fuzzy scores, and ranking rationale for every slot.

---

## 2. Windows Local Execution Instructions

These instructions are tested and verified for Windows PowerShell.

### Step 1: Open PowerShell
Open PowerShell in the project directory:
```powershell
cd "c:\Users\Ragu\Documents\SCN aat"
```

### Step 2: Create Virtual Environment
Using Python 3.11:
```powershell
py -3.11 -m venv .venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*Note on PowerShell Execution Policy:* If script execution is restricted on your system, enable execution for the current process:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run the Pytest Test Suite
```powershell
python -m pytest -v
```
All 28 unit and integration tests will execute and pass in ~5 seconds.

### Step 6: Launch the Streamlit Web Application
```powershell
python -m streamlit run app.py
```
Open your browser to:
```
http://localhost:8501
```

---

## 3. Project Structure

```text
SCN aat/
├── app.py                         # Streamlit multi-tab interactive application
├── README.md                      # Complete system documentation and run guide
├── IMPLEMENTATION.md              # Research specification contract
├── requirements.txt               # Pinned dependencies for Python 3.11
├── pyproject.toml                 # Package definition & pytest config
├── .gitignore                     # Git exclusion rules
├── configs/
│   ├── default_config.yaml        # Fast interactive parameters
│   └── paper_reference_config.yaml # Exact paper configuration (200 pop, 50 gen)
├── data/
│   ├── sample/
│   │   ├── sample_workers.csv     # 20 part-time student profiles (Nguyen & Cohen)
│   │   ├── sample_availability.csv # 1,020 binary slot records (20 workers × 51 slots)
│   │   ├── sample_demand.csv      # Operational demand (4 workers / slot)
│   │   └── sample_workforce.xlsx  # Multi-sheet Excel workbook
│   ├── uploads/                   # Local staging directory for user uploads
│   └── processed/                 # Cached canonical data
├── src/
│   └── workforce_scheduler/
│       ├── __init__.py
│       ├── config.py              # Configuration loading and schema dataclasses
│       ├── data_loader.py         # CSV & multi-sheet Excel loader with SHA-256
│       ├── schema_detection.py    # Auto-detection of column aliases
│       ├── validation.py          # Pre-optimization feasibility checking
│       ├── preprocessing.py       # Canonical data structures & slot mappings
│       ├── scenario_generator.py  # 70-student pool & 20/16-worker scenario sampler
│       ├── membership_functions.py# Triangular & trapezoidal MFs with gene decoding
│       ├── fuzzy_system.py        # Vectorized Mamdani FIS, aggregation & XAI trace
│       ├── chromosome.py          # 34-gene encoding, repair, validation
│       ├── fitness.py             # Multi-scenario RMS objective (Eq. 1)
│       ├── genetic_algorithm.py   # GA optimizer with roulette wheel & elitism
│       ├── scheduler.py           # Sequential slot scheduler & shift builder
│       ├── constraints.py         # Hard constraint auditor
│       ├── baselines.py           # Non-GA Fuzzy & Preference Heuristic
│       ├── metrics.py             # Preference MAE, satisfaction, Jain index
│       ├── experiments.py         # Multi-method testbench & understaffed analysis
│       ├── visualizations.py      # Plotly & Matplotlib heatmaps, convergence, MFs
│       ├── exports.py             # CSV, JSON, and high-res PNG export
│       └── report_builder.py      # Academic research report generator
├── tests/                         # Full pytest suite (28 tests)
│   ├── conftest.py
│   ├── test_chromosome.py
│   ├── test_constraints.py
│   ├── test_data_loader.py
│   ├── test_experiments.py
│   ├── test_exports.py
│   ├── test_fitness.py
│   ├── test_fuzzy_system.py
│   ├── test_metrics.py
│   ├── test_scheduler.py
│   └── test_validation.py
└── outputs/
    ├── schedules/                 # Exported slot matrices and shift rosters
    ├── metrics/                   # Exported worker summaries and coverage tables
    ├── figures/                   # High-res academic figures (PNG)
    └── reports/                   # Generated academic markdown reports
```

---

## 4. Research Paper Fidelity and Explicit Adaptations

| Category | Reference Paper (Nguyen & Cohen) | This Implementation | Classification |
| :--- | :--- | :--- | :--- |
| **Operating Week** | 5 days: Day 1–4: 11h, Day 5: 7h (51 operating hours total) | Exact 51-slot discrete representation | **Paper Method** |
| **Demand** | 4 workers per hour (204 worker-hours) | 4 workers per slot (or custom user demand) | **Paper Method** |
| **Workforce Size** | ~20 student workers (or 16 for understaffed test) | 20 workers standard, 16 workers understaffed | **Paper Method** |
| **Hour Limits** | 25 hours weekly maximum | 25 hours weekly maximum hard constraint | **Paper Method** |
| **Preferences** | Uniform(5, 15) weekly, Uniform(3, 8) shift | Uniform(5, 15) weekly, Uniform(3, 8) shift | **Paper Method** |
| **Fuzzy Inference** | Mamdani FIS1 & FIS2 with centroid defuzzification | Pure NumPy vectorized Mamdani inference | **Paper Method** |
| **Aggregation ($\Pi$)**| $\text{Avail} \times \min(\text{FIS1}, \text{FIS2}) \times \text{Continuity}$ | Exact multiplicative formulation with $C_{cont}=1.30$ | **Paper Method** |
| **Chromosome** | 34 genes (16 FIS1 rules + 9 FIS2 rules + 9 MF parameters) | Exact 34-gene hybrid chromosome structure | **Paper Method** |
| **Fitness Objective**| $\frac{1}{n}\sum [\text{RMS}(\Delta_1) + \text{RMS}(\Delta_2) + \text{RMS}(\Delta_3)]$ | Exact mathematical formulation (Eq. 1) | **Paper Method** |
| **Hard Constraints** | Described in prose; cap penalty in fitness | Independent pre- and post-optimization validation | **Adaptation** |
| **Baselines** | GA vs Non-GA ablation | Non-GA Fuzzy + Greedy Heuristic + Random | **Additional Evaluation** |
| **Data Ingestion** | Fixed unreleased student dataset | CSV & Excel upload with automatic schema mapping | **Additional Feature** |
| **Explainable AI** | Discussion of rule transparency | Interactive per-slot decision breakdown in UI | **Additional Feature** |

---

## 5. Troubleshooting Guide

- **PowerShell Execution Policy Error:**
  If you see `File Activate.ps1 cannot be loaded because running scripts is disabled`, run:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
- **Port Conflict on 8501:**
  If port 8501 is occupied, launch Streamlit on an alternate port:
  ```powershell
  python -m streamlit run app.py --server.port 8502
  ```
- **Missing Excel Support:**
  Ensure `openpyxl` is installed (`pip install openpyxl`). It is already pinned in `requirements.txt`.
- **Understaffed Infeasible Warning:**
  If custom data contains slots where available workers < demand, the app will flag the scenario as infeasible before optimization rather than producing invalid assignments.
#   S C N  
 