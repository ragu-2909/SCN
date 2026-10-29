# Fuzzy Logic–Based Workforce Scheduling System Optimized by Genetic Algorithm
## Complete End-to-End Technical Specification, Problem Formulation, Implementation Architecture, and Experimental Results

**Course / Subject:** Soft Computing Application Assignment (SCN AAT)  
**Academic Reference:** Tri Nguyen and Kelly Cohen, *"Fuzzy Logic – Based Scheduling System for Part-Time Workforce"*, Department of Aerospace Engineering, University of Cincinnati  
**Implementation Platform:** Python 3.11, Streamlit, NumPy, Pandas, Matplotlib, Plotly, Pytest  
**Repository Working Directory:** `c:\Users\Ragu\Documents\SCN aat`  
**Document Classification:** Master Comprehensive Guide (All-in-One Documentation)

---

## Table of Contents
1. [Executive Summary & High-Level Introduction](#1-executive-summary--high-level-introduction)
2. [The Scheduling Problem & Mathematical Formulation](#2-the-scheduling-problem--mathematical-formulation)
   - [2.1 Real-World Context](#21-real-world-context)
   - [2.2 Operating Environment & Time Horizon](#22-operating-environment--time-horizon)
   - [2.3 Hard Constraints (Zero-Tolerance Operational Rules)](#23-hard-constraints-zero-tolerance-operational-rules)
   - [2.4 Soft Objectives (Worker Preferences & Quality of Life)](#24-soft-objectives-worker-preferences--quality-of-life)
   - [2.5 Why Traditional Scheduling Approaches Struggle](#25-why-traditional-scheduling-approaches-struggle)
3. [The Soft Computing Solution Architecture](#3-the-soft-computing-solution-architecture)
   - [3.1 System Flow & Component Diagram](#31-system-flow--component-diagram)
   - [3.2 The Fuzzy Inference Engine (FIS1 & FIS2)](#32-the-fuzzy-inference-engine-fis1--fis2)
   - [3.3 Membership Functions & Discretization](#33-membership-functions--discretization)
   - [3.4 Fuzzy Rule Bases (Expert Knowledge)](#34-fuzzy-rule-bases-expert-knowledge)
   - [3.5 Mamdani Implication & Centroid Defuzzification](#35-mamdani-implication--centroid-defuzzification)
   - [3.6 The Multiplicative Aggregation Operator ($\Pi$) & Continuity Multiplier](#36-the-multiplicative-aggregation-operator-pi--continuity-multiplier)
4. [The Genetic Algorithm (GA) Optimization Engine](#4-the-genetic-algorithm-ga-optimization-engine)
   - [4.1 Why Genetic Fuzzy Systems (GFS)?](#41-why-genetic-fuzzy-systems-gfs)
   - [4.2 The 34-Gene Chromosome Encoding](#42-the-34-gene-chromosome-encoding)
   - [4.3 Multi-Scenario RMS Fitness Function](#43-multi-scenario-rms-fitness-function)
   - [4.4 Genetic Operators: Selection, Crossover, Mutation, and Elitism](#44-genetic-operators-selection-crossover-mutation-and-elitism)
   - [4.5 Convergence Monitoring & Stopping Criteria](#45-convergence-monitoring--stopping-criteria)
5. [Schedule Generation & Hard Constraint Enforcement](#5-schedule-generation--hard-constraint-enforcement)
   - [5.1 Chronological Sequential Slot Assignment Algorithm](#51-chronological-sequential-slot-assignment-algorithm)
   - [5.2 Pre-Optimization Feasibility Audit](#52-pre-optimization-feasibility-audit)
   - [5.3 Independent Post-Scheduling Constraint Verification](#53-independent-post-scheduling-constraint-verification)
   - [5.4 Continuous Shift Extraction](#54-continuous-shift-extraction)
6. [Comparative Baselines](#6-comparative-baselines)
   - [6.1 Baseline A: Non-GA Fixed-Rule Fuzzy Scheduler (Ablation)](#61-baseline-a-non-ga-fixed-rule-fuzzy-scheduler-ablation)
   - [6.2 Baseline B: Preference-Aware Greedy Heuristic](#62-baseline-b-preference-aware-greedy-heuristic)
   - [6.3 Baseline C: Random Eligible Assignment (Stress Test)](#63-baseline-c-random-eligible-assignment-stress-test)
7. [Comprehensive Measured Results & Findings](#7-comprehensive-measured-results--findings)
   - [7.1 Numerical Comparison Table (Actual Measured Values)](#71-numerical-comparison-table-actual-measured-values)
   - [7.2 Convergence Trajectory Analysis](#72-convergence-trajectory-analysis)
   - [7.3 Preference Satisfaction & Workload Distribution](#73-preference-satisfaction--workload-distribution)
   - [7.4 Fairness Analysis & Jain's Fairness Index](#74-fairness-analysis--jains-fairness-index)
   - [7.5 Understaffed Robustness Stress Test (16 Workers)](#75-understaffed-robustness-stress-test-16-workers)
8. [Explainable AI (XAI) Slot Decision Inspector](#8-explainable-ai-xai-slot-decision-inspector)
9. [Codebase Architecture & File-by-File Breakdown](#9-codebase-architecture--file-by-file-breakdown)
10. [Step-by-Step User & Developer Execution Guide](#10-step-by-step-user--developer-execution-guide)
11. [Paper Fidelity vs. Implementation Adaptations](#11-paper-fidelity-vs-implementation-adaptations)
12. [Limitations, Discussion & Conclusion](#12-limitations-discussion--conclusion)

---

## 1. Executive Summary & High-Level Introduction

Scheduling part-time employees is one of the most critical operational challenges in service facilities, retail, hospitality, and university campus departments. Unlike full-time employees who work predictable 40-hour blocks, part-time student employees have:
1. Highly irregular academic schedules that differ day-by-day.
2. Individual preferences for how many hours they want to work in a week (e.g., 5 to 15 hours).
3. Individual preferences for shift lengths (e.g., 3 to 8 hours).
4. Strict institutional policies (e.g., students cannot exceed 25 hours per week).
5. Facilities that must maintain constant operational coverage (e.g., at least 4 front-desk assistants on duty at all times).

This project implements a complete **Genetic Fuzzy System (GFS)** based on the published research paper by **Tri Nguyen and Kelly Cohen** (*University of Cincinnati*). The system combines:
- **Fuzzy Logic (FL):** Provides human-interpretable reasoning, mapping continuous hour allocations into linguistic terms ("Low", "Medium", "High") and calculating transparent shift likelihoods.
- **Genetic Algorithms (GA):** Automatically optimizes the fuzzy rule tables and output membership function coordinates to maximize preference satisfaction and eliminate shift fragmentation.
- **Strict Hard Constraint Enforcement:** Validates that no student is assigned when unavailable, no student exceeds 25 hours, and every required slot has sufficient staffing.
- **Streamlit Web Interface:** Provides an interactive dashboard for dataset uploads, column schema detection, algorithm tuning, schedule visualization, explainable AI audits, and automated research report generation.

---

## 2. The Scheduling Problem & Mathematical Formulation

### 2.1 Real-World Context
The specific problem modeled is based on university front-desk assistant scheduling at the University of Cincinnati. Approximately 20 student workers are employed each semester. All workers perform homogeneous tasks (no hierarchical seniority or role differences). Each student submits:
- A weekly availability calendar indicating which 1-hour slots they can work.
- Their requested total weekly hours ($5 \le H_{req} \le 15$).
- Their preferred continuous shift length ($3 \le L_{req} \le 8$).

### 2.2 Operating Environment & Time Horizon
- **Operating Week:** 5 days (Monday through Friday).
- **Daily Operating Durations:**
  - Days 1 through 4 (Mon–Thu): 11 operating hours per day (e.g., 08:30 AM to 07:30 PM).
  - Day 5 (Fri): 7 operating hours (e.g., 08:30 AM to 03:30 PM).
  - **Total Discrete Time Slots:** $11 \times 4 + 7 = 51\text{ one-hour slots}$.
- **Staffing Coverage Requirement:** At least 4 employees on duty at all operating times.
- **Total Required Worker-Hours:** $51\text{ slots} \times 4\text{ workers} = 204\text{ worker-hours per week}$.

### 2.3 Hard Constraints (Zero-Tolerance Operational Rules)
Let $W = \{1, 2, \dots, N\}$ be the set of workers ($N = 20$) and $T = \{0, 1, \dots, 50\}$ be the ordered time slots.  
Let the binary decision variable be:
$$x_{w,t} = \begin{cases} 1 & \text{if worker } w \text{ is assigned to time slot } t \\ 0 & \text{otherwise} \end{cases}$$

The schedule is valid **if and only if** all five hard constraints are strictly satisfied:

1. **Worker Availability Constraint:**
   $$x_{w,t} \le A_{w,t}, \quad \forall w \in W, \; \forall t \in T$$
   where $A_{w,t} \in \{0, 1\}$ represents worker $w$'s availability for slot $t$. If a worker is unavailable ($A_{w,t} = 0$), assignment is impossible ($x_{w,t} = 0$).

2. **Operational Coverage Constraint:**
   $$\sum_{w=1}^N x_{w,t} \ge D_t, \quad \forall t \in T$$
   where $D_t = 4$ for all $t$. Every slot must have at least 4 workers on duty.

3. **Weekly Working-Hour Maximum Constraint:**
   $$\sum_{t=0}^{50} x_{w,t} \cdot \Delta t \le H_w^{\max}, \quad \forall w \in W$$
   where $\Delta t = 1.0\text{ hour}$ and $H_w^{\max} = 25.0\text{ hours}$ (regulated institutional cap).

4. **Minimum Shift Duration Constraint:**
   Each contiguous sequence of assigned slots on any given day must be at least 1 hour long ($\ge 1\text{ slot}$) to avoid micro-shifts.

5. **Single Assignment per Slot Constraint:**
   $$x_{w,t} \in \{0, 1\}$$
   A worker cannot be assigned more than once to the same slot.

### 2.4 Soft Objectives (Worker Preferences & Quality of Life)
Subject to hard constraints, the system optimizes four soft quality objectives:
1. **Weekly Hours Preference:** Assigned weekly hours $\sum_t x_{w,t}$ should be as close as possible to the worker's preferred weekly hours $H_{pref}(w)$.
2. **Daily Shift Duration Preference:** When assigned on day $d$, the length of the shift should match the worker's preferred shift duration $L_{pref}(w)$.
3. **Shift Continuity:** Workers should not be assigned fragmented, one-hour-on, one-hour-off schedules. Contiguous hours are prioritized.
4. **Fairness:** Over-assignment and under-assignment should be balanced across available workers.

### 2.5 Why Traditional Scheduling Approaches Struggle
- **Integer Linear Programming (ILP):** Formulating human soft preferences, non-linear continuity incentives, and flexible linguistic rules in ILP requires complex auxiliary variables and large constraint matrices, leading to exponential solve times when problem scales.
- **Pure Metaheuristics (e.g., Pure GA without Fuzzy Logic):** Direct binary chromosome encoding for $(20 \times 51 = 1,020\text{ binary variables})$ creates a search space of $2^{1020} \approx 10^{307}$. Random crossover and mutation routinely produce invalid schedules violating availability and coverage constraints, requiring complex repair operators.
- **The Genetic Fuzzy Advantage:** Instead of searching the raw schedule space, the GA searches the compact parameter space of the **Fuzzy Rule Base and Output Membership Functions** (only 34 genes). The deterministic fuzzy scheduler then builds valid rosters sequentially, guaranteeing fast runtime and explainable decision logic.

---

## 3. The Soft Computing Solution Architecture

### 3.1 System Flow & Component Diagram

```
+-----------------------------------------------------------------------------------+
|                                  INPUT DATA                                       |
|  - 20 Worker Profiles (Requested Hours [5-15h], Desired Shift Length [3-8h])      |
|  - Binary Availability Matrix (20 workers x 51 slots)                              |
|  - Operational Demand (4 workers required per slot)                               |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       PRE-OPTIMIZATION FEASIBILITY AUDIT                          |
|  - Total Available Worker-Hours vs Total Required Worker-Hours (204h)             |
|  - Per-Slot Availability Check: Ensure >= 4 workers available in EVERY slot       |
|  - If any slot < 4 available: Flag scenario as INFEASIBLE before optimization     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|               SEQUENTIAL SLOT SCHEDULING ENGINE (Slot 0 -> Slot 50)               |
|                                                                                   |
|   For each time slot t:                                                           |
|     1. Identify Eligible Workers (Available AND current_weekly_hours < 25h)      |
|                                                                                   |
|     2. For each eligible worker:                                                  |
|        +-----------------------------------------------------------------------+  |
|        | FIS1: Weekly Workload Suitability                                     |  |
|        |   Inputs: (Preferred Weekly Hours, Current Assigned Weekly Hours)     |  |
|        |   Output: Score1 in [0, 1]                                            |  |
|        +-----------------------------------------------------------------------+  |
|        | FIS2: Daily Shift Suitability                                         |  |
|        |   Inputs: (Preferred Shift Length, Current Assigned Daily Hours)      |  |
|        |   Output: Score2 in [0, 1]                                            |  |
|        +-----------------------------------------------------------------------+  |
|        | Multiplicative Aggregation (Pi):                                      |  |
|        |   Score = Availability * min(Score1, Score2) * Continuity_Multiplier  |  |
|        |   (Continuity Multiplier = 1.30 if assigned in previous hour, else 1) |  |
|        +-----------------------------------------------------------------------+  |
|                                                                                   |
|     3. Rank eligible workers by final Score descending.                           |
|     4. Select Top 4 candidates and assign them to slot t.                          |
|     5. Increment their weekly and daily assigned hours.                           |
|     6. When moving to a new day, reset daily assigned hours to 0.                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       GENETIC ALGORITHM OPTIMIZATION LOOP                         |
|  - Population = 50-200 Chromosomes (each chromosome = 34 genes)                   |
|  - Evaluates schedules generated across n training scenarios                      |
|  - Objective: Minimize Multi-Scenario RMS Fitness Function (Eq. 1)                |
|  - Roulette Wheel Selection, Elitism (top 10 preserved), Crossover, Mutation      |
|  - Evolves Rule Tables and Output MF Parameters over generations                  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                     POST-OPTIMIZATION VALIDATION & EXPORT                         |
|  - Exhaustive Hard-Constraint Audit: 0 unavail, 0 cap violations, 100% coverage   |
|  - Export: Schedule Matrix CSV, Shift Roster CSV, High-Res Figures, Report MD     |
+-----------------------------------------------------------------------------------+
```

---

### 3.2 The Fuzzy Inference Engine (FIS1 & FIS2)
The decision to assign a worker to a slot is modeled using two Mamdani Fuzzy Inference Systems:

#### FIS1: Weekly Workload Suitability
- **Purpose:** Decides whether a worker needs more weekly hours.
- **Input 1:** `preferred_weekly_hours` (Universe: $[0, 40]$ hours).
- **Input 2:** `assigned_weekly_hours` (Universe: $[0, 40]$ hours).
- **Linguistic Levels (4 terms):** `Very Low`, `Low`, `High`, `Very High`.
- **Output:** Intermediate likelihood of assignment $\in [0, 1]$.

#### FIS2: Daily Shift Suitability
- **Purpose:** Decides whether a worker should continue working on the current day based on their shift length preference.
- **Input 1:** `preferred_shift_length` (Universe: $[0, 12]$ hours).
- **Input 2:** `assigned_daily_hours` (Universe: $[0, 12]$ hours).
- **Linguistic Levels (3 terms):** `Low`, `Medium`, `High`.
- **Output:** Intermediate likelihood of assignment $\in [0, 1]$.

---

### 3.3 Membership Functions & Discretization

#### Triangular Membership Function ($\text{trimf}$)
Defined by three scalar parameters $(a, b, c)$ where $a \le b \le c$:
$$\mu_{\text{tri}}(x; a, b, c) = \max\left(0, \min\left(\frac{x - a}{b - a}, \frac{c - x}{c - b}\right)\right)$$

#### Trapezoidal Membership Function ($\text{trapmf}$)
Defined by four scalar parameters $(a, b, c, d)$ where $a \le b \le c \le d$:
$$\mu_{\text{trap}}(x; a, b, c, d) = \max\left(0, \min\left(\frac{x - a}{b - a}, 1, \frac{d - x}{d - c}\right)\right)$$

#### Input Membership Function Parameters (from Paper Figure 2)
- **FIS1 Input MFs (Weekly Hours):**
  - `Very Low`: $\text{trapmf}(0.0, 0.0, 4.0, 8.0)$
  - `Low`: $\text{trimf}(5.0, 10.0, 15.0)$
  - `High`: $\text{trimf}(12.0, 18.0, 25.0)$
  - `Very High`: $\text{trapmf}(20.0, 30.0, 60.0, 60.0)$
- **FIS2 Input MFs (Daily Shift Hours):**
  - `Low`: $\text{trapmf}(0.0, 0.0, 2.0, 3.5)$
  - `Medium`: $\text{trimf}(2.5, 4.5, 6.5)$
  - `High`: $\text{trapmf}(5.0, 7.5, 12.0, 12.0)$

#### Shared Output Membership Functions (Assignment Likelihood)
FIS1 and FIS2 share five triangular output membership functions spanning $[0, 1]$:
1. `Very Low` (mf1): $[0.00, 0.05, 0.18]$
2. `Low` (mf2): $[0.10, 0.22, 0.38]$
3. `Medium` (mf3): $[0.28, 0.45, 0.62]$
4. `High` (mf4): $[0.50, 0.68, 0.85]$
5. `Very High` (mf5): $[0.75, 0.90, 1.00]$

---

### 3.4 Fuzzy Rule Bases (Expert Knowledge)

#### FIS1 Initial Rule Base ($4 \times 4 = 16$ Rules)
Rows represent `preferred_weekly_hours`; columns represent `assigned_weekly_hours`:

| Preferred Weekly Hours | Assigned: Very Low | Assigned: Low | Assigned: High | Assigned: Very High |
| :--- | :---: | :---: | :---: | :---: |
| **Very Low** | Low | Very Low | Very Low | Very Low |
| **Low** | High | Very Low | Very Low | Very Low |
| **High** | Very High | Low | Very Low | Very Low |
| **Very High** | Very High | High | Medium | Very Low |

*Intuitive Logic:* When current assigned hours are low and preferred hours are high, the likelihood of assignment is `Very High`. As assigned hours accumulate, the likelihood drops to `Very Low` to avoid overloading the worker.

#### FIS2 Initial Rule Base ($3 \times 3 = 9$ Rules)
Rows represent `preferred_shift_length`; columns represent `assigned_daily_hours`:

| Preferred Shift Length | Assigned Today: Low | Assigned Today: Medium | Assigned Today: High |
| :--- | :---: | :---: | :---: |
| **Low** | Very High | Very Low | Medium |
| **Medium** | Very High | High | Very High |
| **High** | Very High | High | Medium |

*Intuitive Logic:* Workers who have worked very little today receive high suitability for shift initiation. Workers who prefer longer shifts are maintained with high likelihood during medium hours to encourage contiguous shifts.

---

### 3.5 Mamdani Implication & Centroid Defuzzification

For any given input pair $(x_1, x_2)$:
1. **Fuzzification:** Compute membership degrees $\mu_{A_i}(x_1)$ and $\mu_{B_j}(x_2)$ for each linguistic term.
2. **Rule Evaluation:** For each rule $(i, j) \to k$, compute firing strength via Mamdani AND ($\min$):
   $$\alpha_{i,j} = \min\left(\mu_{A_i}(x_1), \mu_{B_j}(x_2)\right)$$
3. **Implication & Aggregation:** Aggregate firing strengths for each of the 5 output terms ($k \in \{1, 2, 3, 4, 5\}$) using max:
   $$w_k = \max_{(i,j) \text{ with consequent } k} (\alpha_{i,j})$$
   Clip output membership functions at level $w_k$:
   $$\mu_{\text{agg}}(u) = \max_{k=1}^5 \min\left(w_k, \mu_{\text{out},k}(u)\right), \quad u \in [0, 1]$$
4. **Centroid Defuzzification:** Discretize $u \in [0, 1]$ into 101 points:
   $$\text{Centroid} = \frac{\sum_{m=1}^{101} u_m \cdot \mu_{\text{agg}}(u_m)}{\sum_{m=1}^{101} \mu_{\text{agg}}(u_m)}$$
   If $\sum \mu_{\text{agg}} = 0$, default to $0.5$ (neutral likelihood).

---

### 3.6 The Multiplicative Aggregation Operator ($\Pi$) & Continuity Multiplier

For worker $w$ at time slot $t$:
$$\text{Suitability}(w, t) = A_{w,t} \times \min\left(\text{FIS1}(w), \text{FIS2}(w)\right) \times C(w, t)$$
Where:
- $A_{w,t} \in \{0, 1\}$ is availability. If $A_{w,t} = 0$, the entire product is **strictly zero**.
- $\min(\text{FIS1}, \text{FIS2})$ models a conservative bottleneck: a worker is only suitable if they qualify under *both* weekly and daily criteria.
- $C(w, t)$ is the **Continuity Scaling Factor**:
  $$C(w, t) = \begin{cases} 1.30 & \text{if worker } w \text{ was assigned to slot } t-1 \text{ on the same day} \\ 1.00 & \text{otherwise} \end{cases}$$
  This $30\%$ priority bonus encourages the scheduler to extend shifts into continuous blocks rather than creating scattered 1-hour fragments.

---

## 4. The Genetic Algorithm (GA) Optimization Engine

### 4.1 Why Genetic Fuzzy Systems (GFS)?
While the expert-crafted initial rule tables provide a solid starting point, they are not mathematically tuned for the specific statistical distributions of student availability and shift requests. The Genetic Algorithm automatically discovers:
1. Optimal linguistic rule consequents for unusual combinations.
2. Optimal center positions and spread widths of the 5 output membership functions.

### 4.2 The 34-Gene Chromosome Encoding
Matching Table 1 and Section 3.2 of the paper, every candidate individual is encoded as a 34-gene vector:

```
+-----------------------------------+-----------------------------+-------------------------------+
| Genes 0 to 15 (16 Genes)          | Genes 16 to 24 (9 Genes)    | Genes 25 to 33 (9 Genes)      |
| FIS1 Rule Table (4x4 Integers)    | FIS2 Rule Table (3x3 Integ) | Output MFs (Continuous Floats)|
| Allowed: {1, 2, 3, 4, 5}          | Allowed: {1, 2, 3, 4, 5}    | Allowed: [0.01, 0.99]         |
+-----------------------------------+-----------------------------+-------------------------------+
```

- **Genes 0–15:** 16 discrete integers representing rule outputs for FIS1 (1 = Very Low, 2 = Low, 3 = Medium, 4 = High, 5 = Very High).
- **Genes 16–24:** 9 discrete integers representing rule outputs for FIS2.
- **Genes 25–33:** 9 continuous float values. When sorted: $p_0 < p_1 < \dots < p_8$, they define the base corners and peaks of the 5 triangular output membership functions:
  - `Very Low`: $[0.0, p_0, p_1]$
  - `Low`: $[p_0, p_2, p_3]$
  - `Medium`: $[p_2, p_4, p_5]$
  - `High`: $[p_4, p_6, p_7]$
  - `Very High`: $[p_6, p_8, 1.0]$

This 9-point representation guarantees:
- Complete coverage of $[0, 1]$.
- Valid ordering ($a \le b \le c$).
- Overlapping adjacent membership functions with zero degenerate triangles.

---

### 4.3 Multi-Scenario RMS Fitness Function
The GA evaluates each chromosome across $n$ independent workforce scenarios ($n = 30$ in paper training setup, or $n = 5$ for quick interactive mode).  
Following Eq. (1) of Nguyen & Cohen:
$$\text{Fitness} = \frac{1}{n} \sum_{i=1}^n \left[ \text{RMS}(\Delta_{1,i}) + \text{RMS}(\Delta_{2,i}) + \text{RMS}(\Delta_{3,i}) \right]$$

Where for scenario $i$:
1. **Daily Shift Length Deviation ($\Delta_1$):**
   $$\Delta_1(w, d) = \text{assigned\_hours}(w, d) - L_{pref}(w)$$
   *Important rule:* If worker $w$ receives $0$ hours on day $d$, $\Delta_1$ is **not calculated**.
   $$\text{RMS}(\Delta_1) = \sqrt{\frac{1}{K} \sum_{(w,d) \in \text{Assigned}} (\Delta_1(w, d))^2}$$
   where $K$ is the total count of assigned worker-days.
2. **Weekly Working Hours Deviation ($\Delta_2$):**
   $$\Delta_2(w) = \text{assigned\_weekly\_hours}(w) - H_{pref}(w)$$
   $$\text{RMS}(\Delta_2) = \sqrt{\frac{1}{N} \sum_{w=1}^N (\Delta_2(w))^2}$$
3. **Weekly Maximum Limit Penalty ($\Delta_3$):**
   $$\Delta_3(w) = \max\left(0, \text{assigned\_weekly\_hours}(w) - 25.0\right)$$
   $$\text{RMS}(\Delta_3) = \sqrt{\frac{1}{N} \sum_{w=1}^N (\Delta_3(w))^2}$$
4. **Demand Feasibility Penalty:** If a candidate schedule fails to staff 4 workers in any slot, a penalty of $50.0 \times \text{uncovered\_slots}$ is added to guide the GA toward feasible solutions.

---

### 4.4 Genetic Operators: Selection, Crossover, Mutation, and Elitism
- **Selection:** Minimization-adapted Roulette Wheel Selection. Given fitness values $f_i$, raw selection weights are:
  $$w_i = (\max(f) + 1.0) - f_i, \quad p_i = \frac{w_i}{\sum_j w_j}$$
  Lower fitness candidates receive proportionally higher selection probabilities.
- **Elitism:** The top 10 best-performing chromosomes are copied directly into the next generation without modification, guaranteeing that the best solution never degrades.
- **Crossover (Fraction = 0.8):**
  - Discrete rule genes (0–24): Two-point crossover.
  - Continuous MF genes (25–33): Arithmetic blending with random interpolation factor $\alpha \sim U(0.2, 0.8)$:
    $$c_1 = \alpha p_1 + (1 - \alpha) p_2, \quad c_2 = (1 - \alpha) p_1 + \alpha p_2$$
- **Mutation (Fraction = 0.2):**
  - Discrete rule genes: Random replacement with an integer in $\{1, 2, 3, 4, 5\}$.
  - Continuous MF genes: Gaussian perturbation $\mathcal{N}(0, 0.05^2)$, followed by clipping to $[0.01, 0.99]$ and re-sorting.

---

## 5. Schedule Generation & Hard Constraint Enforcement

### 5.1 Chronological Sequential Slot Assignment Algorithm
The scheduler proceeds chronologically from Monday morning ($t = 0$) to Friday afternoon ($t = 50$):

```python
For t = 0 to 50:
    day = slot_days[t]
    If day != current_day:
        assigned_today.fill(0)
        current_day = day

    # Step 1: Identify eligible workers
    eligible = []
    For each worker w:
        If availability[w, t] == 1:
            If assigned_weekly[w] + 1.0 <= max_weekly_hours[w]:
                eligible.append(w)

    # Step 2: Compute fuzzy suitability scores
    scores = []
    For w in eligible:
        prev_assigned = (t > 0 and slot_days[t-1] == day and schedule[w, t-1] == 1)
        score1 = FIS1(pref_weekly[w], assigned_weekly[w])
        score2 = FIS2(pref_shift[w], assigned_today[w])
        cont = 1.30 if prev_assigned else 1.00
        final_score = min(score1, score2) * cont
        scores.append((w, final_score))

    # Step 3: Rank descending and select top 4
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [x[0] for x in scores[:4]]

    # Step 4: Commit assignments
    For w in selected:
        schedule[w, t] = 1
        assigned_weekly[w] += 1.0
        assigned_today[w] += 1.0
```

### 5.2 Pre-Optimization Feasibility Audit
Before launching the GA, the system executes an automated feasibility check:
1. Calculates required worker-hours: $51 \times 4 = 204\text{ hours}$.
2. Calculates available worker-hours: $\sum_{w,t} A_{w,t}$.
3. Evaluates per-slot availability: If for any slot $t$, $\sum_w A_{w,t} < 4$, the scenario is mathematically impossible to staff. The app immediately flags `INFEASIBLE` with the specific slot indices, preventing wasted computation.

### 5.3 Independent Post-Scheduling Constraint Verification
Every generated roster is audited by `validate_schedule_constraints()`:
- `unavailable_assignments == 0`
- `uncovered_slots == 0`
- `weekly_cap_violations == 0`
- `min_shift_violations == 0`
Only if all four checks pass is the roster certified as `FEASIBLE`.

### 5.4 Continuous Shift Extraction
To make schedules operationally practical for management, the binary matrix is parsed into a tabular shift roster:
`worker_id`, `day`, `start_time`, `end_time`, `duration_hours`.
For example:
- `W01` assigned to slots 0, 1, 2, 3 on Day 1 is formatted as:  
  `Day 1 | 08:30 | 12:30 | 4.0 hrs`.

---

## 6. Comparative Baselines

To rigorously evaluate the Genetic Algorithm's contribution, three benchmark methods are implemented under identical scenario seeds:

### 6.1 Baseline A: Non-GA Fixed-Rule Fuzzy Scheduler (Ablation)
- Employs the identical dual Mamdani FIS architecture and multiplicative aggregator.
- Uses the paper's initial expert-crafted rule tables (Figure 5) and default membership functions (Figure 6) without evolutionary tuning.
- **Ablation Purpose:** Isolates the exact improvement contributed by the Genetic Algorithm.

### 6.2 Baseline B: Preference-Aware Greedy Heuristic
- A direct algorithmic heuristic without fuzzy logic.
- At each slot, ranks available workers by priority score:
  $$\text{Priority} = (H_{pref}(w) - H_{assigned}(w)) + 0.5 \times (L_{pref}(w) - L_{today}(w)) + \text{Bonus}_{cont}$$
- Respects the identical hard constraints (availability, 25h cap, 4-worker coverage).

### 6.3 Baseline C: Random Eligible Assignment (Stress Test)
- Randomly samples 4 available workers who have not exceeded the 25h cap.
- Serves as a baseline to measure the performance floor.

---

## 7. Comprehensive Measured Results & Findings

### 7.1 Numerical Comparison Table (Actual Measured Values)
The following results were measured directly by running the testbench on the 20-worker, 51-slot reference dataset under identical scenario seeds:

| Metric | GA-Optimized Fuzzy | Non-GA Fuzzy Baseline | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Objective Fitness (RMS Sum)** | **`5.2119`** | `6.6511` | **21.6% lower (better)** |
| **RMS($\Delta_1$) Daily Shift Length** | **`3.5687`** | `3.6179` | **1.4% lower** |
| **RMS($\Delta_2$) Weekly Working Hours**| **`1.6432`** | `3.0332` | **45.8% lower** |
| **RMS($\Delta_3$) Weekly Cap Excess** | **`0.0000`** | `0.0000` | Strict 0 violations |
| **Weekly Preference MAE** | **`1.30 hrs`** | `2.40 hrs` | **45.8% lower error** |
| **Daily Shift Length MAE** | **`2.81 hrs`** | `3.12 hrs` | **9.9% lower error** |
| **Preference Satisfaction ($\pm 2\text{h}$)** | **`85.0%`** | `60.0%` | **+25.0% absolute increase** |
| **Staffing Coverage Rate** | **`100.0%`** | `100.0%` | 0 uncovered slots |
| **Hard Constraint Violations** | **`0`** | `0` | Zero tolerance upheld |
| **Overall Feasibility** | **`FEASIBLE`** | **`FEASIBLE`** | Certified valid |
| **Weekly Assigned Hours StdDev** | **`2.80 hrs`** | `4.18 hrs` | 33.0% more balanced |
| **Weekly Assigned Hours Range** | **`7.0 hrs`** | `14.0 hrs` | 50.0% narrower disparity |
| **Workers Assigned Zero Hours** | **`0`** | `0` | No worker excluded |
| **Jain's Fairness Index** | **`0.930`** | `0.856` | Higher allocation equality |
| **Schedule Generation Runtime** | **`0.048 s`** | `0.050 s` | Real-time scheduling speed |

---

### 7.2 Convergence Trajectory Analysis
During GA evolution, the multi-scenario fitness consistently converged downward across generations:

| Generation | Best Fitness | Mean Fitness | Median Fitness | Feasible Ratio |
| :---: | :---: | :---: | :---: | :---: |
| **Gen 0 (Init)** | `6.6511` | `9.4820` | `8.9140` | 100% |
| **Gen 2** | `6.1420` | `8.3210` | `8.0450` | 100% |
| **Gen 4** | `5.6890` | `7.5400` | `7.3100` | 100% |
| **Gen 6** | `5.3410` | `6.8920` | `6.7500` | 100% |
| **Gen 8 (Best)** | **`5.2119`** | `6.4150` | `6.3200` | 100% |

*Key Takeaway:* Within 8 generations, the GA reduced the objective fitness from $6.6511$ down to $5.2119$, proving that evolving fuzzy rule consequents and membership functions significantly improves preference satisfaction.

---

### 7.3 Preference Satisfaction & Workload Distribution
- **Weekly Working Hours:** The Non-GA baseline produced an average error of $2.40\text{ hours}$ per worker, with some workers receiving up to 5 hours more or less than requested. The GA-optimized system compressed the weekly MAE down to $1.30\text{ hours}$.
- **High Satisfaction Rate:** $85.0\%$ of all workers received weekly hours within $\pm 2.0\text{ hours}$ of their exact submitted preference, compared to only $60.0\%$ under the fixed-rule baseline.
- **Zero Cap Violations:** All 20 workers remained strictly under the 25-hour weekly limit. The maximum assigned hours for any worker was 15 hours.

---

### 7.4 Fairness Analysis & Jain's Fairness Index
Fairness was evaluated using Jain's Fairness Index:
$$J(x) = \frac{\left(\sum_{i=1}^N x_i\right)^2}{N \sum_{i=1}^N x_i^2}$$
- **Non-GA Baseline:** $J = 0.856$ (Hours standard deviation: $4.18\text{h}$, Range: $14.0\text{h}$).
- **GA-Optimized Model:** $J = 0.930$ (Hours standard deviation: $2.80\text{h}$, Range: $7.0\text{h}$).
The GA model distributed the 204 required worker-hours substantially more equitably while maintaining full alignment with individual preferences.

---

### 7.5 Understaffed Robustness Stress Test (16 Workers)
Matching Section 6 of the paper, an understaffed scenario was evaluated with only 16 available workers:
- **Total Required Hours:** $51 \times 4 = 204\text{ worker-hours}$.
- **Total Requested Hours:** $16 \times 10.5\text{h avg} \approx 168\text{ hours}$.
- **Condition:** Demand exceeds total requested hours by $36\text{ worker-hours}$.
- **Findings:**
  - The GA scheduler successfully covered 100% of operational demand ($204/204\text{ hours}$).
  - All workers remained within the 25-hour legal maximum.
  - The objective fitness increased to $7.15$ (higher cost due to workers absorbing extra shifts beyond their preference), matching the variance increase documented by Nguyen & Cohen in Figure 7.

---

## 8. Explainable AI (XAI) Slot Decision Inspector

Unlike deep learning schedulers that function as black boxes, every shift allocation in this system can be inspected and audited.

### Real Example from Time Slot 0 (Day 1, 08:30 AM):
- **Demand:** 4 workers needed.
- **Eligible Available Candidates:** 14 workers available.

| Worker ID | Pref Weekly | Curr Weekly | FIS1 Score | Pref Shift | Today Hours | FIS2 Score | Continuity Bonus | Final Score | Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **W05** | 15.0h | 0.0h | 0.892 | 6.0h | 0.0h | 0.840 | 1.00× | **0.8400** | **ASSIGNED (#1)** |
| **W11** | 14.0h | 0.0h | 0.865 | 3.0h | 0.0h | 0.840 | 1.00× | **0.8400** | **ASSIGNED (#2)** |
| **W03** | 13.0h | 0.0h | 0.825 | 8.0h | 0.0h | 0.840 | 1.00× | **0.8250** | **ASSIGNED (#3)** |
| **W10** | 13.0h | 0.0h | 0.825 | 4.0h | 0.0h | 0.840 | 1.00× | **0.8250** | **ASSIGNED (#4)** |
| **W02** | 12.0h | 0.0h | 0.780 | 7.0h | 0.0h | 0.840 | 1.00× | 0.7800 | EXCLUDED (#5) |
| **W01** | 5.0h | 0.0h | 0.320 | 6.0h | 0.0h | 0.840 | 1.00× | 0.3200 | EXCLUDED (#14) |

*Audit Interpretation:*
- Workers W05, W11, W03, and W10 requested high weekly hours (13–15h). FIS1 assigned them high suitability ($0.825 - 0.892$).
- Worker W01 requested only 5 hours per week. FIS1 assigned a low score of $0.320$, properly deferring W01 to preserve their low weekly allocation.
- In Slot 1 (09:30 AM), W05, W11, W03, and W10 receive the $1.30\times$ continuity bonus, keeping them on duty to form a continuous 4-hour block.

---

## 9. Codebase Architecture & File-by-File Breakdown

```
SCN aat/
├── app.py                         # Streamlit multi-tab web application (Tabs 1-7)
├── COMPLETE_PROJECT_GUIDE.md      # This master document
├── README.md                      # Windows setup & launch instructions
├── requirements.txt               # Pinned dependencies
├── pyproject.toml                 # Package definition & pytest config
├── configs/
│   ├── default_config.yaml        # Fast interactive parameters
│   └── paper_reference_config.yaml # Exact paper setup (200 pop, 50 gen, n=30)
├── data/
│   ├── sample/
│   │   ├── sample_workers.csv     # 20 student profiles from paper
│   │   ├── sample_availability.csv # 1,020 binary slot availability entries
│   │   ├── sample_demand.csv      # 51-slot demand (4 workers/slot)
│   │   └── sample_workforce.xlsx  # Multi-sheet Excel workbook
├── src/workforce_scheduler/
│   ├── __init__.py                # Package version
│   ├── config.py                  # YAML loader and configuration dataclasses
│   ├── data_loader.py             # CSV and multi-sheet Excel loader with SHA-256
│   ├── schema_detection.py        # Automatic column alias detector
│   ├── validation.py              # Pre-optimization feasibility auditor
│   ├── preprocessing.py           # Canonical data formatting and slot mapping
│   ├── scenario_generator.py      # 70-student pool & 20/16-worker scenario generator
│   ├── membership_functions.py    # Triangular/trapezoidal MFs and gene decoders
│   ├── fuzzy_system.py            # Mamdani FIS1, FIS2, and Pi aggregator
│   ├── chromosome.py              # 34-gene chromosome encoding, repair, validation
│   ├── fitness.py                 # Eq. (1) multi-scenario RMS fitness calculation
│   ├── genetic_algorithm.py       # GA engine (roulette wheel, elitism, crossover, mutation)
│   ├── scheduler.py               # Sequential slot scheduler & shift builder
│   ├── constraints.py             # Hard constraint validator
│   ├── baselines.py               # Non-GA Fuzzy & Heuristic baseline runners
│   ├── metrics.py                 # Preference MAE, satisfaction rate, Jain index
│   ├── experiments.py             # Comparative testbench runner
│   ├── visualizations.py          # Plotly & Matplotlib figure generators
│   ├── exports.py                 # CSV, JSON, and PNG exporters
│   └── report_builder.py          # Academic research report generator
├── tests/                         # 28 passing unit and integration tests
└── outputs/
    ├── schedules/                 # Exported slot matrix and shift rosters
    ├── metrics/                   # Exported comparative metrics and coverage tables
    ├── figures/                   # 300 DPI publication figures
    └── reports/                   # Generated academic markdown reports
```

---

## 10. Step-by-Step User & Developer Execution Guide

### Prerequisites
- Windows 10/11 operating system.
- Python 3.11 installed.

### Step 1: Open PowerShell and Navigate to the Project
```powershell
cd "c:\Users\Ragu\Documents\SCN aat"
```

### Step 2: Activate the Virtual Environment
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 3: Run the Pytest Test Suite
```powershell
python -m pytest -v
```
*Expected Output:*
```text
============================= 28 passed in 4.29s =============================
```

### Step 4: Launch the Streamlit Web Application
```powershell
python -m streamlit run app.py
```
*Expected Terminal Output:*
```text
Local URL: http://localhost:8501
```

### Step 5: How to Use the Web Interface
1. **Tab 1 (Overview & Method):** Read the problem overview. Click **"Load Paper Reference Demo Dataset"**.
2. **Tab 2 (Data Upload & Validation):** View worker requested hours and slot availability. Notice the pre-optimization feasibility audit confirms `FEASIBLE`. (You can also upload custom CSV/Excel files).
3. **Tab 3 (Fuzzy System Inspection):** Inspect the rule tables for FIS1 and FIS2 and view the output membership functions.
4. **Tab 4 (Optimization & Baselines):**
   - Click **"Run GA Optimization on Active Dataset"** to run a quick optimization (~15 seconds).
   - View the live progress bar and convergence curve.
5. **Tab 5 (Schedule & Explainability):**
   - View the **Weekly Schedule Heatmap** showing dark blocks of continuous shifts.
   - Inspect the **Tabular Shift Roster** with start and end times.
   - Check the **Operational Coverage** plot (exactly 4 workers in every slot).
   - Use the **Explainable AI Inspector** to select any time slot and examine the exact mathematical reason why workers were chosen.
6. **Tab 6 (Comparative Benchmark):** View the comparative bar charts and tables comparing GA-Fuzzy vs. Non-GA Baseline vs. Heuristic.
7. **Tab 7 (Export & Academic Report):** Download CSV rosters, slot matrices, coverage summaries, and the generated academic Markdown report.

---

## 11. Paper Fidelity vs. Implementation Adaptations

| System Component | Research Paper (Nguyen & Cohen) | Our Implementation | Classification |
| :--- | :--- | :--- | :--- |
| **Operating Week** | 5 days: Day 1–4 (11h), Day 5 (7h) = 51 hours | 5 days, 51 discrete 1-hour slots | **Paper Method** |
| **Headcount Demand** | At least 4 workers on duty at all times | 4 workers/slot (configurable) | **Paper Method** |
| **Workforce Size** | ~20 student workers (16 understaffed) | 20 workers standard, 16 understaffed | **Paper Method** |
| **Hour Limits** | 25 hours weekly maximum | 25 hours weekly maximum hard constraint | **Paper Method** |
| **Fuzzy Inference** | Mamdani FIS1 & FIS2 with centroid defuzzification | Pure NumPy Mamdani FIS with centroid defuzzification | **Paper Method** |
| **Aggregation ($\Pi$)**| Availability $\times \min(\text{FIS1}, \text{FIS2}) \times \text{Continuity}$ | Exact multiplicative formulation with $C_{cont}=1.30$ | **Paper Method** |
| **Chromosome** | 34 genes (16 FIS1 + 9 FIS2 + 9 output MFs) | Exact 34-gene hybrid chromosome | **Paper Method** |
| **Fitness Function** | $\frac{1}{n} \sum [\text{RMS}(\Delta_1) + \text{RMS}(\Delta_2) + \text{RMS}(\Delta_3)]$ | Exact mathematical formulation (Eq. 1) | **Paper Method** |
| **Constraint Validation**| Described in prose; cap penalty in fitness | Pre- and post-optimization constraint audit | **Adaptation** |
| **Baselines** | GA vs Non-GA ablation | Non-GA Fuzzy + Greedy Heuristic + Random | **Additional Evaluation** |
| **Data Ingestion** | Fixed unreleased student dataset | Dynamic CSV/Excel upload with automatic schema detection | **Additional Feature** |
| **Explainable AI** | Discussion of rule transparency | Interactive per-slot decision trace inspector in UI | **Additional Feature** |

---

## 12. Limitations, Discussion & Conclusion

### 12.1 Key Insights
1. **Fuzzy Logic Eliminates Sharp Cutoffs:** Unlike rigid heuristic rules, fuzzy sets smoothly transition between "Low" and "High" workloads, avoiding edge-case assignment oscillations.
2. **Genetic Algorithm Delivers Measurable Optimization:** Evolutionary tuning of rule consequents and membership functions reduced weekly hours error by $45.8\%$ and raised preference satisfaction from $60\%$ to $85\%$.
3. **Continuity vs. Equity Trade-Off:** The continuity multiplier ($C_{cont} = 1.30$) successfully creates realistic continuous shifts (average shift length: $3.5\text{ hours}$), but naturally concentrates hours on workers already on duty on a given day.
4. **Guaranteed Operational Feasibility:** By enforcing independent pre-auditing and post-validation, the system guarantees that operational staffing is never compromised to satisfy individual preferences.

### 12.2 Limitations & Future Work
- **Chronological Bias:** Sequential slot scheduling (Monday morning to Friday afternoon) can occasionally constrain late Friday allocations for workers who have limited availability. A bidirectional or look-ahead scheduling heuristic is a promising future extension.
- **Heterogeneous Skill Roles:** The paper assumes homogeneous student workers. Future iterations can incorporate role constraints (e.g., supervisor, technician) and seniority preferences.

---

### End of Documentation
*All metrics, figures, schedules, and test suites are saved and verifiable in `outputs/` and `tests/`.*
