# IMPLEMENTATION.md

## End-to-End AI Coding-Agent Specification

### Soft Computing AAT: Fuzzy Logic--Based Scheduling System for Part-Time Workforce

**Purpose:** This document is the implementation contract for an AI coding agent. Build a complete, locally runnable Streamlit application that implements and evaluates a fuzzy-logic workforce scheduler optimized with a genetic algorithm, based on the research paper *Fuzzy Logic--Based Scheduling System for Part-Time Workforce* by Tri Nguyen and Kelly Cohen (University of Cincinnati).

**Important instruction:** Implement the project described here, but do not treat this specification as permission to fabricate experimental results. Run the code, test the actual behavior, and report only measured results. Preserve the paper's method where specified. Label all extensions and adaptations clearly in the UI, outputs, and academic report.

---

## 1. Paper Overview & Reference Scenario Specifications
- **Workforce:** Approximately 20 part-time student workers (University of Cincinnati front-desk assistant model).
- **Operating week:** 5 days (Mon–Fri).
- **Hours per day:** 11 hours/day for first 4 days (Day 1–4), 7 hours on final day (Day 5). Total: 51 operating hours.
- **Coverage Demand:** At least 4 workers on duty at all times (204 worker-hours total).
- **Minimum Shift Length:** 1 hour.
- **Weekly Maximum Hours:** 25 hours per worker.
- **Preferences:** Uniform random distribution for training/testing scenarios:
  - Weekly preferred hours: 5 to 15 hours.
  - Preferred shift length: 3 to 8 hours.
- **Availability:** Binary matrix (worker x slot: 20 x 51).
- **FIS1:** Weekly workload suitability (Inputs: Preferred weekly hours, Current assigned weekly hours).
- **FIS2:** Daily shift suitability (Inputs: Preferred shift length, Current assigned daily hours).
- **Aggregation:** $\Pi = \text{Availability} \times \min(\text{FIS1}, \text{FIS2}) \times \text{Continuity Factor}$.
- **GA Optimization:** Roulette wheel selection, EliteCount = 10, Crossover fraction = 0.8, Mutation fraction = 0.2, Population = 200, Max generations = 50, Stall generations = 10, Chromosome length = 34.
- **Fitness:** $\text{Fitness} = \frac{1}{n} \sum_{i=1}^n [\text{RMS}(\Delta_1) + \text{RMS}(\Delta_2) + \text{RMS}(\Delta_3)]$.
