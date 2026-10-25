# GitHub Copilot & AI Development Instructions - Food Demand Analysis

## Project Mission
Analyze 145 weeks of food fulfillment data across 77 fulfillment centers and 51 meal items to resolve operational inventory mismatch (spoilage vs. stockouts), quantify promotion lifts, and build actionable demand forecasting.

## Core Rules for All AI Assistance

### Rule 1: Code Standards, Immutability & Data Integrity
* NEVER modify, overwrite, or mutate any raw files in `data/raw/`. Treat them as strictly read-only.
* ALWAYS print and assert the row counts and sum of `num_orders` before and after any table join, merge, or filtering operation.
* Every cleaning decision must be explicitly logged in `docs/cleaning_decisions.md` with count of affected records and technical justification.
* Write clean, reproducible, and PEP8-compliant Python code. Set random seeds (`random_state=42`) for any stochastic processes.

### Rule 2: Analytical Honesty, Traceability & Non-Causality
* No number, percentage, or finding may appear in `reports/client_report.md` or `README.md` unless it is explicitly derived and computed in `notebooks/analysis.ipynb`.
* NEVER use causal terminology such as "causes", "caused by", or "يسبب" without experimental/A-B trial verification. Promotions frequently co-occur with price discounts; use terms like "is associated with" or "يرتبط بـ".
* Distribution of orders is right-skewed; always apply non-parametric statistical methods (Mann-Whitney U test, Spearman rank correlation) rather than assuming normality.
* Every report must feature a prominent "Limitations" section discussing masked variables, lack of real calendar dates, and unobserved supply constraints.

## Custom Skills Available
* `.github/skills/promotion-impact-evaluator/SKILL.md`: Controls within-center/meal comparisons, Mann-Whitney U testing, and promotion lift computation.
* `.github/skills/demand-volatility-kpis/SKILL.md`: Computes revenue, discount percentage, Pareto 80/20 breakdown, and Coefficient of Variation (CV) for inventory buffer planning.
