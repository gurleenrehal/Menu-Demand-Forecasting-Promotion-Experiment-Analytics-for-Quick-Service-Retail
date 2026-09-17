# Dashboard Specification

This project ships static Plotly/Matplotlib charts (`reports/figures/`) rather
than a live BI file, so the repo runs from a single `pip install` with no
Tableau/Power BI license required. The KPIs below are written the way they'd
be laid out on a Tableau or Power BI dashboard, so the design can be rebuilt
in either tool directly from `data/coffee_shop_sales.csv`.

## Page 1 — Executive Overview
- KPI cards: Total Revenue, Total Transactions, Avg. Order Value, Revenue
  Growth MoM
- Line chart: Revenue by month, with a store-location filter
- Bar chart: Revenue by store location
- Bar chart: Revenue by product category

## Page 2 — Demand & Operations
- Heatmap: Transactions by hour x day-of-week (staffing insight)
- Line chart: Actual vs. forecast daily units sold, filterable by
  store/category (feeds directly from the model's predictions)
- Table: Top 15 products by revenue and by transaction count

## Page 3 — Experimentation
- Power-analysis calculator: required sample size given assumed uplift %,
  baseline mean/std (mirrors `src/experiment_design.py`)
- Simulated A/B test result panel: control vs. treatment mean, lift %,
  p-value, confidence interval — clearly labeled "simulated" per
  `data/README.md`

## KPI Definitions
| KPI | Formula |
|---|---|
| Revenue | `SUM(transaction_qty * unit_price)` |
| Avg. Order Value | `Revenue / COUNT(DISTINCT transaction_id)` |
| Repeat-purchase rate | Not computable from this dataset (no customer_id) — flagged as a gap, not fabricated |
| Forecast MAPE | `MEAN(ABS((actual - forecast) / actual)) * 100` |
