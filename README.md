# Menu Demand Forecasting & Promotion-Experiment Analytics for Quick-Service Retail

Built for the **Yum! Brands (Yum India Global Services) Data Scientist Intern**
application. This is an independent portfolio project using public data — it
does not use or claim access to any Yum!/KFC/Pizza Hut/Taco Bell/Habit
internal data.

## 1. Business Problem
Quick-service restaurant chains run on thin margins where two decisions
repeat every day: **how much of each menu item to prep** (too little =
stockouts and lost sales, too much = waste) and **whether a proposed
promotion is actually worth running**. This project builds (a) a next-day
demand forecasting model at the store x product-category level and (b) the
statistical experiment-design toolkit a Data Scientist would use to validate
a promo before it launches at scale — the same two problems Yum!'s
menu-management and digital-commerce platforms exist to solve.

## 2. Executive Summary
Using 149,116 real transactions from a 3-location coffee/bakery retailer
(Jan-Jun 2023), I built a full pipeline from raw transaction log to
next-day demand forecasts and an experiment-design tool. An XGBoost model
cuts next-day demand forecast error by **37% vs. a naive baseline** (MAE
10.4 vs. 16.5 units/day) and 4 hypothesis tests surface concrete, actionable
findings (e.g., store location is a statistically significant driver of
basket size; day-of-week is not).

## 3. Objectives
1. Forecast next-day demand per store x product-category using only
   information available the day before.
2. Quantify which operational/business factors actually drive basket size
   and demand, using proper hypothesis tests (not just eyeballing charts).
3. Build a reusable experiment-design workflow (power analysis + simulated
   A/B readout) for evaluating a future promotion.
4. Package all of the above the way a production analytics team would —
   modular code, tests of the pipeline's own assumptions, a documented data
   limitation section, and a BI dashboard spec.

## 4. Dataset
"Coffee Shop Sales" — Maven Analytics Data Playground / Kaggle, public
domain. 149,116 rows, 11 columns, Jan 1 - Jun 30 2023, 3 NYC stores, 9
product categories. Full column dictionary and stated limitations (no
customer ID, single city, no real promo flag) are in `data/README.md` —
please read that section before the results below, since it explains what
this dataset can and cannot support.

## 5. Methodology
Data Understanding → Cleaning (dedupe / null / non-positive filters — none
were needed, the source data was already clean) → EDA → Statistical
Analysis → Feature Engineering (daily demand table + lag/rolling/calendar
features) → Modeling (baseline → Linear Regression → Random Forest →
XGBoost) → Cross-validation (`TimeSeriesSplit`, no shuffling) →
Explainability (feature importance + SHAP) → Experiment design → Business
recommendations.

## 6. Data Cleaning
0 missing values and 0 duplicate rows were found (see `01_data_understanding.ipynb`).
`src/data_preprocessing.py` still enforces `transaction_qty > 0` and
`unit_price > 0` and parses `transaction_time` into an hour field, so the
pipeline is defensive even though this particular file didn't need it —
that guard is what makes the same code safe to point at a messier dataset.

## 7. EDA — Key Findings
- Revenue grew from **$81.7K (Jan) to $166.5K (Jun)** — roughly doubling
  over 6 months.
- Coffee (**$270K**) and Tea (**$196K**) are ~67% of total revenue
  ($698.8K); Packaged Chocolate, Flavours and Loose Tea are long-tail
  categories.
- The three stores are close in total revenue ($230K-$237K each) but, per
  the ANOVA below, not close in *average transaction size*.
- Transactions peak 7-10 AM, consistent with a coffee-shop morning-rush
  pattern.
(Charts: `reports/figures/01`–`05`.)

## 8. Statistical Analysis
| Test | Question | Result | Interpretation |
|---|---|---|---|
| Welch's t-test | Does avg. transaction revenue differ weekday vs. weekend? | p = 0.743 (not significant) | Basket size doesn't change on weekends — any weekend revenue swing is a *volume* story, not a basket-size story |
| One-way ANOVA | Does avg. transaction revenue differ by store? | F = 36.09, p < 0.001 (significant) | Lower Manhattan ($4.81) > Hell's Kitchen ($4.66) > Astoria ($4.59) — stores are not interchangeable |
| Chi-square | Is product-category mix associated with store? | χ² = 1009.9, p < 0.001 (significant) | Confirms store-specific assortment/promo planning is justified by the data, not just intuition |
| Pearson correlation | unit_price vs. quantity purchased | r = -0.12, p < 0.001 (significant but weak) | Statistically real, practically small — a caution against over-reading a large-n significant result as a strong effect |

Full null/alternative hypotheses, test statistics and code: `03_statistical_analysis.ipynb`, `src/statistical_analysis.py`.

## 9. Feature Engineering
The line-item log is re-aggregated into a **daily demand table**
(store x category x day, with zero-filled gap days) and enriched with:
`lag_1`, `lag_7`, `rolling_mean_7`, `rolling_mean_14`, `rolling_std_7`,
plus calendar fields (day-of-week, weekend flag, month, day-of-month) and
one-hot encoded store/category. Rows in the first 14 days of each series are
dropped because their rolling windows aren't full yet — a disclosed
trade-off, not missing data. See `src/feature_engineering.py`.

## 10. ML Methodology
Target: next-day `units_sold` per store x category (regression). Time-based
80/20 train/test split (test = most recent ~20% of days — never randomly
shuffled, since shuffling a time series leaks the future into training).
`TimeSeriesSplit` (5 folds) used for cross-validation on the training set.
Models: naive lag-7 baseline, Linear Regression, Random Forest, XGBoost.

## 11. Model Comparison
| Model | Test MAE | Test RMSE | Test MAPE | CV MAE (train) |
|---|---|---|---|---|
| Baseline (naive, lag-7) | 16.53 | 28.31 | 50.53% | — |
| Linear Regression | 11.90 | 19.76 | 45.10% | 9.89 |
| Random Forest | 10.46 | 19.23 | **31.59%** | **8.87** |
| **XGBoost** | **10.40** | **18.34** | 32.93% | 9.22 |

XGBoost and Random Forest are close; XGBoost is used as the final model for
its lower test MAE/RMSE, with Random Forest noted as a strong, slightly more
interpretable alternative (Random Forest has the better MAPE and the best
cross-validated MAE, so a team could reasonably pick either).

## 12. Evaluation Metrics
Regression task → MAE, RMSE, MAPE (all reported above); R² is also computed
in `src/evaluation.py` for completeness. Classification/clustering metrics
were not used since there is no classification or clustering task in this
project.

## 13. Explainability
XGBoost feature importance and SHAP values agree: `rolling_mean_14` and
`rolling_mean_7` dominate (recent demand momentum), followed by `lag_1`,
`month`, and the Coffee category flag. Store/category one-hot features
matter far less than recency features — i.e., "what happened here in the
last two weeks" beats "which store/category is this" as a predictor. See
`05_modeling.ipynb` for the SHAP summary plot.

## 14. Key Findings
1. A 37% MAE reduction over a naive baseline is achievable with off-the-shelf
   gradient boosting and no external data (weather, holidays, foot traffic)
   — those would be the natural next features to add.
2. Store location is a real, statistically significant driver of basket
   size; day-of-week is not.
3. Recent rolling demand outweighs calendar effects for forecasting —
   important for deciding what data a production system needs to keep fresh.

## 15. Business Recommendations
1. Prioritize forecasting accuracy investment on Coffee and Tea (67% of
   revenue) over long-tail categories.
2. Investigate *why* Lower Manhattan's basket size is higher before
   assuming a promo strategy transfers directly to Astoria.
3. Before running a real promotional experiment, use the power analysis in
   `src/experiment_design.py` to size it correctly — running it under-sized
   (as the simulated example deliberately does, 4,000 vs. the required
   5,110 transactions/arm) risks either missing a real effect or over
   trusting a lucky significant result.
4. Because this dataset has no `customer_id`, a natural next iteration is
   joining this demand-forecasting work with the customer-level churn/
   cross-sell models already in this candidate's portfolio (see the parent
   application notes) to move from *aggregate* to *personalized* demand
   signals.

## 16. Dashboard
See `dashboard/README.md` for the full KPI/page spec (built to be
reproduced in Tableau or Power BI); static equivalents of the core charts
are in `reports/figures/`.

## 17. Technology Stack
Python 3.11+, pandas, numpy, scipy, statsmodels, scikit-learn, XGBoost,
SHAP, matplotlib, seaborn, Jupyter. SQL is demonstrated separately (see the
ATS mapping in the application notes) since this dataset was analyzed
in-pipeline rather than queried from a warehouse.

## 18. Repository Structure
```
menu-demand-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── coffee_shop_sales.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_statistical_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modeling.ipynb
│   └── 06_business_insights.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── statistical_analysis.py
│   ├── experiment_design.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── visualization.py
├── models/
│   ├── xgboost_demand_forecast.pkl
│   └── random_forest_demand_forecast.pkl
├── reports/
│   ├── model_comparison.csv
│   └── figures/
└── dashboard/
    └── README.md
```

## 19. How to Run
```bash
pip install -r requirements.txt
cd src && python data_preprocessing.py      # cleans data, prints a quality report
python statistical_analysis.py              # runs all 4 hypothesis tests
python feature_engineering.py               # builds the daily demand feature table
python modeling.py                          # trains + compares all 4 models
python experiment_design.py                 # power analysis + simulated A/B test
# or: jupyter notebook ../notebooks   (run 01 through 06 in order)
```

## 20. Limitations
- Single retail concept, single city, 6 months of history — see
  `data/README.md` for the full, upfront list.
- No customer-level data, so this project cannot speak to personalization,
  loyalty, or churn — that's covered elsewhere in this candidate's
  portfolio, deliberately not duplicated here.
- The A/B test is a labeled simulation, not a real measured business
  result — see `src/experiment_design.py`'s docstring.
- No cloud deployment or MLOps pipeline is implemented; see "Skills to
  learn" in the application notes for what a production version would add.

## 21. Future Improvements
- Add external regressors (weather, local events, holidays) to the
  forecasting model.
- Extend to a proper hierarchical/multi-series forecasting approach
  (e.g., a single global model with store/category embeddings, or
  Prophet/SARIMA per series) and compare against the current per-row
  tree-ensemble approach.
- Deploy the trained model behind a small API (FastAPI) with a scheduled
  retraining job, and log predictions vs. actuals for drift monitoring —
  the natural MLOps extension.
- Rebuild the dashboard spec as an actual Tableau/Power BI workbook once
  a live data source is available.
