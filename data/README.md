# Data

**File:** `coffee_shop_sales.csv` (149,116 rows x 11 columns)

**Source:** "Coffee Shop Sales" — Maven Analytics Data Playground, originally
published as a Kaggle notebook dataset
(kaggle.com/code/serhiimyroniuk/coffee-shop-sales). Public domain license.
Transaction-level sales records for *Maven Roasters*, a fictitious coffee
shop chain with three stores in New York City (Astoria, Hell's Kitchen,
Lower Manhattan), January 1 - June 30, 2023.

| Column | Type | Description |
|---|---|---|
| transaction_id | int | Unique line-item ID |
| transaction_date | date | Calendar date of the sale |
| transaction_time | time | Time of day of the sale |
| transaction_qty | int | Units of that product sold in the line item |
| store_id | int | Numeric store identifier |
| store_location | string | Astoria / Hell's Kitchen / Lower Manhattan |
| product_id | int | Numeric product identifier |
| unit_price | float | Retail price per unit ($) |
| product_category | string | 9 categories (Coffee, Tea, Bakery, ...) |
| product_type | string | Sub-category (e.g. "Brewed Chai Tea") |
| product_detail | string | Specific SKU (e.g. "Spicy Eye Opener Chai Lg") |

**Limitations (stated up front, not hidden):**
- No `customer_id` — this dataset cannot support customer-level personalization,
  loyalty, or churn analysis. That's why this project's ML target is *demand*
  (units sold per store/category/day), not customer behavior — Project 1
  (churn) and Project 2 (cross-sell/market-basket) already cover the
  customer-level angle for this candidate's portfolio.
- Single city (NYC), 6 months, one retail concept (coffee/bakery) — a
  restaurant chain the size of KFC/Pizza Hut/Taco Bell/Habit would have far
  more stores, geographies and menu complexity. The *methodology* (lag/rolling
  demand features, baseline vs. tree-ensemble comparison, time-based
  validation) generalizes; the specific numbers do not.
- There is no real promotion/discount flag in the data, so the A/B-test
  component of this project (see `src/experiment_design.py`) is explicitly a
  **simulation** built on top of the real revenue distribution, not a claim
  that a real promotion was run or measured.
