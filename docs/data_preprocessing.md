# AI Agent Guide: Dataset Preprocessing Pipeline

## Overview
This pipeline transforms raw financial data of Spanish SMEs (`raw_data.csv`) into a survival-analysis-ready format (`train_set.csv`, `test_set.csv`). The processed data is used for Cox Proportional Hazards models predicting business insolvency.

---

## Input: `raw_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `Unnamed: 0` | int | Row index — **removed** |
| `Year` | int | Observation year (1999–2020) — **removed in survival format** |
| `CIF` | str | Company tax ID (93,641 unique in train) |
| `FCA` | float | Year of bankruptcy/closure (69% missing if company survived) |
| `Y` | int | Event label (0=censored, 1=bankrupt in that year) — **69% missing** |
| `N1` | float | Raw numeric feature (Age?) |
| `N2` | float | Raw numeric feature (Employees?) |
| `N3` | str | Sector (Agriculture, Construction, Industry, Services) |
| `N4` | str | Legal form (Cooperative, Limited Company, Public Limited Company, Other) |
| `N5` | str | SME category (Yes/No) |
| `N6` | float | Raw numeric feature |
| `N7` | float | Raw numeric feature |
| `N8` | str | Company size (Microenterprise, Small, Medium) |
| `N9` | str | Boolean flag (Yes/No) |
| `N10` | str | Audit opinion (Favorable, Qualified, Denied, Unfavorable, Does not exist...) |
| `N11` | str | Gender (Male, Female, nan) |
| `N12` | float | Raw numeric feature |
| `N13` | float | Raw numeric feature |
| `N14` | float | Raw numeric feature |
| `N15` | str | Boolean flag (Yes/No) |
| `F16`–`F43` | float | Financial ratios (various magnitudes, some with many missing) |
| `F26`, `F32` | float | **Removed** (87% missing each) |
| `Type` | str | Insolvency type label — **removed** |

---

## Output: `train_set.csv` / `test_set.csv`

| Column Group | Count | Description |
|-------------|-------|-------------|
| `CIF` | 1 | Company ID, preserved |
| `N1`, `N2`, `N6`, `N7`, `N12`, `N13`, `N14` | 7 | Yeo-Johnson normalized (mean≈0, std≈1) |
| `N3_*` | 5 | One-hot from N3 (Agriculture, Construction, Industry, nan, Services) |
| `N4_*` | 5 | One-hot from N4 (Cooperative, Limited Company, nan, Other, Public Limited Company) |
| `N5_*` | 2 | One-hot from N5 (No, Yes) |
| `N8_*` | 3 | One-hot from N8 (Medium enterprise, Microenterprise, Small company) |
| `N9_*` | 2 | One-hot from N9 (No, Yes) |
| `N10_*` | 5 | One-hot from N10 (Denied, Does not exist..., Favorable, Qualified, Unfavorable) |
| `N11_*` | 3 | One-hot from N11 (Female, Male, nan) |
| `N15_*` | 2 | One-hot from N15 (No, Yes) |
| `F16`–`F43` (except F26, F32) | 26 | Yeo-Johnson normalized |
| `anomaly_score` | 1 | Optional anomaly score column |
| `Start`, `Stop` | 2 | Survival interval bounds (integers, years since first observation) |
| `Event` | 1 | 1=bankrupt in this interval, 0=censored |
| **Total** | **64** | |

---

## Preprocessing Pipeline (step by step)

### 1. Column Removal
Remove `Unnamed: 0`, `F26`, `F32` (F26/F32 have >87% missing).

### 2. Anomaly Score (optional)
Merge `anomaly_score` from external file (`anomaly_analysis_output/datos_con_anomalias.csv`), matched by row order.

### 3. Train/Test Split by CIF
80/20 stratified by **company** (CIF), not by row. This prevents data leakage (same company appearing in both sets).

### 4. MICE Imputation (POST-SPLIT)
Numerical columns only: `N1, N2, N6, N7, N12, N13, N14, F16–F43` (excluding F26, F32).

- **Train**: `IterativeImputer(BayesianRidge(), max_iter=3)` — fit + transform
- **Test/New**: Transform using the trained imputer (no refit)

Missing categorical values (N3, N4, N11) are handled by One-Hot Encoding producing a dedicated `_nan` column.

### 5. One-Hot Encoding
`sklearn.preprocessing.OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')`

Variables (8 → 27 columns):
- **N3**: Agriculture, Construction, Industry, nan, Services
- **N4**: Cooperative, Limited Company, nan, Other legal form, Public Limited Company
- **N5**: No, Yes
- **N8**: Medium enterprise, Microenterprise, Small company
- **N9**: No, Yes
- **N10**: Denied, Does not exist or is not audited, Favorable, Qualified, Unfavorable
- **N11**: Female, Male, nan
- **N15**: No, Yes

Unknown categories in new data produce all-zeros (per `handle_unknown='ignore'`).

### 6. Yeo-Johnson Normalization
`PowerTransformer(method='yeo-johnson', standardize=True)` — handles both positive and negative values.

Applied to: `N1, N2, N6, N7, N12, N13, N14, F16–F43` (same set as imputation).

Fit on **train only**, transform both train and test.

### 7. Censoring Logic
For each company (grouped by CIF):
- Compute `gap = FCA - last_observation_year`
- If `gap > 3`: company is **censored** (set FCA=NaN, Y=0 for all rows)
- If `gap <= 3`: company has **event** (set Y=1 in last observation year)
- Companies without FCA remain with Y=0 (censored)

### 8. Survival Format Conversion
- Fill missing Y with 0
- Remove rows after first event (cumulative sum of Y > 1)
- Compute `Start = year - first_observation_year`, `Stop = Start + 1`
- Remove temporary columns: `Year`, `FCA`, `Y`

### 9. Post-processing
- Remove `Type` column
- Reorder columns: `CIF` → `N*` (sorted) → `F*` (sorted) → `anomaly_score` (optional) → `Start`, `Stop`, `Event`

### 10. MICE Imputer Serialization
The fitted `IterativeImputer` is saved with `joblib.dump()` to `models/mice_imputer.joblib` **after** fitting on train data.

---

## Exported Artifacts (`models/`)

To apply the trained preprocessing pipeline to new data, these serialized objects must be loaded:

### Fitted Transformers (must NOT be re-fitted on new data)

| Artifact | Object | Shape/Size | Created by |
|----------|--------|------------|------------|
| `mice_imputer.joblib` | `IterativeImputer(BayesianRidge(), max_iter=3)` | 33 numerical cols | Step 4 |
| `one_hot_encoder.joblib` | `OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')` | 8 cats → 27 cols | Step 5 |
| `yeo_johnson_transformer.joblib` | `PowerTransformer(method='yeo-johnson', standardize=True)` | 33 numerical cols | Step 6 |

### Column Name References (metadata for correct column mapping)

| Artifact | Content |
|----------|---------|
| `ohe_columns.joblib` | List of 27 OHE column names (e.g. `N3_Agriculture`, `N3_Construction`, ...) |
| `numerical_columns.joblib` | List of 33 numerical column names normalized with Yeo-Johnson |
| `mice_imputation_columns.joblib` | Same list as `numerical_columns.joblib` (columns passed to MICE) |
| `expected_columns.joblib` | Ordered list of the 64 final columns (model contract) |

### Configuration

| Artifact | Content |
|----------|---------|
| `pipeline_metadata.json` | Runtime parameters: `test_size`, `random_state`, `categorical_columns`, `numerical_columns`, `initial_removed_columns`, `anomaly_path` |

### Apply Pipeline Flow (for new data inference)

1. **Load** all artifacts from `models/` using `joblib.load()`
2. **Remove columns**: same as training (`Unnamed: 0`, `F26`, `F32` from metadata)
3. **Anomaly score** (optional): match by row order
4. **MICE imputation**: `mice_imputer.transform(X_new)` — **no refit**
5. **One-Hot Encoding**: `ohe.transform(X_new)` — **no refit**, rename with `ohe_columns`
6. **Yeo-Johnson**: `pt.transform(X_new)` — **no refit**
7. **Survival format** (optional): only if outcome columns (`FCA`, `Y`) are present
8. **Column reordering**: reorder to match `expected_columns`

---

## Key Rules for AI Agents

1. **Data leakage prevention**: All transformations (imputation means, OHE categories, normalization params) must be fitted **only on training data**, then applied to test/new data.
2. **MICE imputation**: Must use Bayesian Ridge estimator with `max_iter=3` and `sample_posterior=False`.
3. **OHE unknown handling**: `handle_unknown='ignore'` — unseen categories become all-zero vectors.
4. **Survival format** is only for model training; inference may skip steps 7–8.
5. **Anomaly score** is optional and matched by row index, not by CIF.
6. **MICE imputer serialization**: The fitted `IterativeImputer` must be saved with `joblib.dump()` after training and loaded with `joblib.load()` for inference. Never re-fit on new data.
7. **Column order** must match training exactly for model compatibility. Use `expected_columns.joblib` as the reference.
