# WIDS-2025 Competition Summary

**Goal**: Predict two target variables — `ADHD_Outcome` (ADHD diagnosis) and `Sex_F` (sex, female) — using quantitative, categorical, and functional connectome data.

---

## 🧩 Feature Selection & Engineering

### 📊 Data Sources

- **Quantitative Data**:  
  - `train_q` and `test_q` (Excel files)  
  - Numerical features: EHQ scores, APQ metrics, SDQ scores, MRI scan age

- **Categorical Data**:  
  - `train_c` and `test_c` (Excel files)  
  - Features: ethnicity, race, parental education/occupation

- **Functional Connectome Data**:  
  - `train_f` and `test_f` (CSV files)  
  - Pearson correlation matrices for brain connectivity

- **Labels**:  
  - `TRAINING_SOLUTIONS.xlsx` with `ADHD_Outcome` and `Sex_F`

---

### 🧹 Missing Value Handling

- **Identified NaNs**:
  - `train_q`: 18 columns (e.g., `MRI_Track_Age_at_Scan` has 360 missing)
  - `train_c`: 7 columns (e.g., `Barratt_Barratt_P2_Occ` has 222 missing)
  - `test_q`: 17 columns
  - `test_c`: 6 columns
  - `train_f`, `test_f`: **No missing values**

- **Imputation**:
  - Median imputation for numerical columns
  - Imputation values derived from training set (avoids data leakage)

---

### 🧠 Feature Selection

- Selected **top 200 correlated features** (excluding targets) for both `Sex_F` and `ADHD_Outcome`
- Excluded target columns to avoid leakage

#### 🔑 Key Features for ADHD Modeling:
- `APQ_P_APQ_P_INV`
- `APQ_P_APQ_P_PP`
- `SDQ_SDQ_Hyperactivity`
- `MRI_Track_Age_at_Scan`
- `SDQ_SDQ_Generating_Impact`

- **Engineered Features**:
  - Predicted `sex_proba` (probability of Sex_F) added as a feature
  - Created interaction terms: `I_{col} = col * sex_proba`

- **Combined Data**:
  - Merged quantitative, categorical, and functional data into `train_combined` and `test_combined`

- **Stratification**:
  - Stratified by combination of `ADHD_Outcome` and `Sex_F` to mitigate class imbalance

---

## 🧪 Training Methods

### 🎯 Sex_F Prediction

**Model**: `CatBoostClassifier`  
**Parameters**:
- `iterations=500`
- `learning_rate=0.05`
- `depth=6`
- `scale_pos_weight=1.5`
- `random_state=42`

**Training**:
- Used `features_sex` (top 200 correlated features)
- Applied sample weights for class imbalance

---

### 🎯 ADHD_Outcome Prediction

**Model 1**: `Calibrated RandomForestClassifier`

- **Base Model**: `RandomForestClassifier`
  - `n_estimators=200`, `max_depth=10`, `class_weight='balanced'`, `random_state=42`
- **Calibration**: Sigmoid, 3-fold cross-validation
- **Training Features**: `features_adhd` + `sex_proba` + interaction features

**Model 2**: `H2O AutoML`

- Explored: GBM, GLM, DRF, XGBoost, Stacked Ensembles
- **Best Model**: `GBM_2`
- **Performance**: AUC = 0.54108 (weak performance overall)

**Evaluation Metrics**:
- AUC, log loss, AUC-PR, mean per-class error, RMSE, MSE

---

## 🎯 Threshold Tuning

### Sex_F:
- Searched thresholds: **10th–90th percentile** of OOF (`sex_oof`) predictions
- Optimized using **F1 score** with sample weights

### ADHD_Outcome:
- Searched thresholds: **0.3–0.7**
- Optimized for **F1 score** with sample weights

---

## ⚖️ Sample Weights

- Used `compute_sample_weight` with `'balanced'` strategy
- Special weight for samples where:
  - `ADHD_Outcome = 1` and `Sex_F = 1` ⇒ **double weight**

---

## 📈 Evaluation

- **Metric**: Weighted F1 score
- Used **OOF predictions** (`sex_oof`, `adhd_oof`) for tuning and validation
- Applied **statistical tests**:
  - Kolmogorov-Smirnov (KS)
  - Mann-Whitney U (MWU)
- Compared:
  - OOF vs Test prediction distributions
  - Positive prediction shares between OOF and Test sets

---

## 🔍 Important Findings

### 🚨 Data Issues

- **Missing Values**:
  - Especially high in `MRI_Track_Age_at_Scan`
- **Connectome Data**:
  - No missing values — most reliable source

- **AutoML Performance**:
  - Weak AUC (~0.5) — indicates low feature informativeness or class imbalance

---

### 🔍 Model Performance

- **Sex_F**:
  - Best threshold and F1 score found (exact value not shown)
- **ADHD_Outcome**:
  - Optimal threshold ~0.3–0.7
  - Moderate performance (exact F1 not provided)
- **AutoML**: AUC ~0.5 across models

---

### ⭐ Feature Importance

- Top 15 features from RandomForest used for ADHD_Outcome (specific features not listed)
- `sex_proba` and interaction features played a significant role

---

## 📝 Submission

- Created `submit.csv` with binary predictions for:
  - `ADHD_Outcome`
  - `Sex_F`
- Used tuned thresholds
- File downloaded for final competition submission
"""
