# Real-Time Fraud Detection System

## 1. Problem Statement
Financial fraud is an increasingly sophisticated and pervasive threat across global payment networks, banking systems, and e-commerce platforms. As transaction volumes surge and digital payment rails shift toward instant settlement, legacy rule-based detection systems fall short due to high false-positive rates, inability to detect novel fraud vectors, and excessive operational latency.

Modern fraud mitigation demands a high-performance machine learning system capable of evaluating transactions in real time with low latency. Critically, the system must balance high detection sensitivity (recall) with low false alarms (precision), while providing interpretable explanations so risk officers and fraud analysts can understand and audit automated decisions.

---

## 2. Project Objective
The goal of this project is to architect, develop, and deploy an end-to-end, production-grade **Real-Time Fraud Detection System**.

When fully developed, the system will deliver:
1. **Fraud Probability**: Calibrated likelihood score representing the probability that an incoming transaction is fraudulent.
2. **Risk Categorization**: Dynamic risk tiering (e.g., *Low*, *Medium*, *High*) to drive downstream routing (auto-approve, two-factor challenge, or manual review).
3. **Prediction Explainability**: Local feature attribution (via SHAP / TreeSHAP) detailing the primary signals driving each specific prediction.
4. **Real-Time Serving API**: Low-latency RESTful API endpoint for scoring incoming financial transactions synchronously.
5. **Interactive Operations Dashboard**: Live web-based monitoring interface for transaction telemetry, risk trends, and model decision inspections.

> **Project Status (Phase 11 - Completed)**: Additive SQLite persistence layer and audit history system implemented with SQLAlchemy (`api/database.py`, `api/models.py`, `api/repositories.py`, `data/fraud_detection.db`). Persists real-time transaction scoring and exact SHAP feature attributions without modifying the ML pipeline or inference contract. Exposes REST endpoints (`GET /transactions`, `GET /transactions/{transaction_id}`, `GET /stats`) with dynamic SQL aggregation and duplicate transaction ID handling. Streamlit dashboard updated with a dual-tab experience incorporating real-time scoring, aggregate database telemetry, and an interactive SHAP audit inspector. Comprehensive test suite (`tests/`) expanded to 31 tests with 100% pass rate.


---

## 3. Dataset & Data Understanding
The project uses the standard benchmark **Credit Card Fraud Detection** dataset comprising actual card transactions made by European cardholders.

- **Dataset Name**: Credit Card Fraud Detection
- **Source**: Machine Learning Group (MLG) - ULB / Hosted on Google Cloud Storage Public Datasets
- **Number of Rows (Transactions)**: 284,807
- **Number of Features**: 31 (30 input features + 1 target variable)
  - `Time`: Seconds elapsed between this transaction and the first transaction in the dataset.
  - `V1` through `V28`: Anonymized principal components resulting from PCA transformation for privacy.
  - `Amount`: Transaction monetary amount.
- **Target Column**: `Class`
  - `0`: Legitimate transaction
  - `1`: Fraudulent transaction
- **Number of Legitimate Cases**: 284,315 (99.8273%)
- **Number of Fraud Cases**: 492 (0.1727%)
- **Class Imbalance**: Extreme class imbalance with a ratio of **~1:578** (only 0.1727% fraud incidence). Exactly 1 fraudulent transaction occurs for every ~578 legitimate transactions.
- **Missing Values**: 0 missing values across all columns.
- **Duplicates**: 1,081 duplicate rows identified in raw data (preserved without modification per project guidelines).
- **Exploration Notebook**: Documented and visualized in [`notebooks/01_data_understanding.ipynb`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/notebooks/01_data_understanding.ipynb).

---

## 4. Exploratory Data Analysis (EDA)
In-depth pattern discovery and statistical profiling are documented in [`notebooks/02_eda.ipynb`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/notebooks/02_eda.ipynb).

### Major Findings Discovered:
1. **Transaction Amount Bimodal Behavior**:
   - Legitimate transactions have a median of $\$22.00$ and mean of $\$88.29$.
   - Fraudulent transactions exhibit a smaller median of $\$9.25$ (with $25\%$ of frauds being $\le \$1.00$, indicating card-testing micro-transactions) alongside a higher mean of $\$122.21$ (due to high-value fraudulent attempts up to $\$2,125.87$).
2. **Key Discriminative PCA Features**:
   - **Strong Negative Linear Separation**: `V17` ($r = -0.33$), `V14` ($r = -0.30$), `V12` ($r = -0.26$), and `V10` ($r = -0.22$). Fraud cases have marked negative distribution shifts (e.g., `V14` median is $-4.49$ for fraud vs $+0.04$ for legitimate).
   - **Strong Positive Linear Separation**: `V11` ($r = +0.15$) and `V4` ($r = +0.13$). Fraud cases show higher positive medians.
3. **Diurnal Time Dynamics**:
   - Total transaction volume plummets during nighttime hours (01:00 to 06:00).
   - However, fraud attempts remain steady during nighttime, causing the **hourly fraud rate to peak between 02:00 AM and 04:00 AM (reaching $1.0\%$ to $1.7\%$)**, compared to daytime fraud rates of $\approx 0.1\%$.
4. **Outlier Paradox & Danger of Blind Removal**:
   - Over **$46.75\%$** of all fraudulent transactions in `V14` and **$28.25\%$** in `V17` fall into the statistical outlier tail according to the standard IQR threshold ($Q_1 - 1.5 \times \text{IQR}$).
   - **Conclusion**: Outlier removal algorithms must NOT be applied to this dataset, as the anomalies themselves represent the fraudulent behavior to be detected.
5. **Zero Feature Multicollinearity Among V-Features**:
   - Due to the orthogonal properties of PCA, mutual correlations between $V_1$ through $V_{28}$ are zero, preventing multicollinearity issues in linear baseline models.

---

## 5. Data Preprocessing & Feature Engineering
Full preprocessing pipeline, split validations, and transformation routines are implemented in [`src/preprocessing.py`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/src/preprocessing.py) and documented in [`notebooks/03_preprocessing.ipynb`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/notebooks/03_preprocessing.ipynb).

- **Data Cleaning**: Verified zero null/missing values across all records. Duplicate transactions (1,081 rows) and statistical outliers are intentionally preserved to maintain empirical fraud patterns and transaction auditability.
- **Stratified Train/Test Split**: 80/20 train/test split (`random_state=42`, `stratify=y`):
  - **Training Set (80%)**: 227,845 transactions (227,451 legitimate, 394 fraudulent; 0.1729% fraud rate).
  - **Test Set (20%)**: 56,962 transactions (56,864 legitimate, 98 fraudulent; 0.1720% fraud rate).
- **Engineered Features**:
  1. `hour`: Integer hour of day $[0, 23]$ capturing circadian transaction volume and late-night risk spikes.
  2. `hour_sin` & `hour_cos`: Cyclical continuous coordinates ($\sin(2\pi \cdot \text{hour}/24)$ and $\cos(2\pi \cdot \text{hour}/24)$) eliminating boundary discontinuity between 23:00 and 00:00.
  3. `log_amount`: Log-transformed amount ($\log(1 + \text{Amount})$) stabilizing high variance and extreme positive skewness.
  4. `scaled_amount`: Scaled via `RobustScaler` (median subtraction and IQR division) for outlier resilience.
  5. `scaled_time`: Scaled via `RobustScaler` for normalized temporal progression.
  - **Final Processed Feature Count**: 34 features ($V_1$–$V_{28}$ + 6 engineered/scaled features).
- **Feature Scaling Strategy**: Features $V_1$ to $V_{28}$ are retained without re-scaling because PCA components are already zero-centered with standard unit variance. `RobustScaler` was selected for `Amount` and `Time` because standard mean/variance scalers are distorted by heavy financial outliers.
- **Leakage Prevention**: All scaler parameters ($\text{Median}_{\text{train}} = \$22.00, \text{IQR}_{\text{train}} = \$71.42$) are learned strictly from `X_train`. The test set is transformed strictly using the training-fitted preprocessor, ensuring zero test information leaks into the modeling pipeline.
- **Serialized Artifacts**:
  - Fitted preprocessor: [`models/preprocessor.joblib`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/models/preprocessor.joblib)
  - Processed training split: `data/processed/train.csv`
  - Processed test split: `data/processed/test.csv`

---

## 6. Baseline Modeling
Initial linear and non-linear baseline models have been trained and evaluated strictly on the test set. Detailed analysis is documented in [`notebooks/04_baseline_models.ipynb`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/notebooks/04_baseline_models.ipynb).

### Models Trained:
1. **Balanced Logistic Regression**: Linear baseline with `class_weight='balanced'`, `max_iter=1000`, `random_state=42`.
2. **Balanced Random Forest**: Tree ensemble baseline with `class_weight='balanced'`, `n_estimators=100`, `max_depth=10`, `random_state=42`.

### Measured Test Set Results ($N_{\text{test}} = 56,962, N_{\text{fraud}} = 98$):
| Model | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | `0.9739` | `0.0574` | **`0.9184`** | `0.1080` | `0.7220` | `0.9738` | `1,479` | **`8`** |
| **Random Forest (Balanced)** | **`0.9990`** | **`0.6640`** | `0.8469` | **`0.7444`** | **`0.8338`** | **`0.9795`** | **`42`** | `15` |

### Key Methodological Findings & Tradeoffs:
- **The Accuracy Paradox**: Both models achieve $>97\%$ accuracy, but accuracy is an uninformative metric: a dummy model predicting zero fraud achieves $99.83\%$ accuracy while catching no fraudulent activity.
- **ROC-AUC vs. PR-AUC Disparity**: Both models score nearly identical ROC-AUC values ($0.9738$ vs $0.9795$) because the huge legitimate test class ($56,864$) minimizes the False Positive Rate ($\text{FPR} < 2.6\%$). Conversely, PR-AUC evaluates precision strictly against the minority class, revealing Random Forest's clear superiority ($0.8338$ vs $0.7220$).
- **Precision vs. Recall Tradeoff**:
  - **Logistic Regression** prioritizes Recall ($91.84\%$, catching 90/98 frauds), but produces $1,479$ false alarms ($5.74\%$ precision), which would overwhelm fraud review operations.
  - **Random Forest** dramatically cuts false alarms to $42$ ($66.40\%$ precision) while maintaining strong Recall ($84.69\%$, catching 83/98 frauds).
- **Serialized Baseline Artifacts**:
  - Logistic Regression model: [`models/logistic_regression_baseline.joblib`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/models/logistic_regression_baseline.joblib)
  - Random Forest model: [`models/random_forest_baseline.joblib`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/models/random_forest_baseline.joblib)
  - Training metadata: [`models/baseline_metadata.json`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/models/baseline_metadata.json)

---

## 7. Imbalanced Learning & Threshold Optimization
Techniques for mitigating extreme class imbalance ($0.17\%$ fraud) and operational threshold tuning are documented in [`notebooks/05_imbalanced_learning.ipynb`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/notebooks/05_imbalanced_learning.ipynb).

### 1. Data Partitioning Hierarchy & Leakage Prevention:
To prevent leakage during hyperparameter and threshold selection, the Phase 4 training portion was subdivided into:
- **Training Subset (64% of total data = 182,276 records, 315 frauds)**: Used for fitting classifiers and SMOTE.
- **Validation Subset (16% of total data = 45,569 records, 79 frauds)**: Used strictly for algorithm comparison and decision threshold calibration.
- **Test Set (20% held-out = 56,962 records, 98 frauds)**: Kept completely untouched until final evaluation.

### 2. Algorithmic Class Weighting vs. SMOTE (Validation Set Results):
| Approach | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Class-Weight)** | **`0.7949`** | `0.7848` | **`0.7898`** | **`0.7649`** | `0.9658` | **`16`** | `17` |
| **Random Forest (SMOTE)** | `0.5478` | **`0.7975`** | `0.6495` | `0.7505` | **`0.9814`** | `52` | **`16`** |
| **Logistic Regression (Class-Weight)** | `0.0578` | `0.8861` | `0.1085` | `0.6807` | `0.9771` | `1,141` | `9` |
| **Logistic Regression (SMOTE)** | `0.0558` | `0.8608` | `0.1048` | `0.6714` | `0.9737` | `1,151` | `11` |

- **Why SMOTE is Applied ONLY to Training Data**: Applying SMOTE to validation or test data would synthesize synthetic fraud points that do not exist in reality, invalidating evaluation on empirical distributions.
- **Finding**: SMOTE degraded Random Forest precision from $79.49\%$ to $54.78\%$ by more than tripling false alarms ($16 \rightarrow 52$) while catching only 1 additional fraud ($62 \rightarrow 63$). Algorithmic class weighting proved vastly superior in both fidelity and computation time.

### 3. Decision Threshold Optimization (Random Forest on Validation Set):
By sweeping the decision threshold $\tau \in [0.10, 0.90]$, we directly control the business tradeoff:
- **Low Threshold ($\tau = 0.20$)**: High recall ($83.54\%$, $66/79$ frauds caught), but $151$ false alarms ($30.41\%$ precision).
- **Selected Calibrated Threshold ($\tau = 0.40$)**: Optimal operating point catching $79.75\%$ of validation frauds ($63/79$) with $74.12\%$ precision and only $22$ false alarms ($F_1 = 0.7683$).
- **High Threshold ($\tau = 0.60$)**: High precision ($88.24\%$, only $8$ false alarms), but misses $19$ fraudulent transactions ($75.95\%$ recall).

### 4. Final Evaluation on Untouched Test Set ($N_{\text{test}} = 56,962, N_{\text{fraud}} = 98$):
Evaluating the calibrated Random Forest model at the chosen threshold $\tau = 0.40$ on unseen test data:
- **Precision**: `64.34%`
- **Recall**: `84.69%` (Caught $83$ out of $98$ frauds)
- **F1-Score**: `0.7313`
- **PR-AUC**: `0.8223`
- **ROC-AUC**: `0.9815`
- **False Positives (FP)**: `46` (0.08% false alarm rate across 56,864 legitimate transactions)
- **False Negatives (FN)**: `15`

### 5. Serialized Imbalanced Learning Artifacts:
- Calibrated Random Forest model: [`models/rf_calibrated_threshold_model.joblib`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/models/rf_calibrated_threshold_model.joblib)
- Decision threshold config: [`models/threshold_config.json`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/models/threshold_config.json)

---

## 8. Advanced Modeling (XGBoost)
Gradient boosted decision trees via XGBoost were trained, tuned, calibrated, and evaluated. Full experimental workflows, validation curves, calibration plots, and single-evaluation test metrics are documented in [`notebooks/06_advanced_model.ipynb`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/notebooks/06_advanced_model.ipynb).

### 1. Controlled Class Imbalance via `scale_pos_weight`:
- The raw class ratio in the training subset is $\frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{181,961}{315} \approx 577.65$.
- **Baseline XGBoost (`scale_pos_weight=577.65`)**:
  - While boosting with the full reciprocal ratio provides strong recall ($78.48\%$), it forces tree nodes to over-weight positive misclassifications, leading to suboptimal probability calibration and slight inflation of false alarms ($7$ FP, Precision $89.86\%$).
- **Controlled Weighting Discovery (`scale_pos_weight=5.0`)**:
  - Moderating `scale_pos_weight` to $5.0$ provides the gradient booster with adequate positive gradient emphasis while preserving sharp probability discrimination. Combined with decision threshold tuning, this eliminates oversensitivity and maximizes the area under the Precision-Recall curve (PR-AUC).

### 2. Hyperparameter Tuning (Validation Set Optimization):
Tuning was executed strictly on the training subset ($182,276$ rows) and evaluated against the validation subset ($45,569$ rows):
- **Architecture**: `max_depth=5`, `n_estimators=150`, `learning_rate=0.08`
- **Regularization & Stochasticity**: `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=5.0`
- **Objective / Metric**: `binary:logistic`, evaluated via `logloss` and PR-AUC.
- **Validation Metric Performance**:
  - **PR-AUC**: `0.8287`
  - **ROC-AUC**: `0.9736`
  - **Brier Score**: `0.000460`

### 3. Probability Calibration & Threshold Optimization:
- **Probability Calibration**: Evaluated using reliability curves and the Brier score. The tuned XGBoost model achieved an exceptionally low Brier score ($0.000460$ on validation, $0.000374$ on test), demonstrating that predicted probabilities are well-calibrated and reliably reflect true risk likelihoods without requiring secondary Isotonic/Platt calibration.
- **Threshold Selection on Validation Set**:
  - $\tau = 0.50$ (Default): Precision `95.08%`, Recall `73.42%`, F1 `0.8286`, FP `3`, FN `21`.
  - **$\tau = 0.35$ (Selected Optimal Threshold)**: Precision `92.54%`, Recall `78.48%`, F1 `0.8493`, FP `5`, FN `17`.
  - Rationale: Shifting the decision threshold from $0.50$ to $0.35$ reclaims $4$ additional fraud cases ($62$ caught vs $58$) with only $2$ additional false alarms, achieving the peak validation F1-score of $0.8493$.

### 4. Single Evaluation on Untouched Test Set ($N_{\text{test}} = 56,962, N_{\text{fraud}} = 98$):
Evaluating the tuned model at $\tau = 0.35$ on the unseen test set:
- **PR-AUC**: **`0.8749`** (substantial improvement over Random Forest's `0.8338` and Logistic Regression's `0.7220`)
- **Precision**: **`90.22%`** (only **9** false alarms across 56,864 legitimate transactions)
- **Recall**: **`84.69%`** (**83** out of 98 fraud cases detected)
- **F1-Score**: **`0.8737`**
- **ROC-AUC**: `0.9765`
- **Brier Score**: **`0.000374`**
- **Test Confusion Matrix**:
  - True Negatives (TN): `56,855`
  - False Positives (FP): `9` (False Alarm Rate: $0.0158\%$)
  - False Negatives (FN): `15`
  - True Positives (TP): `83`

### 5. Multi-Model Performance Benchmark:
| Model / Configuration | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Test FP | Test FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced, $\tau=0.50$)** | `0.0574` | **`0.9184`** | `0.1080` | `0.7220` | `0.9738` | `1,479` | **`8`** |
| **Random Forest (Balanced, $\tau=0.50$)** | `0.6640` | `0.8469` | `0.7444` | `0.8338` | `0.9795` | `42` | `15` |
| **Random Forest (Calibrated, $\tau=0.40$)** | `0.6434` | `0.8469` | `0.7313` | `0.8223` | **`0.9815`** | `46` | `15` |
| **Tuned XGBoost (Selected, $\tau=0.35$)** | **`0.9022`** | `0.8469` | **`0.8737`** | **`0.8749`** | `0.9765` | **`9`** | `15` |

**Takeaway**: Tuned XGBoost matches the highest ensemble recall ($84.69\%$) while reducing false positive alerts by **$78.6\%$** compared to Random Forest ($9$ vs $42$) and by **$99.4\%$** compared to Logistic Regression ($9$ vs $1,479$).

### 6. Serialized Phase 7 Artifacts:
- Tuned XGBoost Model: [`models/xgboost_advanced_model.joblib`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/models/xgboost_advanced_model.joblib)
- Threshold & Hyperparameter Config: [`models/xgboost_threshold_config.json`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/models/xgboost_threshold_config.json)

---

## 9. Explainability & Interpretability (SHAP)
Fraud detection systems operate in highly regulated environments where automated decisions directly impact cardholders and merchants. Black-box predictions are unacceptable: risk analysts, fraud investigators, and audit compliance officers require transparent explanations detailing why an individual transaction was approved, challenged, or declined.

Detailed experiments, visualizations, and case studies are documented in [`notebooks/07_shap_explainability.ipynb`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/notebooks/07_shap_explainability.ipynb) and implemented for production in [`src/explainability.py`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/src/explainability.py).

### 1. What is SHAP & TreeSHAP?
**SHAP (SHapley Additive exPlanations)** is an axiomatic game-theoretic approach (Lloyd Shapley, 1953) to explaining machine learning predictions by treating each feature as a player in a cooperative game and computing its marginal contribution across all possible coalitions.

SHAP satisfies four fundamental mathematical axioms:
- **Efficiency / Additivity**: Feature attributions sum exactly to the difference between model output and expected baseline.
- **Symmetry**: Two features with identical marginal contributions across all subsets receive identical SHAP values.
- **Dummy / Null Player**: A feature with zero marginal contribution receives zero attribution.
- **Monotonicity / Consistency**: If a model changes such that a feature's marginal contribution increases or stays the same, its SHAP value never decreases.

For gradient-boosted decision trees, **TreeSHAP** computes exact Shapley values in polynomial time ($O(T L D^2)$) rather than exponential time. For our binary XGBoost classifier, TreeSHAP operates in **margin (log-odds) space**, converted to probabilities via the logistic sigmoid:
$$\text{margin}(x) = \text{base\_value} + \sum_{i=1}^M \phi_i(x)$$
$$P(\text{fraud}) = \sigma(\text{margin}(x)) = \frac{1}{1 + e^{-\text{margin}(x)}}$$
- **Base Value (Expected Margin)**: $-5.9109$
- **Base Probability**: $\sigma(-5.9109) = 0.002702$ ($0.2702\%$, closely reflecting prior fraud prevalence in training data).

### 2. Global vs. Local Explanations
- **Global Explanations**: Measure overall feature impact across the entire transaction population ($N = 56,962$ test transactions) by computing the mean absolute SHAP value:
  $$\text{Global Importance}(X_j) = \frac{1}{N} \sum_{i=1}^N |\phi_{i,j}|$$
- **Local Explanations**: Decompose a single transaction's prediction into positive pushes ($+$, increased fraud risk) and negative pushes ($-$, decreased fraud risk) leading to the final probability.
- **Relative Attribution Share**: Normalized percentage share of total attribution:
  $$\text{relative\_attribution\_share}_i = \frac{|\phi_i|}{\sum_{j=1}^M |\phi_j|} \times 100\%$$

### 3. Measured Global Feature Importance (Entire Test Set, $N = 56,962$)
| Rank | Feature | Mean \|SHAP\| | Feature Type | Operational Impact / Directionality |
| :---: | :--- | :---: | :--- | :--- |
| **1** | `V4` | **`1.1044`** | PCA Component | Strong positive values elevate fraud probability |
| **2** | `V14` | **`0.8722`** | PCA Component | Severe negative values ($< -5.0$) add $+5$ to $+8$ log-odds toward fraud |
| **3** | `V8` | **`0.3232`** | PCA Component | Discriminates legitimate clusters from anomalous spikes |
| **4** | `V12` | **`0.3157`** | PCA Component | Low values strongly correlate with increased fraud risk |
| **5** | `V10` | **`0.3142`** | PCA Component | Negative distribution shifts push prediction toward fraud |
| **6** | `V11` | **`0.2310`** | PCA Component | Positive values increase fraud risk |
| **7** | `V26` | **`0.1928`** | PCA Component | Fine-grained decision surface refinement |
| **8** | `V19` | **`0.1896`** | PCA Component | Directional shift in high-risk transactions |
| **9** | `V13` | **`0.1855`** | PCA Component | Intermediate separation component |
| **10** | `V3` | **`0.1627`** | PCA Component | Negative values increase fraud risk |
| ... | ... | ... | ... | ... |
| **14** | `scaled_amount` | **`0.1314`** | Engineered | Robust-scaled monetary amount; large anomalous amounts elevate risk |
| **20** | `scaled_time` | **`0.0947`** | Engineered | Robust-scaled transaction sequence time |
| **21** | `hour` | **`0.0904`** | Engineered | Circadian hour; captures late-night fraud rate spikes (02:00-04:00 AM) |
| **25** | `hour_sin` | **`0.0647`** | Engineered | Continuous cyclical hour coordinate |

Visual summaries saved:
- Global Importance Bar Chart: [`reports/figures/shap_global_importance.png`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/reports/figures/shap_global_importance.png)
- Beeswarm Summary Plot: [`reports/figures/shap_beeswarm_summary.png`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/reports/figures/shap_beeswarm_summary.png)

### 4. Representative Local Operational Case Studies
We evaluated four deterministic test cases representing all quadrants of the confusion matrix at calibrated threshold $\tau = 0.35$:

| Case Type | Test Index | Actual Ground Truth | Model $P(\text{fraud})$ | Decision | Risk Level | Top Attributing Feature | Waterfall Figure |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **True Positive (TP)** | `840` | Fraud (`1`) | **`0.9926`** | `FRAUD` | `HIGH` | $V_{14} = -6.17$ ($+5.52$ SHAP, $39.36\%$ share) | [`shap_waterfall_tp.png`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/reports/figures/shap_waterfall_tp.png) |
| **False Positive (FP)** | `165` | Legitimate (`0`) | **`0.8867`** | `FRAUD` | `HIGH` | $V_{17} = -6.20$ ($+3.42$ SHAP, $27.87\%$ share) | [`shap_waterfall_fp.png`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/reports/figures/shap_waterfall_fp.png) |
| **False Negative (FN)** | `9179` | Fraud (`1`) | **`0.3162`** | `LEGITIMATE` | `MEDIUM` | $V_{14} = -8.64$ ($+4.75$ SHAP, $28.35\%$ share) | [`shap_waterfall_fn.png`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/reports/figures/shap_waterfall_fn.png) |
| **True Negative (TN)** | `0` | Legitimate (`0`) | **`0.000039`** | `LEGITIMATE` | `LOW` | $V_4 = -1.33$ ($-2.06$ SHAP, $34.98\%$ share) | [`shap_waterfall_tn.png`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/reports/figures/shap_waterfall_tn.png) |

#### Operational Insights from Case Studies:
- **True Positive (Index 840)**: Strong multi-feature alignment ($V_{14} = -6.17$, $V_{17} = -6.54$, $V_{10} = -4.88$) pushed margin from $-5.91$ up to $+4.91$, resulting in near-certain fraud detection ($99.26\%$).
- **False Positive (Index 165)**: The legitimate cardholder exhibited an atypical transaction profile where $V_{17} = -6.20$ and $V_{14} = -2.43$ mimicked fraudulent PCA projections, overpowering the dampening effect of $V_8$ ($-0.80$ SHAP). This provides fraud analysts with exact criteria for manual dispute review.
- **Borderline False Negative (Index 9179)**: Although positive indicators ($V_{14} = -8.64$, $V_{17} = -13.68$) pushed risk upward, conflicting signals from $V_{28}$ ($-0.85$ SHAP) kept the probability at $31.62\%$, just under $\tau = 0.35$. Crucially, our tiered risk engine assigns this transaction **`MEDIUM`** risk ($> 10\%$), triggering step-up multi-factor authentication rather than an automated pass.
- **True Negative (Index 0)**: Normal baseline PCA coordinates ($V_4 = -1.33$, $V_{14} = 0.27$) drove margin down to $-10.16$, yielding near-zero fraud risk ($0.0039\%$).

### 5. Methodological Limits of Interpreting Anonymized Features
- **PCA Anonymization Boundary**: Features $V_1$ through $V_{28}$ are anonymized principal components generated via PCA to preserve cardholder privacy. We strictly avoid inventing fictitious semantic meanings (e.g. claiming $V_{14}$ represents "merchant MCC" or "card velocity").
- **Statistical Interpretation**: We interpret $V$-features strictly through their mathematical distribution shifts, correlations with fraud labels, and tree split directions.
- **Engineered Operational Features**: Features like `scaled_amount`, `log_amount`, `hour`, and cyclical trigonometric terms (`hour_sin`, `hour_cos`) carry explicit physical and business semantics, providing domain-interpretable risk rationale.

### 6. Serialized Explainability Artifacts:
- Reusable explainer class: [`src/explainability.py`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/src/explainability.py)
- Global feature importance: [`models/global_feature_importance.json`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/models/global_feature_importance.json)
- Explainer metadata & summary: [`models/shap_summary_metadata.json`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/models/shap_summary_metadata.json)
- Visual figures directory: [`reports/figures/`](file:///C:/Users/ssmja/.gemini/antigravity/scratch/fraud-detection-system/reports/figures/)

---

## 10. Real-Time Serving API (FastAPI)
The machine learning pipeline and explainability engine are exposed as a low-latency, production-style REST API using **FastAPI** and **Uvicorn**.

### 1. Architecture & Design Principles:
- **Lifespan Context Manager**: Model weights, robust scalers, and the `FraudExplainer` singleton are loaded once during application startup. No disk I/O occurs within request handlers.
- **Raw Transaction Intake**: Clients send only raw transaction fields (`Time`, `Amount`, `V1`–`V28`). Feature engineering (`scaled_amount`, `log_amount`, `hour`, `hour_sin`, `hour_cos`, `scaled_time`) occurs strictly inside the service layer via `TransactionPreprocessor`, ensuring perfect parity with model training.
- **Dynamic Threshold Evaluation**: Operating decision threshold ($\tau = 0.35$) is loaded dynamically from `models/xgboost_threshold_config.json`, decoupling business rules from code.
- **Strict Data Contracts**: Pydantic v2 schemas enforce type safety, non-negative amounts, and reject unknown extra attributes.
- **Self-Documenting OpenAPI**: Comprehensive Swagger UI and ReDoc specifications generated automatically.

### 2. Available Endpoints:
| Method | Path | Summary | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Root Overview | Service status, version, and links to documentation |
| `GET` | `/health` | Health Check | Validates model readiness, algorithm name, and operational threshold |
| `POST` | `/predict` | Transaction Scoring | Evaluates transaction, computes fraud probability, applies threshold, generates top-5 SHAP feature explanations, and persists prediction to SQLite |
| `GET` | `/transactions` | Transaction History | Returns paginated list of historically scored transactions ordered newest first (`limit`, `offset`) |
| `GET` | `/transactions/{transaction_id}` | Transaction Audit | Retrieves prediction record and saved SHAP explanations by client transaction ID (returns latest if duplicates exist) |
| `GET` | `/stats` | Aggregated Telemetry | Computes real-time evaluation counts (total, fraud, legitimate) and live fraud rate (%) across recorded transactions |
| `GET` | `/docs` | Swagger UI | Interactive API documentation and exploratory testing interface |
| `GET` | `/redoc` | ReDoc Documentation | Detailed technical schema specification |


### 3. Running the API:
Start the server using `uvicorn` from the repository root:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Access the interactive Swagger UI at: `http://localhost:8000/docs`

### 4. Sample Request & Response:

#### Sample Request (`POST /predict`):
```json
{
  "transaction_id": "tx-406-live",
  "Time": 406.0,
  "Amount": 67.88,
  "V1": -2.3122265, "V2": 1.9519920, "V3": -1.6098507, "V4": 3.9979056,
  "V5": -0.5221879, "V6": -1.4265453, "V7": -2.5373873, "V8": 1.3916572,
  "V9": -2.7700893, "V10": -2.7722721, "V11": 3.2020332, "V12": -2.8999074,
  "V13": -0.5952219, "V14": -4.2892538, "V15": 0.3897241, "V16": -1.1407472,
  "V17": -2.8300557, "V18": -0.0168225, "V19": 0.4169557, "V20": 0.1269106,
  "V21": 0.5172324, "V22": -0.0350494, "V23": -0.4652111, "V24": 0.3201982,
  "V25": 0.0445192, "V26": 0.1778398, "V27": 0.2611450, "V28": -0.1432759
}
```

#### Sample Response (`200 OK`):
```json
{
  "transaction_id": "tx-406-live",
  "fraud_probability": 0.982961,
  "is_fraud": true,
  "decision": "FRAUD",
  "threshold": 0.35,
  "risk_level": "HIGH",
  "base_value": -5.910916,
  "margin": 4.055085,
  "top_contributing_features": [
    {
      "feature": "V14",
      "feature_value": -4.289254,
      "shap_value": 5.582695,
      "direction": "increases_risk",
      "relative_attribution_share": 38.62
    },
    {
      "feature": "V17",
      "feature_value": -2.830056,
      "shap_value": 1.87005,
      "direction": "increases_risk",
      "relative_attribution_share": 12.94
    },
    {
      "feature": "V4",
      "feature_value": 3.997906,
      "shap_value": 1.211644,
      "direction": "increases_risk",
      "relative_attribution_share": 8.38
    },
    {
      "feature": "V10",
      "feature_value": -2.772272,
      "shap_value": 1.200179,
      "direction": "increases_risk",
      "relative_attribution_share": 8.3
    },
    {
      "feature": "hour",
      "feature_value": 0.0,
      "shap_value": -0.797528,
      "direction": "decreases_risk",
      "relative_attribution_share": 5.52
    }
  ],
  "latency_ms": 49.239,
  "timestamp": "2026-09-22T18:20:02.291386+00:00"
}
```

---

## 11. Interactive Monitoring Dashboard (Streamlit)
The frontend user interface is built with **Streamlit** (`dashboard/app.py`), serving as a real-time operations console and risk monitoring telemetry dashboard for fraud analysts and system operators.

### 1. Architectural Separation (Client-Server Decoupling):
- **Zero Frontend ML Dependencies**: The dashboard never imports `joblib`, never loads model weights, and never instantiates the preprocessor or SHAP explainer.
- **REST Communication**: Streamlit acts strictly as an HTTP client communicating with the FastAPI microservice (`GET /health` and `POST /predict`) via `dashboard/services/api_client.py`.
- **Fault-Tolerant Offline Handling**: If FastAPI is offline or restarting, the dashboard displays a non-blocking warning badge and instructions rather than crashing, updating automatically once connectivity is restored.

### 2. Dashboard Features:
1. **System Health & Configuration Banner**:
   - Live connectivity indicator reflecting FastAPI health status, active ML algorithm (`XGBClassifier`), and operating decision threshold (`τ = 0.35`).
2. **Transaction Intake Form & Demonstration Presets**:
   - Quick-load buttons for interview demonstration:
     - `📌 Load Legitimate Preset`: Populates form with an actual legitimate transaction from test data.
     - `⚠️ Load Fraud Preset`: Populates form with an actual detected fraudulent transaction from test data.
     - `🔄 Reset Inputs`: Clears form fields.
   - Accepts client-supplied raw fields (`Time`, `Amount`, `V1`–`V28`, and optional `transaction_id`).
3. **Scoring Decision & Risk Meter**:
   - Automated Decision Banner: High-visibility `🚨 FRAUD DETECTED` or `✅ LEGITIMATE TRANSACTION`.
   - KPI metrics: Fraud Probability %, Decision, Operational Risk Tier (`LOW`, `MEDIUM`, `HIGH`), and API Latency (ms).
   - Visual probability progress bar calibrated against the operating threshold line.
4. **Dynamic SHAP Feature Attribution**:
   - Horizontal bar chart plotting the top-5 features returned dynamically by the API, color-coded by direction (Red = increases fraud risk, Green = decreases fraud risk).
   - Detailed attribution table showing feature name, raw value, SHAP value, risk direction, and relative attribution share (%).
5. **Live Session Telemetry & Audit Stream**:
   - Aggregates metrics for the current browser session: Total Checked, Frauds Flagged, Legitimate Cleared, and Fraud Incidence Rate %.
   - Reverse-chronological transaction audit table recording timestamp, transaction ID, probability, decision, risk level, and primary risk driver.
6. **Dual-Tab Operations Workspace**:
   - **`⚡ Real-Time Scoring`**: Interactive transaction intake form, presets, probability gauge, risk tier badges, and SHAP attribution waterfall.
   - **`🗄️ Database & Transaction History`**: Live database-wide telemetry cards from `GET /stats`, historical transaction audit table from `GET /transactions`, and an interactive single-transaction inspector rendering stored SHAP features from `GET /transactions/{transaction_id}`.

### 3. Running FastAPI and Streamlit:

#### Step 1: Start the FastAPI Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Step 2: Start the Streamlit Dashboard
```bash
streamlit run dashboard/app.py --server.port 8501
```
Open your browser at: `http://localhost:8501`

---

## 12. Database Persistence & Transaction Audit (Phase 11)

Phase 11 introduces persistent audit storage for all evaluated transactions using **SQLite + SQLAlchemy** without modifying, slowing down, or coupling into the machine learning inference pipeline.

### 1. Architectural Design & Boundaries:
- **Additive Persistence**: The database is purely an audit and history sink. It does **not** participate in the ML prediction, feature engineering, or threshold decision.
- **Fail-Safe Decoupling**: If database persistence encounters an error, the incident is logged with full context while the `/predict` endpoint returns the untouched, valid `PredictionResponse`.
- **Automatic Initialization**: SQLite database tables are verified and created idempotently at application startup via FastAPI lifespan context (`init_db()`). No manual SQL commands required.
- **Zero Data Loss on Restart**: Transactions and audit history persist across server restarts in `data/fraud_detection.db`.
- **Strict Client-Server Isolation**: The Streamlit dashboard never touches SQLite directly; all history and aggregations are queried through FastAPI REST endpoints.

### 2. Database Schema (`transaction_predictions` table):
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Autoincrement | Unique auto-incrementing database audit record ID |
| `transaction_id` | `VARCHAR(100)` | Nullable, Indexed | Client-provided transaction correlation identifier |
| `time` | `FLOAT` | Not Null | Transaction elapsed time (seconds) |
| `amount` | `FLOAT` | Not Null | Monetary transaction amount |
| `fraud_probability` | `FLOAT` | Not Null | Calibrated fraud probability score in $[0.0, 1.0]$ |
| `threshold` | `FLOAT` | Not Null | Operating decision threshold applied ($\tau = 0.35$) |
| `is_fraud` | `BOOLEAN` | Not Null | Binary classification outcome |
| `decision` | `VARCHAR(20)` | Not Null | Categorical decision (`FRAUD` or `LEGITIMATE`) |
| `risk_level` | `VARCHAR(20)` | Not Null | Operational risk tier (`LOW`, `MEDIUM`, `HIGH`) |
| `model_algorithm` | `VARCHAR(50)` | Not Null | Model algorithm name (`XGBClassifier`) |
| `model_version` | `VARCHAR(20)` | Not Null | Deployed model version (`1.0.0`) |
| `latency_ms` | `FLOAT` | Nullable | Combined inference and SHAP attribution latency |
| `top_contributing_features` | `TEXT` | Not Null | JSON-serialized array of top-5 SHAP feature attributions |
| `created_at` | `DATETIME` | Not Null, Indexed | UTC timestamp of the persistent record |

### 3. Duplicate Transaction ID Semantics:
In high-throughput financial environments, transactions may be resubmitted for re-evaluation, audit verification, or post-settlement scoring.
- The internal database `id` remains unique and auto-incrementing.
- Duplicate submissions with the same client `transaction_id` create distinct chronological records in the database, preserving complete historical audit trails.
- Querying `GET /transactions/{transaction_id}` deterministically returns the **latest evaluation** for that correlation identifier.

### 4. History & Analytics Endpoints:
1. `GET /transactions?limit=50&offset=0`:
   - Returns paginated list of stored transactions ordered newest first.
   - Includes pagination metadata (`count`, `total`, `limit`, `offset`).
2. `GET /transactions/{transaction_id}`:
   - Retrieves the latest recorded prediction for a specific client transaction ID.
   - Returns full stored metadata and exact preserved SHAP attributions.
   - Returns HTTP 404 if the transaction ID is not found.
3. `GET /stats`:
   - Executes dynamic SQL aggregate queries: `total_transactions`, `fraud_count`, `legitimate_count`, and `fraud_rate` (%).
   - Accurately represents actual transactions processed by the running system.

---

## 13. High-Level System Architecture

```
[ Financial Transaction Ingestion ]
                 │
                 ▼
       [ Web Operations Dashboard ]
           (Streamlit Monitoring)
                 │
                 │ HTTP (REST)
                 ▼
       [ Real-Time REST API ]
         (FastAPI Microservice)
                 │
                 ├──► [ Feature Pipeline / Transformation ]
                 │
                 ├──► [ ML Scoring Engine (XGBoost) ]
                 │          │
                 │          ├──► Output: Probability & Risk Tier
                 │
                 ├──► [ Explainability Module (SHAP TreeExplainer) ]
                 │          │
                 │          └──► Output: Top-5 Risk Drivers
                 │
                 ├──► [ Persistence Layer (SQLAlchemy ORM) ]
                 │          │
                 │          └──► [ SQLite Audit Store ] (data/fraud_detection.db)
                 │
                 ▼
     [ Real-Time JSON Response ]
```

---

## 14. Planned Technologies

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.10+ | Core development runtime |
| **Data Processing** | Pandas, NumPy, Scipy | Tabular processing, numerical computations, matrix operations |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM | Baseline modeling, ensemble gradient boosting, cross-validation |
| **Class Imbalance Handling** | Imbalanced-learn (SMOTE, UnderSampling) | Addressing severe transaction class skew |
| **Explainability** | SHAP (SHapley Additive exPlanations) | Feature importance and transaction-level explanation |
| **API Serving** | FastAPI, Uvicorn, Pydantic | High-performance async microservice with strict schema validation |
| **Persistence & Audit** | SQLite, SQLAlchemy | Lightweight transactional storage and historical audit logging |
| **Monitoring Dashboard** | Streamlit, Altair | Real-time monitoring, metric inspection, and manual review UI |
| **Testing & Quality** | Pytest, Flake8, Black | Unit/integration testing, linting, and code formatting |

---

## 15. Planned Development Phases

Development proceeds incrementally across structured phases:

- [x] **Phase 1: Project Initialization & Scaffolding** *(Completed)*
  - Establish modular repository structure
  - Define dependencies, environment configurations, and version-control boundaries
  - Document system architecture and project roadmap
- [x] **Phase 2: Dataset Acquisition & Data Understanding** *(Completed)*
  - Acquire legitimate Credit Card Fraud Detection benchmark dataset
  - Verify data integrity, schema, missingness, and baseline statistics
- [x] **Phase 3: Exploratory Data Analysis (EDA)** *(Completed)*
  - Analyze class imbalance, monetary amount distributions, and log scales
  - Correlate features with fraud target and analyze key PCA components
  - Examine 48-hour time dynamics and document the outlier paradox
- [x] **Phase 4: Data Preprocessing & Feature Engineering** *(Completed)*
  - Implement leak-free stratified 80/20 train/test splitting
  - Construct circadian time features (`hour`, `hour_sin`, `hour_cos`) and `log_amount`
  - Fit robust scalers strictly on training data and serialize `models/preprocessor.joblib`
  - Export processed clean dataset splits to `data/processed/`
- [x] **Phase 5: Baseline Model Development & Evaluation** *(Completed)*
  - Train Balanced Logistic Regression and Balanced Random Forest baselines
  - Evaluate accuracy paradox, precision-recall curves, and confusion matrices
  - Perform error analysis on false negatives and false positives
  - Serialize baseline artifacts and metadata to `models/`
- [x] **Phase 6: Imbalanced Learning & Threshold Optimization** *(Completed)*
  - Implement leak-free train/validation/test hierarchy
  - Benchmark algorithmic class weighting against synthetic oversampling (SMOTE)
  - Perform validation threshold calibration ($\tau = 0.40$) for operational FP/FN tradeoff
  - Serialize calibrated model and threshold configuration to `models/`
- [x] **Phase 7: Advanced Gradient Boosting (XGBoost)** *(Completed)*
  - Train high-performance gradient boosted decision trees with cost-sensitive weighting
  - Controlled hyperparameter tuning on validation data prioritizing PR-AUC
  - Calibrate probability thresholds ($\tau = 0.35$) and verify Brier score reliability
  - Single evaluation on untouched test set (PR-AUC: `0.8749`, Precision: `90.22%`, Recall: `84.69%`)
  - Serialize tuned model and threshold config to `models/`
- [x] **Phase 8: Explainability & Interpretability Engine (SHAP)** *(Completed)*
  - Reconstruct TreeExplainer from fitted XGBoost model in margin space
  - Compute exact population-level global feature importance ($N=56,962$)
  - Render beeswarm summary plot and evaluate directional feature impact
  - Extract deterministic local waterfall explanations for TP, FP, FN, and TN cases
  - Build reusable production `FraudExplainer` class in `src/explainability.py`
  - Serialize explainability metadata and figures to `models/` and `reports/figures/`
- [x] **Phase 9: Real-Time API Development (FastAPI)** *(Completed)*
  - Build asynchronous RESTful FastAPI microservice (`api/main.py`)
  - Implement Pydantic data schemas for raw transaction input and predictions (`api/schemas.py`)
  - Lifespan context manager loading XGBoost model, preprocessor, and TreeSHAP explainer once at startup
  - Expose `GET /health` and `POST /predict` endpoints with dynamic threshold evaluation ($\tau = 0.35$)
  - Measure execution latency and package top-5 SHAP features with relative attribution shares
  - Comprehensive unit and integration test suite (`tests/test_api.py`) passing with 100% success rate
- [x] **Phase 10: Interactive Monitoring Dashboard (Streamlit)** *(Completed)*
  - Build interactive web dashboard (`dashboard/app.py`) communicating with FastAPI over REST
  - Maintain strict client-server decoupling with zero direct ML loading inside Streamlit
  - Transaction input form with quick-fill presets for legitimate and fraudulent cases
  - Automated decision banner, calibrated probability progress bar, and operational risk tiers
  - Dynamic SHAP attribution horizontal bar chart and detailed feature contribution data table
  - Real-time session telemetry counters (Total Checked, Frauds Flagged, Cleared, Fraud %) and audit log
  - Automated dashboard test suite (`tests/test_dashboard.py`) passing with 100% success rate
- [x] **Phase 11: Database Persistence & Transaction History** *(Completed)*
  - Design SQLite persistence layer and ORM model (`api/database.py`, `api/models.py`, `api/repositories.py`)
  - Automatic idempotent database table initialization on startup (`data/fraud_detection.db`)
  - Seamless persistence hook in `/predict` preserving exact SHAP outputs and response contract
  - New REST API endpoints for history and metrics (`GET /transactions`, `GET /transactions/{id}`, `GET /stats`)
  - Extended `FraudApiClient` and created dual-tab Streamlit dashboard (`dashboard/components/history_view.py`)
  - Expanded test suite to 31 tests with isolated temporary SQLite testing fixtures (`tests/test_database.py`)
- [ ] **Phase 12: Production Containerization & Deployment**
  - Containerize microservice and dashboard using multi-stage Dockerfiles
  - Define Docker Compose orchestrating FastAPI, SQLite volume mounts, and Streamlit
  - Establish production logging, health probes, and deployment guidelines

---

## 16. What the System Will Eventually Do
Upon completion of all phases, an operator or client application will be able to:
- Send a transaction payload (timestamp, amount, user tokens, location indicators, transaction characteristics) via a REST API endpoint.
- Receive a sub-second response indicating:
  - `fraud_probability`: Floating-point score between `0.0` and `1.0`.
  - `risk_level`: `LOW`, `MEDIUM`, or `HIGH`.
  - `explanation`: Top contributing features (e.g., unusually high amount, anomalous time-of-day, atypical velocity).
- Query transaction history, historical risk distributions, and model explanations via persistent REST endpoints.
- Monitor system performance, transaction distribution, and live alerts through a centralized web dashboard.

---

## 17. Directory Structure

```
fraud-detection-system/
│
├── data/
│   ├── raw/                       # Raw source dataset (creditcard.csv)
│   ├── processed/                 # Cleaned splits (train.csv, test.csv)
│   └── fraud_detection.db         # Persistent SQLite audit database
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_imbalanced_learning.ipynb
│   ├── 06_advanced_model.ipynb
│   └── 07_shap_explainability.ipynb
│
├── src/                           # Core application source code
│   ├── preprocessing.py           # TransactionPreprocessor pipeline
│   └── explainability.py          # Production FraudExplainer with SHAP TreeExplainer
│
├── models/                        # Serialized model artifacts, configs & metadata
│   ├── preprocessor.joblib
│   ├── logistic_regression_baseline.joblib
│   ├── random_forest_baseline.joblib
│   ├── rf_calibrated_threshold_model.joblib
│   ├── threshold_config.json
│   ├── baseline_metadata.json
│   ├── xgboost_advanced_model.joblib
│   ├── xgboost_threshold_config.json
│   ├── global_feature_importance.json
│   └── shap_summary_metadata.json
│
├── reports/
│   └── figures/                   # Exported visualizations & explainability plots
│       ├── shap_global_importance.png
│       ├── shap_beeswarm_summary.png
│       ├── shap_waterfall_tp.png
│       ├── shap_waterfall_fp.png
│       ├── shap_waterfall_fn.png
│       └── shap_waterfall_tn.png
│
├── api/                           # FastAPI service definitions, schemas, and routes
│   ├── main.py                    # FastAPI application routes & lifespan initialization
│   ├── schemas.py                 # Pydantic v2 data models for requests, responses & history
│   ├── services.py                # FraudService orchestration layer
│   ├── database.py                # SQLite engine & SQLAlchemy session management
│   ├── models.py                  # TransactionRecord ORM persistent model
│   └── repositories.py            # Persistence CRUD and SQL aggregation queries
│
├── dashboard/                     # Streamlit operations & monitoring frontend
│   ├── app.py                     # Streamlit main entrypoint with dual-tab workspace
│   ├── components/                # Modular UI components (header, form, results, shap, telemetry, history)
│   │   ├── header.py
│   │   ├── transaction_form.py
│   │   ├── results_view.py
│   │   ├── shap_charts.py
│   │   ├── telemetry.py
│   │   └── history_view.py
│   ├── services/                  # Decoupled REST client (api_client.py)
│   └── utils/                     # Raw demonstration presets (presets.py)
│
├── tests/                         # Automated unit and integration test suite
│   ├── test_explainability.py     # SHAP explainer unit tests
│   ├── test_api.py                # FastAPI endpoints & persistence integration tests
│   ├── test_dashboard.py          # Streamlit client & component integration tests
│   └── test_database.py           # SQLite database layer & repository tests
│
├── README.md                      # Project documentation and architectural overview
├── requirements.txt               # Core dependencies including SQLAlchemy
└── .gitignore                     # Version control exclusions
```

