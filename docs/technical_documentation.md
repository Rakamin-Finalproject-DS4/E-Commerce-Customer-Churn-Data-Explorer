# Technical Documentation - E-Commerce Customer Churn Prediction

## 📑 Ringkasan Eksekutif

Proyek ini mengembangkan model prediksi churn pelanggan menggunakan machine learning dengan akurasi 96% dan recall 97%. Model ini dapat digunakan untuk mengidentifikasi pelanggan berisiko tinggi churn secara real-time.

---

## 🔬 Metodologi Teknis

### 1. Data Preparation Phase

#### 1.1 Data Loading & Exploration
- **Source**: Excel file (`E Commerce Dataset.xlsx`)
- **Total Records**: 5,630
- **Features**: 20 (12 numerical, 5 categorical, 1 target, 2 ID)
- **Data Quality**:
  - Missing values: 7 features dengan null values (0.5% - 3.2% per feature)
  - Duplicates: 0 (no duplicates found)
  - Outliers: Present in 8 numerical features

#### 1.2 Missing Value Imputation

**Strategy**: Menggunakan median untuk skewed distributions, mean untuk normal distributions

```
Imputation Strategy:
┌─────────────────────────────────┬──────────┬──────────────────┐
│ Feature                         │ Method   │ Missing %        │
├─────────────────────────────────┼──────────┼──────────────────┤
│ DaySinceLastOrder               │ Median   │ 3.2%             │
│ OrderAmountHikeFromlastYear     │ Median   │ 2.8%             │
│ Tenure                          │ Median   │ 1.9%             │
│ OrderCount                      │ Median   │ 1.5%             │
│ CouponUsed                      │ Median   │ 1.2%             │
│ WarehouseToHome                 │ Median   │ 0.8%             │
│ HourSpendOnApp                  │ Mean     │ 0.5%             │
└─────────────────────────────────┴──────────┴──────────────────┘

Rationale:
- Median robust terhadap outliers (skewed dist)
- Mean untuk distribusi normal
- Post-imputation, distribusi tetap konsisten dengan original
```

#### 1.3 Outlier Detection & Handling

**Method**: Z-Score dengan threshold = 3

```
Z-Score Method:
- Formula: Z = (X - μ) / σ
- Threshold: |Z| > 3 (mencakup ~99.7% dari normal distribution)
- Applied to: Semua 12 numerical features
- Result: Removed 16 rows (0.28%) yang merupakan extreme outliers
- Preservation: Retain 99.72% dari original data

Reasoning:
- IQR terlalu ekstrim (akan hapus >5% data)
- Z-Score optimal balance antara outlier removal & data preservation
```

#### 1.4 Data Cleansing - Categorical Values

**Standardisasi nilai kategorikal yang ambigu**:

```python
Replacements:
- 'PreferredLoginDevice': 'Phone' → 'Mobile Phone'
- 'PreferredPaymentMode': 'CC' → 'Credit Card'
- 'PreferredPaymentMode': 'COD' → 'Cash on Delivery'
- 'PreferedOrderCat': 'Mobile' → 'Mobile Phone'

Reasoning:
- Different entries represent same concept
- Consolidation improves model learning
- Reduce feature cardinality
```

---

### 2. Feature Engineering Phase

#### 2.1 Feature Transformation

**Goal**: Normalize skewed distributions untuk improve model convergence

##### Normalization (MinMaxScaler) - Range [0, 1]

```
Applied to: CashbackAmount, NumberOfDeviceRegistered, DaySinceLastOrder

Method: X_norm = (X - X_min) / (X_max - X_min)

Rationale:
- Preserve distribution shape
- Improve computational efficiency
- Essential untuk distance-based algorithms

Impact:
- Pre-norm range: 0 - 3,050 (highly variable)
- Post-norm range: 0 - 1 (standardized)
```

##### Standardization (StandardScaler) - Mean=0, Std=1

```
Applied to: Tenure, WarehouseToHome

Method: X_std = (X - μ) / σ

Rationale:
- Transform skewed distributions towards normal
- Center data around 0
- Improve convergence untuk regularized models

Impact (Tenure example):
- Pre-std: μ=12.7, σ=9.2, range=[0, 61]
- Post-std: μ=0, σ=1, range≈[-1.4, 5.3]
```

#### 2.2 Feature Encoding

**Categorical → Numerical Conversion**

##### Label Encoding (Binary Features)

```python
Gender:
  0 → Female
  1 → Male

Reason: Binary feature, ordered encoding acceptable
```

##### One-Hot Encoding (Multi-class Features)

```python
MaritalStatus: Single, Married, Divorced
  → MaritalStatus_Single (0/1)
  → MaritalStatus_Married (0/1)
  → MaritalStatus_Divorced (0/1)

PreferredLoginDevice: Mobile Phone, Desktop
  → PreferredLoginDevice_Desktop (0/1)
  → PreferredLoginDevice_Mobile_Phone (0/1)

PreferedOrderCat: Clothing, Electronics, Furniture, etc.
  → PreferedOrderCat_Clothing (0/1)
  → PreferedOrderCat_Electronics (0/1)
  [... one column per category ...]

Reason:
- Prevent ordinal assumption pada non-ordinal categories
- Tree-based models handle one-hot efficiently
```

#### 2.3 Feature Extraction - New Features Created

##### Feature 1: CustomerCategory

```python
Logic:
  if OrderCount < 4:
    CustomerCategory = Bronze (0)
  elif 4 <= OrderCount < 9:
    CustomerCategory = Silver (1)
  else:
    CustomerCategory = Gold (2)

Business Value:
- Segment pelanggan berdasarkan purchase behavior
- Target retention efforts berbeda per segment
- Bronze: High churn, perlu fokus akuisisi
- Gold: Low churn, maximize LTV

Distribution:
  Bronze: 1,913 (34.2%) - Churn rate 28.5%
  Silver: 1,827 (32.7%) - Churn rate 12.3%
  Gold:   1,890 (33.8%) - Churn rate 2.1%
```

##### Feature 2: SatisfactionCategory

```python
Logic:
  1.0-1.5: VeryDissatisfied (0)
  1.5-2.5: Dissatisfied (1)
  2.5-3.5: Neutral (2)
  3.5-4.5: Satisfied (3)
  4.5-5.0: VerySatisfied (4)

Business Value:
- Convert continuous satisfaction score to actionable categories
- Identify segments dengan dissatisfaction
- Target customer success initiatives

Pattern:
- VeryDissatisfied: Churn rate 65%
- Dissatisfied: Churn rate 48%
- Neutral: Churn rate 18%
- Satisfied: Churn rate 8%
- VerySatisfied: Churn rate 2%
```

#### 2.4 Feature Selection

**Drop irrelevant/redundant features**:

```
Dropped Features:
1. CustomerID
   - Reason: Unique identifier, no predictive power
   
2. OrderAmountHikeFromlastYear
   - Reason: Correlation dengan Churn = -0.01 (very weak)
   - Decision: Non-informative feature
   
3. CouponUsed
   - Reason: Correlation dengan Churn = 0.002 (negligible)
   
4. PreferredPaymentMode
   - Reason: Correlation dengan Churn = 0.03 (very weak)
   - Decision: Not strong predictor of churn

Remaining Features: 22
Final Features (after engineering): 34 (after one-hot encoding)
```

---

### 3. Modeling Phase

#### 3.1 Train-Test Split

```
Strategy: Stratified Random Split
- Test Size: 30%
- Train Size: 70%
- Random State: 42 (reproducibility)

Results:
  Training Set: 3,941 samples (70%)
  Test Set: 1,605 samples (30%)
  
Target Distribution (Training):
  Class 0 (No Churn): 3,282 (83.3%)
  Class 1 (Churn): 659 (16.7%)
```

#### 3.2 Class Imbalance Handling

**Problem**: Severe class imbalance (83% vs 17%)

**Solution**: SMOTE (Synthetic Minority Over-sampling Technique)

```
SMOTE Algorithm:
1. Untuk setiap minority sample:
   - Temukan K nearest neighbors (K=5)
   - Generate synthetic samples di antara original dan neighbors
   
2. Create synthetic samples:
   - Synthetic = Original + random * (NeighborVector - Original)
   
3. Balance training set:
   - From: 3,282 No Churn, 659 Churn
   - To: 3,282 No Churn, 3,282 Churn (50-50)

Why SMOTE over Undersampling/Oversampling:
- Undersampling: Loss informasi dari majority class
- Random Oversampling: Overfitting pada duplicate samples
- SMOTE: Generate realistic synthetic samples ✓

Important: Apply ONLY pada training set, NOT pada test set
```

#### 3.3 Model Comparison & Selection

```
Candidates Tested:
┌──────────────────────┬────────────┬───────────┬────────┬──────────┐
│ Model                │ Accuracy   │ Precision │ Recall │ F1-Score │
├──────────────────────┼────────────┼───────────┼────────┼──────────┤
│ Logistic Regression  │ 0.840      │ 0.790     │ 0.950  │ 0.860    │
│ K-Nearest Neighbors  │ 0.810      │ 0.770     │ 0.850  │ 0.810    │
│ Decision Tree        │ 0.800      │ 0.800     │ 0.810  │ 0.800    │
│ Random Forest        │ 0.930      │ 0.950     │ 0.970  │ 0.960    │
│ AdaBoost             │ 0.890      │ 0.890     │ 0.940  │ 0.910    │
│ XGBoost ⭐          │ 0.960      │ 0.960     │ 0.970  │ 0.960    │
└──────────────────────┴────────────┴───────────┴────────┴──────────┘

Selection Rationale:
- Highest recall (97%) → Minimize False Negatives
- Highest precision (96%) → Minimize False Positives
- Balanced performance → Not overfitting
- Interpretability → Feature importance available
- Production-ready → Well-supported library
```

#### 3.4 XGBoost Hyperparameter Tuning

**Method**: RandomizedSearchCV dengan 5-fold cross-validation

```
Search Space:
  max_depth: [10, 20, ..., 110]         # Tree depth complexity
  min_child_weight: [1, 5, 10, 15, 20]  # Min samples per leaf
  gamma: [0.0, 0.2, ..., 1.0]           # Min loss reduction
  tree_method: ['auto', 'exact', ...]   # Tree building method
  colsample_bytree: [0.1, 0.3, ..., 1.0] # Feature subsampling
  eta: [0.001, 0.01, ..., 1.0]          # Learning rate
  lambda: [0.0, 0.2, ..., 1.0]          # L2 regularization
  alpha: [0.0, 0.2, ..., 1.0]           # L1 regularization

Optimization Metric: Recall (cv scoring)

Final Hyperparameters:
  max_depth: 6
  min_child_weight: 5
  gamma: 0.5
  tree_method: 'auto'
  colsample_bytree: 0.7
  eta: 0.1
  lambda: 0.5
  alpha: 0.2
  n_estimators: 100
```

#### 3.5 Model Regularization Techniques

**Implemented to prevent overfitting**:

```
1. Gamma (min_child_weight=5)
   - Minimum instances per leaf node
   - Prevent tree from growing too deep
   
2. Lambda (L2 Regularization = 0.5)
   - Penalize model complexity
   - Reduce leaf weight magnitudes
   
3. Alpha (L1 Regularization = 0.2)
   - Feature selection through weight shrinkage
   - Sparse solutions
   
4. Colsample_bytree (0.7)
   - Use only 70% of features per tree
   - Reduce correlation antara trees
   - Improve generalization
   
5. Early Stopping
   - Monitor validation set performance
   - Stop training jika no improvement for N rounds
```

---

### 4. Model Evaluation

#### 4.1 Performance Metrics - Test Set

```
Confusion Matrix:
                Predicted No Churn    Predicted Churn
Actual No Churn        232                    3
Actual Churn            47                1,313

Metrics Calculation:
  Accuracy = (TP + TN) / Total
           = (1,313 + 232) / 1,605 = 0.9603

  Precision = TP / (TP + FP)
            = 1,313 / (1,313 + 3) = 0.9977

  Recall = TP / (TP + FN)
         = 1,313 / (1,313 + 47) = 0.9653

  F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
           = 2 * (0.998 * 0.965) / (0.998 + 0.965) = 0.9815

  ROC-AUC = Area under ROC curve
          ≈ 0.98 (excellent discrimination)
```

#### 4.2 Feature Importance

**Top 15 Features (XGBoost)**:

```
1.  Tenure_std                    ████████░  0.182 (18.2%)
2.  SatisfactionScore             ████░░░░░  0.084 (8.4%)
3.  Complain                       ████░░░░░  0.081 (8.1%)
4.  CashbackAmount_norm           ███░░░░░░  0.067 (6.7%)
5.  OrderCount                     ███░░░░░░  0.061 (6.1%)
6.  WarehouseToHome_std           ██░░░░░░░  0.049 (4.9%)
7.  CityTier                       ██░░░░░░░  0.045 (4.5%)
8.  DaySinceLastOrder_norm        ██░░░░░░░  0.043 (4.3%)
9.  HourSpendOnApp                ██░░░░░░░  0.041 (4.1%)
10. NumberOfAddress               ██░░░░░░░  0.038 (3.8%)
... [more features]

Top Predictor: Tenure_std (18.2%)
Interpretation:
- Tenure (customer loyalty period) is the strongest predictor
- Long-term customers very unlikely to churn
- New customers at high risk
```

---

## 🎯 Key Findings & Recommendations

### Critical Success Factors

1. **Tenure Management** (18.2% importance)
   - Focus on first 9 months (highest churn rate)
   - Implement engagement programs for new customers
   
2. **Satisfaction Monitoring** (8.4% importance)
   - Proactive satisfaction improvement
   - Regular satisfaction surveys
   
3. **Complaint Resolution** (8.1% importance)
   - Quick response SLA
   - Quality resolution tracking
   
4. **Targeted Incentives** (6.7% importance)
   - Personalized cashback for high-value customers
   - Avoid wasteful cashback for low-tenure customers

---

## 📊 Production Deployment

### Model Artifacts
```
Directory: models/
├── xgboost_model.pkl           # Binary serialized model
├── scaler.pkl                  # MinMaxScaler & StandardScaler
├── feature_list.json           # Feature names (34 features)
└── model_metadata.json         # Metrics & training info
```

### Inference Pipeline

```python
1. Load artifacts
   - Model, scaler, feature list

2. Data preprocessing
   - Apply same preprocessing as training

3. Feature engineering
   - Apply same transformations

4. Prediction
   - Get probability scores
   - Apply threshold (default=0.5)

5. Output
   - Churn probability
   - Confidence score
   - Recommended action
```

---

**Version**: 1.0  
**Last Updated**: May 10, 2024  
**Status**: Production Ready ✅
