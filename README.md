# E-Commerce Customer Churn Prediction

## 👥 Tim Proyek

**Batch 5 - Kelompok 4 MSIB Data Science**:
1. Wesly Daud Siahaan
2. Rizki Nurhafizd Achmad
3. Dwi Mutiara Senen
4. Muhammad Luthfi Nurhadi
5. Chellcia Mutiara Iwfanka
6. Arzad Lintang Maharani
7. Muhammad Fathur Arkana

---

## 📋 Daftar Isi
- [Tim Proyek](#-tim-proyek)
- [Deskripsi Proyek](#-deskripsi-proyek)
- [Struktur Direktori](#-struktur-direktori)
- [Hasil Model](#-hasil-model)
- [Business Insights](#-business-insights)
- [Rekomendasi Bisnis](#-rekomendasi-bisnis)
- [Setup & Instalasi](#-setup--instalasi)
- [Panduan Penggunaan](#-panduan-penggunaan)
- [Dokumentasi Teknis](#-dokumentasi-teknis)

---

## 🎯 Deskripsi Proyek

### Latar Belakang
Pelanggan yang berhenti menggunakan layanan (churn) merupakan tantangan utama bagi perusahaan e-commerce. Mempertahankan pelanggan yang ada lebih efisien dari pada mengakuisisi pelanggan baru dari sisi biaya operasional.

### Objektif
Proyek ini bertujuan untuk:
- **Memprediksi** pelanggan yang berisiko churn dengan akurasi tinggi
- **Mengidentifikasi** faktor-faktor utama yang mempengaruhi keputusan pelanggan untuk churn
- **Memberikan** rekomendasi bisnis yang actionable untuk meningkatkan retensi pelanggan

### Dataset
- **Total Samples**: 5,630 pelanggan
- **Total Features**: 20 fitur (campuran numerik dan kategorikal)
- **Target Variable**: `Churn` (Binary: 0 = No Churn, 1 = Churn)
- **Class Distribution**: 
  - No Churn: 83.16% (4,682 pelanggan)
  - Churn: 16.84% (948 pelanggan)
  - **Status**: Imbalanced dataset

### Metodologi
Proyek ini mengikuti **CRISP-DM Framework**:
1. **Business Understanding** → Memahami masalah churn
2. **Data Understanding** → Exploratory Data Analysis (EDA)
3. **Data Preparation** → Cleaning, Preprocessing, Feature Engineering
4. **Modeling** → Training & Hyperparameter Tuning
5. **Evaluation** → Model Assessment & Metrics
6. **Deployment** → Save Artifacts untuk Production

---

## 📁 Struktur Direktori

```
ecommerce_churn/
├── README.md                           # Dokumentasi proyek (file ini)
├── requirements.txt                    # Daftar dependencies
│
├── config/
│   └── config.yaml                     # Konfigurasi semua parameter
│
├── data/
│   ├── raw/
│   │   └── E Commerce Dataset.xlsx     # Data mentah dari sumber
│   └── processed/
│       ├── preprocessed_data.csv       # Data setelah preprocessing
│       └── model_data.csv              # Data siap untuk modeling
│
├── src/                                # Source code modular
│   ├── data_preprocessing.py           # Data cleaning & preparation
│   ├── feature_engineering.py          # Feature transformation & encoding
│   └── model_trainer.py                # Model training & evaluation
│
├── scripts/
│   └── save_artifacts.py               # Save model, scaler, features
│
├── models/                             # Saved artifacts
│   ├── model_rf.pkl                    # Trained Random Forest model
│   ├── model_xgb.pkl                   # Trained XGBoost model
│   ├── preprocessor.pkl                # Preprocessor pipeline for inference
│   └── scaler.pkl                      # Standard scaler used for numeric features
│
├── notebook/
│   ├── ecommerce_prediction.ipynb      # Jupyter notebook (original)
│   └── Source_Code.ipynb
│
├── doc/                                # Documentation & analysis
│   └── technical_documentation.md      # Dokumentasi teknis
│
├── docs/                               # Dokumentasi tambahan dan panduan penggunaan
│   ├── project_overview.md
│   ├── model_artifacts.md
│   └── testing_guide.md
│
├── presentations/                      # Folder untuk presentasi
│   └── (empty - untuk file PPT Anda)
│
└── tests/                              # Unit tests
    └── (test files)
```

---

## 📊 Hasil Model

### Model Terbaik: **XGBoost**

#### Performance Metrics (Test Set):
| Metrik | Score | Interpretasi |
|--------|-------|--------------|
| **Accuracy** | 0.96 | 96% data diprediksikan dengan benar |
| **Precision** | 0.96 | 96% prediksi churn benar-benar churn |
| **Recall** | 0.97 | 97% pelanggan churn berhasil diidentifikasi |
| **F1-Score** | 0.96 | Keseimbangan sempurna antara precision & recall |
| **ROC-AUC** | 0.98 | Excellent discrimination ability |

#### Confusion Matrix (Test Set):
```
                Predicted No Churn    Predicted Churn
Actual No Churn        232                 3
Actual Churn            47               1,313
```

**Interpretasi:**
- **True Positives (TP)**: 1,313 → Berhasil mengidentifikasi pelanggan yang churn
- **True Negatives (TN)**: 232 → Berhasil mengidentifikasi pelanggan yang tidak churn
- **False Positives (FP)**: 3 → Salah memprediksi churn (biaya false alarm minimal)
- **False Negatives (FN)**: 47 → Melewatkan beberapa pelanggan churn (masih acceptable)

### Perbandingan Model

| Model | Accuracy | Precision | Recall | F1-Score | Status |
|-------|----------|-----------|--------|----------|--------|
| Logistic Regression | 0.84 | 0.79 | 0.95 | 0.86 | ❌ Overfitting |
| K-Nearest Neighbors | 0.81 | 0.77 | 0.85 | 0.81 | ❌ Overfitting |
| Decision Tree | 0.80 | 0.80 | 0.81 | 0.80 | ❌ Overfitting |
| Random Forest | 0.93 | 0.95 | 0.97 | 0.96 | ✅ Good |
| AdaBoost | 0.89 | 0.89 | 0.94 | 0.91 | ⚠️ Acceptable |
| **XGBoost** | **0.96** | **0.96** | **0.97** | **0.96** | **✅ Best** |

---

## 🔍 Business Insights

### 1. **Tenure (Loyalitas Pelanggan) - Faktor Paling Penting**

**Insight:**
- Pelanggan dengan tenure rendah (< 9 bulan) memiliki churn rate paling tinggi
- 70% dari semua churn terjadi pada pelanggan dengan tenure < 20 bulan
- Churn rate menurun drastis seiring dengan peningkatan tenure

**Implikasi:**
- Bulan-bulan pertama adalah periode kritis ("onboarding phase")
- Pelanggan baru perlu perhatian khusus untuk meningkatkan retensi
- Investasi pada customer engagement di awal sangat penting

### 2. **Customer Category & Order Count**

**Insight:**
- **Bronze** (OrderCount < 4): Churn rate 28.5%
- **Silver** (OrderCount 4-8): Churn rate 12.3%
- **Gold** (OrderCount ≥ 9): Churn rate 2.1%

**Pola:**
- Semakin banyak order yang dilakukan, semakin loyal pelanggan tersebut
- Mayoritas pelanggan adalah kategori Bronze (34.2% dari total)

### 3. **Complain Rate & Kepuasan Pelanggan**

**Insight:**
- 31.67% dari pelanggan yang churn adalah pelanggan yang pernah komplain
- Satisfaction Score rendah (1-2) berkorelasi dengan churn rate tinggi
- Komplain yang tidak ditangani dengan baik menjadi driver churn utama

**Implikasi:**
- Sistem customer service perlu ditingkatkan
- Response time dan resolution quality adalah kunci retensi

### 4. **Cashback & Promosi Strategy**

**Insight:**
- Pelanggan Bronze menggunakan 45% dari total cashback yang diberikan
- Pelanggan yang menerima besar cashback cenderung lebih besar churn-nya
- Indikasi: Pelanggan hanya tertarik pada benefit jangka pendek, bukan loyalitas

**Pola Penggunaan:**
- Pelanggan dengan Tenure 0-1 bulan mendapat cashback terbesar
- Cashback tidak efektif untuk retensi jangka panjang

### 5. **Coupon & Ordering Behavior**

**Insight:**
- Korelasi kuat antara CouponUsed dan OrderCount (correlation = 0.75)
- Setiap peningkatan coupon penggunaan, order count meningkat
- Strategi coupon lebih efektif dari cashback untuk mendorong pembelian

### 6. **Marital Status**

**Insight:**
- Single customers: Churn rate 19.2%
- Married customers: Churn rate 14.8%
- Divorced/Widowed: Churn rate 14.5%

**Interpretasi:**
- Single customers lebih price-sensitive dan suka membandingkan platform
- Married customers lebih settled dan loyal

---

## 💡 Rekomendasi Bisnis

### 🎯 Strategi Jangka Pendek (0-3 Bulan)

#### 1. **Optimalisasi Onboarding Pelanggan Baru**
- **Target**: Pelanggan Bronze (< 9 bulan)
- **Aksi**:
  - Buat welcome program yang menarik
  - Implement loyalty game/point system untuk engagement
  - Personalized product recommendations
  - Follow-up email/push notification berkala
- **Expected Impact**: Turunkan Bronze churn dari 28.5% → 20%

#### 2. **Improve Customer Service**
- **Target**: Mengurangi 31.67% churn dari complain
- **Aksi**:
  - Implement chatbot untuk instant support
  - SLA response time: max 2 jam untuk komplain
  - Kompensasi untuk komplain yang unresolved
  - Tracking dan monitoring complaint resolution
- **Expected Impact**: Turunkan complaint-related churn 50%

#### 3. **Optimize Coupon Strategy**
- **Target**: Replace ineffective cashback dengan targeted coupon
- **Aksi**:
  - Gunakan coupon untuk mendorong purchase frequency
  - Personalized coupon based on browsing history
  - Time-based coupon (flash sale) untuk Bronze customers
  - Bundle deals untuk cross-selling
- **Expected Impact**: Increase OrderCount untuk Bronze segment

### 🎯 Strategi Jangka Menengah (3-6 Bulan)

#### 1. **Develop Loyalty Program**
- **Tier-based Program**: Bronze → Silver → Gold
- **Benefit**:
  - Exclusive discounts untuk Silver & Gold
  - Early access ke produk baru
  - VIP customer service
  - Milestone rewards
- **Expected Impact**: Accelerate customer progression dari Bronze → Silver

#### 2. **Personalized Engagement Strategy**
- **Data-Driven Segmentation**:
  - Segment berdasarkan product category preference
  - Segment berdasarkan demographic (gender, age)
  - Segment berdasarkan behavior (purchase frequency, AOV)
- **Targeted Campaign**:
  - Targeted email untuk setiap segment
  - Product recommendation engine
  - Time-based triggered messages
- **Expected Impact**: Increase retention by 15-20%

#### 3. **Enhance Product Offering**
- **Current Top Categories**: Mobile Phone, Laptop & Accessories
- **Action**:
  - Expand inventory untuk popular categories
  - Develop new categories untuk less popular ones
  - Partner dengan new brands untuk exclusivity
- **Expected Impact**: Increase product variety attractiveness

### 🎯 Strategi Jangka Panjang (6+ Bulan)

#### 1. **Build Community & Engagement**
- Review & rating system
- User-generated content platform
- Social features (sharing, wishlists)
- Community events & webinar

#### 2. **Data Infrastructure**
- Real-time churn prediction using XGBoost model
- Proactive intervention untuk high-risk customers
- A/B testing framework untuk campaign optimization
- Marketing automation untuk personalized journeys

#### 3. **Product Features**
- Subscription program untuk repeat items
- AI-powered personalization engine
- Extended warranty & protection plans
- Trade-in program untuk electronics

---

## ⚙️ Setup & Instalasi

### Prerequisites
- Python 3.8+
- pip atau conda
- 2GB free disk space

### Installation Steps

```bash
# 1. Clone repository
cd /path/to/ecommerce_churn

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import pandas, sklearn, xgboost; print('All packages installed successfully!')"
```

### Required Libraries
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
xgboost>=1.4.0
imbalanced-learn>=0.8.0
pyyaml>=5.4.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## 📖 Panduan Penggunaan

### 1. **Data Preprocessing**

```python
from src.data_preprocessing import preprocess_pipeline
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run preprocessing
df = preprocess_pipeline(
    config_dict=config,
    input_path='data/raw/E Commerce Dataset.xlsx',
    output_path='data/processed/preprocessed_data.csv'
)
```

### 2. **Feature Engineering**

```python
from src.feature_engineering import feature_engineering_pipeline

# Run feature engineering
df_engineered = feature_engineering_pipeline(
    config_dict=config,
    df=df,
    drop_original_features=True
)
```

### 3. **Model Training**

```python
from src.model_trainer import complete_training_pipeline

# Train model
model, evaluation_results = complete_training_pipeline(
    config_dict=config,
    X=X,
    y=y,
    model_name="xgboost"
)
```

### 4. **Save Artifacts**

```python
from scripts.save_artifacts import ArtifactManager

# Save model, scaler, features
manager = ArtifactManager(artifacts_dir="models")
paths = manager.save_all_artifacts(
    model=model,
    scaler=scaler,
    feature_names=X.columns.tolist(),
    metadata={
        'model_type': 'XGBoost',
        'accuracy': evaluation_results['accuracy'],
        'precision': evaluation_results['precision'],
        'recall': evaluation_results['recall']
    },
    model_name="xgboost_final"
)
```

---

## 📚 Dokumentasi Teknis

### Metodologi Preprocessing

#### Handling Missing Values
| Column | Method | Reasoning |
|--------|--------|-----------|
| Tenure | Median | Skewed distribution |
| OrderCount | Median | Skewed distribution |
| CashbackAmount | Median | Skewed distribution |
| HourSpendOnApp | Mean | Normal-like distribution |

#### Outlier Detection & Removal
- **Method**: Z-Score (threshold = 3)
- **Reasoning**: More conservative than IQR, preserves ~99.7% valid data
- **Impact**: Removed ~0.3% of outliers

#### Feature Transformation
- **Normalization** (MinMaxScaler): CashbackAmount, DaySinceLastOrder, NumberOfDeviceRegistered
- **Standardization** (StandardScaler): Tenure, WarehouseToHome
- **Purpose**: Improve model convergence and performance

#### Feature Encoding
- **Label Encoding**: Gender (0=Female, 1=Male)
- **One-Hot Encoding**: MaritalStatus, PreferredLoginDevice, PreferedOrderCat

### Handling Class Imbalance

**Problem**: Dataset imbalanced (83% No Churn, 17% Churn)

**Solution**: SMOTE (Synthetic Minority Over-sampling Technique)
- Generates synthetic samples dari minority class
- Preserves class distribution information
- Applied only pada training set, NOT pada test set
- Training set balanced: 50% No Churn, 50% Churn

### Model Selection Justification

**Why XGBoost?**
1. **Superior Performance**: Highest recall & precision
2. **Handles Imbalance**: Built-in scale_pos_weight parameter
3. **Feature Importance**: Interpretable feature rankings
4. **Efficiency**: Fast training & inference
5. **Production-Ready**: Proven in industry

**Hyperparameters Configuration**:
```yaml
max_depth: 20              # Tree complexity
min_child_weight: 5        # Minimum instances per leaf
gamma: 0.5                 # Minimum loss reduction
colsample_bytree: 0.7      # Feature subsampling
eta: 0.1                   # Learning rate
lambda: 0.5                # L2 regularization
```

### Evaluation Metrics Rationale

**Primary Metric: Recall**
- **Why**: Minimize False Negatives (missed churn customers)
- **Impact**: Each missed churn = loss of customer LTV
- **Business Cost**: Higher than False Positive

**Secondary Metric: Precision**
- **Why**: Minimize False Positives (wrongly targeted as churn)
- **Impact**: Unnecessary retention cost & poor targeting
- **Business Impact**: Budget inefficiency

**Tertiary Metric: F1-Score**
- **Why**: Balance between Recall & Precision
- **Usage**: Overall model quality assessment

---

## 📝 Catatan Penting

### Asumsi & Batasan
1. Dataset diasumsikan representative dari populasi pelanggan actual
2. Pattern historis diasumsikan akan berlanjut di masa depan
3. Model tidak mempertimbangkan faktor eksternal (ekonomi makro, kompetitor)
4. Feature engineering dilakukan tanpa domain expert feedback

### Future Improvements
1. Collect more features (demographic, transaction history depth)
2. Implement real-time churn prediction pipeline
3. A/B test retention strategies
4. Ensemble model dengan multiple algorithms
5. Deep learning approaches (Neural Networks)

---

## 📞 Kontak & Support

Untuk pertanyaan atau feedback, silakan buat issue di repository.

---

**Last Updated**: May 10, 2024  
**Version**: 1.0  
**Status**: Production Ready ✅
