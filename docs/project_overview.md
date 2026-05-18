# Project Overview

## Purpose

Proyek ini bertujuan memprediksi churn pelanggan e-commerce dengan menggunakan metode machine learning yang terstruktur menurut kerangka CRISP-DM. Tujuan utamanya adalah membantu tim bisnis mengidentifikasi pelanggan berisiko tinggi churn dan memberikan rekomendasi retensi.

## Struktur Utama

- `notebook/ecommerce_prediction.ipynb`
  - Notebook utama yang memuat alur CRISP-DM lengkap: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, dan Insights.
- `src/`
  - `data_preprocessing.py`: Logika pembersihan dan impor data.
  - `feature_engineering.py`: Transformasi, encoding, dan ekstraksi fitur.
  - `model_trainer.py`: Training model, imbalance handling, dan evaluasi.
- `models/`
  - Tempat artifacts model tersimpan setelah notebook dieksekusi.
- `tests/`
  - Unit test untuk memverifikasi komponen preprocessing, feature engineering, training, dan penyimpanan artifact.
- `docs/`
  - Dokumentasi tambahan yang menjelaskan penggunaan, artifacts, dan pengujian.

## Model Artifacts

Setelah notebook dijalankan, file-file berikut disimpan di `models/`:

- `model_rf.pkl` - Model Random Forest terlatih
- `model_xgb.pkl` - Model XGBoost terlatih
- `preprocessor.pkl` - Preprocessor `ColumnTransformer`
- `scaler.pkl` - Scaler standar yang digunakan untuk numeric features

## Bagaimana Menggunakan

1. Pastikan semua dependency terpasang:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan notebook `notebook/ecommerce_prediction.ipynb`.
3. Model dan artifact akan disimpan di folder `models/`.
4. Jalankan tests untuk memvalidasi modul:
   ```bash
   pytest tests/ -q
   ```
