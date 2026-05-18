# Testing Guide

## Tujuan

Panduan ini menjelaskan cara menjalankan unit test untuk memverifikasi modul preprocessing, feature engineering, model training, dan model persistence.

## Prasyarat

Pastikan dependency terpasang:

```bash
pip install -r requirements.txt
```

## Menjalankan Semua Test

Jalankan perintah berikut dari root project:

```bash
pytest tests/ -q
```

## Tes yang Dibuat

- `tests/test_data_preprocessing.py`
  - Memvalidasi imputasi missing value
  - Membersihkan kategori tidak konsisten
  - Menghapus outlier dengan metode Z-score
- `tests/test_feature_engineering.py`
  - Memastikan label encoding dan one-hot encoding berjalan benar
  - Mengekstrak fitur baru seperti `CustomerCategory` dan `SatisfactionCategory`
  - Memeriksa normalisasi dan standardisasi fitur
- `tests/test_model_trainer.py`
  - Memvalidasi pembagian data train/test
  - Memeriksa penanganan class imbalance dengan oversampling
  - Melatih model Random Forest dan menghitung metrik evaluasi
  - Mengambil feature importance
- `tests/test_artifact_manager.py`
  - Menyimpan/ memuat model dan scaler
  - Menyimpan serta memuat daftar fitur dan metadata
  - Memastikan semua artifact tercatat di folder `models/`

## Catatan

- Uji coba ini tidak memerlukan file data eksternal karena menggunakan data sintetis.
- Jika ingin menjalankan notebook end-to-end, gunakan `python -m nbconvert --to notebook --execute --inplace notebook/ecommerce_prediction.ipynb`.
