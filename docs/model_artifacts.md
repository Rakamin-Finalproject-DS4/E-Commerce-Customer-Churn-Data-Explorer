# Model Artifacts

## Tujuan

Dokumentasi ini menjelaskan artifact yang dihasilkan oleh pipeline model dan bagaimana menggunakannya untuk inferensi.

## Lokasi

Semua artifact disimpan di dalam folder `models/`.

## Artifact Utama

- `model_rf.pkl`
  - Random Forest classifier terlatih.
- `model_xgb.pkl`
  - XGBoost classifier terlatih.
- `preprocessor.pkl`
  - `ColumnTransformer` yang berisi scaler numerik dan one-hot encoder untuk fitur kategorikal.
- `scaler.pkl`
  - `StandardScaler` yang digunakan pada fitur numerik.

## Cara Memuat Artifact

Contoh pemuatan menggunakan `joblib`:

```python
import joblib

model = joblib.load('models/model_xgb.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')
scaler = joblib.load('models/scaler.pkl')
```

## Catatan

- `preprocessor.pkl` penting ketika model digunakan untuk inferensi pada data baru karena memastikan proses transformasi fitur identik dengan data pelatihan.
- `scaler.pkl` dapat digunakan untuk memeriksa atau menerapkan skala yang sama pada kolom numerik saat membuat fitur baru.
- Jika artifact model di-update, simpan kembali semua berkas pendukung agar format input tidak berubah.
