# Prediksi Prolonged Length of Stay (LOS) di IGD
### *Clinician-Guided Feature Engineering & Explainable Machine Learning untuk Sistem IGD Rumah Sakit Indonesia*

---

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green?logo=lightgbm)
![SHAP](https://img.shields.io/badge/XAI-SHAP-orange)
![Status](https://img.shields.io/badge/Status-Research-yellow)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## Tentang Penelitian

Penelitian ini mengembangkan model prediksi **Prolonged Length of Stay (LOS)** pasien di **Instalasi Gawat Darurat (IGD) RSUD dr. Soedono Madiun** menggunakan data tahun 2025. Model dirancang untuk mendukung pengambilan keputusan klinis sejak dini, membantu tenaga medis mengidentifikasi pasien berisiko tinggi perpanjangan rawat sebelum kondisi memburuk atau sumber daya terlambat dialokasikan.
Berbeda dari studi sejenis yang hanya menggunakan fitur rekam medis mentah, penelitian ini memperkenalkan **clinician-guided feature engineering** yaitu rekayasa fitur interaksi yang dirancang berdasarkan hipotesis klinis domain IGD. Selain itu dalam penelitian ini juga dilakukan **studi ablasi variabel administratif** (jenis asuransi/kd_customer) untuk mengevaluasi kontribusi dan potensi bias sistematis dalam model.

## Tujuan Penelitian

- Membangun model prediksi biner prolonged LOS di IGD menggunakan data rekam medis nyata
- Menguji apakah rekayasa fitur interaksi yang dirancang berdasarkan hipotesis klinis yang secara eksplisit memodelkan hubungan antara kondisi klinis, beban operasional, dan profil pasien dapat meningkatkan performa prediksi dan utilitas klinis model dibandingkan dengan model baseline yang hanya menggunakan fitur struktural.
- Mengevaluasi apakah kondisi klinis tertentu (kardiak, infeksi, trauma, respirasi) menghasilkan pola LOS yang berbeda secara signifikan ketika berinteraksi dengan tingkat kepadatan IGD
- Melakukan studi ablasi sistematis terhadap variabel asuransi (kd_customer: BPJS vs Umum) melalui tiga skenario model, Full Model, No Insurance, dan Insurance Only untuk mengukur kontribusi prediktif sekaligus mendeteksi potensi bias administratif yang dapat mempengaruhi keadilan model (algorithmic fairness)
- Menilai apakah performa model tetap stabil ketika divalidasi menggunakan temporal split (train: data awal–tengah 2025, test: data akhir 2025), sebagai pendekatan yang lebih mencerminkan skenario deployment nyata di mana model dilatih dengan data historis dan digunakan untuk memprediksi pasien baru

## Dataset

| Atribut | Detail |
|---|---|
| **Sumber** | IGD RSUD dr. Soedono Madiun |
| **Periode** | Januari – Desember 2025 |
| **Jumlah kunjungan** | 25.126 kunjungan |
| **Jumlah pasien unik** | 16.878 pasien |
| **Distribusi kelas** | ~80.96% non-prolonged / ~19.04% prolonged LOS |
| **Fitur awal** | 13 kolom (demografis, tanda vital, diagnosis, tindakan, asuransi) |
| **Fitur setelah engineering** | 78 fitur (Model A) / 94 fitur (Model B) |

> ⚠️ **Catatan Privasi:** Dataset tidak dipublikasikan karena mengandung data rekam medis yang dilindungi. Identitas pasien telah di-pseudonymisasi menggunakan SHA-256 sebelum pemrosesan.

## Metodologi

### Alur Kerja

```
Data Mentah IGD
      │
      ▼
1. Exploratory Data Analysis (EDA)
      │
      ▼
2. Preprocessing
   ├── Handling missing values (median / most frequent imputation)
   ├── Outlier detection & cleaning (IQR Method)
   ├── Pseudonymisasi kd_pasien (SHA-256)
   ├── Parsing tanda vital: tensi → SBP/DBP, GCS numerik
   └── Encoding: jenis kelamin, asuransi, waktu masuk → jam & hari
      │
      ▼
3. Feature Engineering
   ├── Model A: Fitur Struktural
   │   ├── Fitur Operasional IGD (load_4h, shift, jam_peak_flag, weekend_flag)
   │   ├── Fitur Historis Pasien (visit_count_prev, avg_los_prev, max_los_prev)
   │   ├── Fitur Diagnosis (dx_cardiac, dx_infeksi, dx_trauma, ... 17 kategori ICD)
   │   ├── Fitur Top Diagnosis (pd_R10.4, pd_K30, pd_N18.5, ... 25 diagnosa)
   │   └── Fitur Tindakan (tind_operasi, tind_imaging, tind_abx_iv, ... 9 kategori)
   │
   └── Model B: + Clinician-Guided Interaction Features (16 fitur tambahan)
       ├── Operasional × Klinis    : cardiac_load, trauma_night, respir_peak, infeksi_load
       ├── Demografis × Klinis     : elderly_cardiac, frequent_load
       ├── Historis × Klinis       : prev_los_load, ckd_repeat
       ├── Diagnosis × Kompleksitas: cardiac_complex, digest_lab
       ├── Diagnosis × Tanda Vital : cardiac_tachy, trauma_low_gcs, infeksi_fever
       └── Diagnosis × Prosedur    : cardiac_major_proc, trauma_imaging, infeksi_abx
      │
      ▼
4. Penanganan Imbalanced Data
   └── SMOTE (diterapkan hanya pada training set)
      │
      ▼
5. Pemodelan & Evaluasi
   ├── Baseline: Logistic Regression (class_weight='balanced')
   ├── Model Utama: LightGBM
   ├── Hyperparameter Tuning: Optuna (30 trials, 5-fold StratifiedKFold)
   └── Threshold Optimization: iterasi 0.25 – 0.50 → threshold 0.30
      │
      ▼
6. Explainability
   └── SHAP (SHapley Additive exPlanations)
      │
      ▼
7. Ablation Study
   ├── Full Model (semua fitur)
   ├── No Insurance (tanpa kd_customer)
   └── Insurance Only (kd_customer saja)
```

## Hasil Utama

### Perbandingan Model

| Model | Algoritma | ROC-AUC | Recall (LOS Tinggi) | Precision (LOS Tinggi) | Accuracy |
|---|---|---|---|---|---|
| Baseline A | Logistic Regression | 0.747 | 0.94 | 0.25 | 0.45 |
| Baseline B | Logistic Regression | 0.747 | 0.94 | 0.25 | 0.45 |
| **Model A** | **LightGBM + Optuna** | **0.773** | **0.59** | **0.39** | **0.74** |
| Model B | LightGBM + Optuna | 0.770 | 0.58 | 0.38 | 0.74 |

> *Threshold keputusan = 0.30 untuk semua model (dipilih berdasarkan optimasi recall-accuracy)*

### Studi Ablasi Variabel Asuransi

| Skenario | ROC-AUC | Recall | Precision | Accuracy |
|---|---|---|---|---|
| Full Model | 0.773 | 0.59 | 0.39 | 0.74 |
| No Insurance | 0.753 | 0.55 | 0.39 | 0.75 |
| Insurance Only | 0.655 | 0.89 | 0.27 | 0.51 |

**Temuan kunci:** `kd_customer` (jenis asuransi) memiliki daya prediktif kuat secara individual, namun tidak cukup untuk membangun model yang stabil. Model tidak sepenuhnya bergantung pada variabel administratif ini — fitur klinis dan operasional tetap berkontribusi signifikan.

## 🔑 Kebaruan Penelitian

Dibandingkan dengan studi prediksi ED LOS yang sudah ada, penelitian ini berkontribusi pada tiga aspek:

**1. Clinician-Guided Feature Engineering**
Penelitian ini secara eksplisit merancang 16 fitur interaksi berdasarkan hipotesis klinis — bukan sekadar menggunakan fitur mentah. Contoh: `cardiac_load` (pasien jantung saat IGD padat), `trauma_low_gcs` (trauma dengan kesadaran menurun), dan `elderly_cardiac` (lansia dengan penyakit jantung). Pendekatan ini berbeda dari mayoritas studi yang hanya menggunakan encoding diagnosis standar.

**2. Ablation Study Variabel Administratif**
Studi ablasi sistematis terhadap variabel asuransi (BPJS vs Umum) melalui tiga skenario model untuk mengevaluasi kontribusi, ketergantungan, dan potensi bias administratif. Pendekatan ini belum dilaporkan dalam literatur prediksi ED LOS sejenis.

**3. Konteks LMIC / Indonesia**
Penelitian ini menggunakan data nyata dari rumah sakit pemerintah Indonesia dengan sistem asuransi nasional BPJS — konteks yang sangat kurang terwakili dalam literatur prediksi ED LOS yang didominasi data dari negara maju.


## 📚 Referensi Utama

- Wang et al. (2025). LightGBM-based prediction of prolonged ED LOS. *PMC12093424*
- Sulaiman et al. (2025). Explainable ED LOS prediction with rule extraction. *Frontiers in Digital Health. PMC11861435*
- Zeleke et al. (2023). Prolonged LOS prediction from ED data. *Frontiers in Artificial Intelligence. PMC10426288*
- Wong et al. (2025). Prolonged LOS prediction in Emergency Medicine Ward. *Hong Kong Journal of Emergency Medicine*
- Gill et al. (2023). LightGBM for acute clinical deterioration with incremental feature sets. *Scientific Reports. PMC10442440*


## 👤 Penulis

**Noriandini Salyasari**
RSUD dr. Soedono Madiun — Pranata Komputer Ahli Pertama

## 📋 Lisensi & Etika

Penelitian ini menggunakan data rekam medis nyata yang telah mendapatkan **izin institusional** dari RSUD dr. Soedono Madiun. Semua identitas pasien telah di-pseudonymisasi sebelum analisis. Dataset tidak didistribusikan secara publik sesuai regulasi privasi data kesehatan yang berlaku.

*README ini dibuat sebagai bagian dari dokumentasi penelitian menuju publikasi ilmiah.*
