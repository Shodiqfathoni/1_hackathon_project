# 🏭 Prediksi Emisi Tambang dan estimasi kebutuhan Carbon Credit  
Menggunakan Machine Learning (Python / Scikit-learn)

## 📌 1. Deskripsi Project
Project ini bertujuan untuk **memprediksi emisi gas rumah kaca (CO₂e) pada kegiatan pertambangan** menggunakan model *machine learning*. Dengan prediksi emisi yang akurat, perusahaan tambang dapat:
- Mengestimasi **emisi di masa depan**
- Mengetahui apakah **emisi berpotensi melebihi batas regulasi**
- Mengambil keputusan:
  - Membeli **Carbon Credit** (jika emisi tinggi)
  - Melakukan **optimasi operasional** agar emisi turun
  - Perencanaan lingkungan jangka panjang

## 🍃 2. Apa Itu Carbon Credit?
**Carbon credit** adalah izin yang mewakili hak untuk mengeluarkan *1 ton CO₂e*.  
Jika perusahaan menghasilkan emisi melebihi batas, mereka **harus membeli carbon credit**.

Dengan model prediksi ini, perusahaan dapat:
- Memperkirakan jumlah emisi tahun depan
- Menghitung potensi carbon credit yang dibutuhkan
- Mengoptimalkan produksi supaya tidak melebihi batas
- Mengurangi biaya kepatuhan terhadap regulasi karbon

## 🎯 3. Tujuan Project
- Memprediksi emisi CO₂e berdasarkan data operasional tambang  
- Membantu strategi pembelian carbon credit  
- Mengoptimalkan energi & produksi  

## 🧠 4. Model Machine Learning
Pipeline mencakup:
- Median Imputation
- Robust Scaling
- OneHot Encoding
- Model:
  - Ridge
  - ElasticNet
  - HuberRegressor
  - HistGradientBoosting
  - **Stacking Regressor** (terbaik)

## 📂 5. Struktur Project
```
project-hackathon/
 ├── data/
 │   ├── raw/
 │   └── processed/
 ├── src/
 │   ├── train.py
 │   ├── preprocessing.py
 │   ├── data_loader.py
 │   ├── model_builder.py
 │   ├── evaluate.py
 │   └── utils.py
 ├── models/
 ├── outputs/
 ├── README.md
 └── requirements.txt
```

## ⚙️ 6. Instalasi & Setup

### 1. Buat Conda Environment
```
buka terminal/cmd di pc kamu
ketik cd -ganti dengan file path penyimpanan folder project hackathon-
conda create -n emisi python=3.10 -y
conda activate emisi
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

## 🚀 7. Cara Menjalankan Model
```
buka vscode 
buka file dan open folder project hackathon
pilih file train.py
tekan ctrl+shift+p lalu cari, Python: Select Interpreter
pilih environtment yang baru saja kamu buat
```

Pastikan berada di folder root:

Jalankan diterminal:

```
python -m src.train --data_path data/raw/dataset_emisi_tambang_5000_hackthon.csv --output_dir outputs
```

## 📊 8. Output
- Model terlatih (.pkl)
- Evaluasi (MAE, RMSE, R²)
- Prediksi emisi

## 🌱 9. Manfaat
### Untuk Perusahaan Tambang:
- Mengetahui potensi kelebihan emisi  
- Strategi pembelian carbon credit  
- Kontrol biaya energi  
- Memenuhi regulasi ESG

### Untuk Regulator:
- Monitoring tren emisi  
- Penyusunan kebijakan berbasis data  

### Untuk Lingkungan:
- Mengurangi gas rumah kaca  
- Efisiensi energi  

## 🧩 10. Alur Kerja
1. Load data  
2. Preprocessing  
3. Train model  
4. Evaluasi  
5. Simpan hasil  
6. Prediksi emisi masa depan
