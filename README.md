# 🕳️ Pothole Segmentation using MANet

![Last Updated](https://img.shields.io/badge/Last%20Updated-February%202026-blue)
![Python](https://img.shields.io/badge/Python-3.12-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF)
![Task](https://img.shields.io/badge/Task-Semantic%20Segmentation-orange)
![Competition](https://img.shields.io/badge/Competition-Data%20Science%20ARA%207.0-purple)

> **Binary semantic segmentation untuk deteksi pothole (lubang jalan) menggunakan arsitektur MANet dengan encoder MiT-B3, dilengkapi multi-scale TTA dan top-K model averaging.**

---

## 📋 Deskripsi Proyek

Proyek ini merupakan solusi untuk kompetisi **Data Science ARA 7.0** yang berfokus pada segmentasi lubang jalan (*pothole*) dari citra. Pipeline mencakup proses **EDA**, **training**, dan **inference** secara end-to-end dalam satu notebook Kaggle.

---

## 📊 Dataset

| Properti | Detail |
|---|---|
| **Total Training Images** | 498 |
| **Total Training Masks** | 498 |
| **Empty Mask** | 0 (semua gambar mengandung pothole) |
| **Foreground (Pothole)** | ~8.39% piksel |
| **Background** | ~91.61% piksel |
| **Resolusi** | Bervariasi (129–4608 × 200–4080 px) |
| **Format** | JPG (image), PNG (mask) |

---

## 🔍 Exploratory Data Analysis (EDA)

Notebook ini melakukan analisis menyeluruh terhadap dataset, meliputi:

- **Pothole Area Statistics** — Distribusi luas pothole (min: 0.02%, max: 67.40%, mean: 13.49%)
- **Sharpness Analysis** — Variasi ketajaman gambar (17.44 – 22,579.67)
- **Spatial Distribution Heatmap** — Probabilitas kemunculan pothole per area gambar (center-biased)
- **Object Count Analysis** — Jumlah objek pothole per gambar
- **Dimension Analysis** — Variasi dimensi gambar dalam dataset
- **Sample Visualization** — Overlay mask pada gambar training

---

## 🏗️ Arsitektur Model

| Komponen | Detail |
|---|---|
| **Arsitektur** | MANet (Multi-scale Attention Network) |
| **Encoder** | MiT-B3 (`mit_b3`) |
| **Pretrained Weights** | ImageNet |
| **Input Size** | 640 × 640 |
| **Output** | Binary mask (1 channel) |
| **Library** | `segmentation-models-pytorch` |

**Mengapa MANet?** — MANet memiliki *Attention Mechanism* yang membantu membedakan fitur jalan yang relevan dari noise, sekaligus menjaga konteks global tekstur jalan.

---

## ⚙️ Konfigurasi Training

```python
IMGSIZE     = 640
BATCH_SIZE  = 4
EPOCHS      = 25–28
LR          = 6e-5
ENCODER     = "mit_b3"
WEIGHTS     = "imagenet"
OPTIMIZER   = AdamW (weight_decay=1e-3)
SCHEDULER   = CosineAnnealingWarmRestarts (T_0=5, T_mult=2)
AMP         = GradScaler (Mixed Precision)
```

---

## 📉 Loss Function

Kombinasi weighted loss untuk menangani class imbalance:

| Loss | Bobot | Keterangan |
|---|---|---|
| **Dice Loss** | 0.35–0.60 | Overlap-based, sensitif terhadap segmentasi |
| **Tversky Loss** | 0.50 | α=0.3, β=0.7 — menekan false negative |
| **Focal Loss** | 0.15–0.40 | α=0.25/0.5, γ=2.0 — fokus pada hard examples |

> Bobot loss bervariasi antar iterasi eksperimen (Best 1, Best 2, Best 3).

---

## 🔄 Data Augmentation

Menggunakan **Albumentations** untuk mencegah overfitting:

- `HorizontalFlip`, `VerticalFlip`
- `Perspective`, `ShiftScaleRotate`
- `GridDistortion`
- `CLAHE`, `RandomGamma`, `ColorJitter`
- `RandomShadow`
- `Normalize` + `ToTensorV2`

---

## 🧪 Strategi Inference

### Multi-Scale Test Time Augmentation (TTA)
- **Skala**: `[0.75, 1.0, 1.25]` atau `[0.75, 1.0, 1.25, 1.5]`
- **Flip**: Horizontal flip (True/False)
- Prediksi di-*average* dari seluruh kombinasi skala dan flip

### Post-Processing
1. **Adaptive Thresholding** — Threshold disesuaikan berdasarkan rata-rata brightness gambar
2. **Morphological Closing** — Kernel ellipse 3×3 untuk menutup gap kecil
3. **Small Object Removal** — Filter area minimum berdasarkan dimensi gambar
4. **RLE Encoding** — Run-Length Encoding untuk format submission

### Top-K Model Averaging
- Menyimpan **top 3 model** dengan Dice Score terbaik selama training
- Bobot ketiga model di-*average* untuk prediksi yang lebih robust

---

## 📈 Hasil

| Versi | Strategi | Skor (Dice) |
|---|---|---|
| Best 1 | Single model + multi-scale TTA | 0.75973 |
| Best 2 | Advanced TTA + adaptive threshold | 0.76397 |
| Best 3 | Top-3 averaging + advanced post-processing | *Best submission* |

---

## 📁 Struktur File

```
DatSciARA7_GakKuliah.ipynb
├── EDA
│   ├── Dataset Overview
│   ├── Pothole Area Statistics
│   ├── Sharpness Analysis
│   ├── Spatial Distribution Heatmap
│   └── Object Count Analysis
├── Final Pipeline
│   ├── Best 1 (Single Model)
│   ├── Best 2 (Advanced TTA)
│   └── Best 3 (Top-K Averaging)
├── Utilities (Opsional)
└── Render Test Images
```

---

## 🛠️ Dependencies

```
torch
segmentation-models-pytorch
albumentations
opencv-python (cv2)
numpy
pandas
matplotlib
scikit-image
tqdm
```

---

## 🚀 Cara Menjalankan

1. Upload notebook ke **Kaggle**
2. Attach dataset kompetisi **Data Science ARA 7.0**
3. Aktifkan **GPU** sebagai accelerator
4. Jalankan seluruh cell secara berurutan (`Run All`)
5. File submission (`submission_manet_*.csv`) akan tersedia di output

---

## 👤 Author

**Tim GakKuliah** — Data Science ARA 7.0 Competition

---

*Last updated: February 13, 2026*
