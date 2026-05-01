# 🫁 BreathSense

**AI-based Lung Sound Classification using DenseNet121 + BiLSTM**

---

## 🚀 Overview

BreathSense is a deep learning system that classifies lung sounds into **6 respiratory conditions** using Mel Spectrograms and a hybrid **CNN + RNN architecture**.
It combines spatial feature extraction with temporal modeling for improved accuracy on respiratory audio signals.

---

## 🧠 Key Features

* 🔍 **DenseNet121** for spatial feature extraction
* 🔁 **BiLSTM** for temporal sequence learning
* ⚖️ **Focal Loss** to handle class imbalance
* 🎧 **Mel Spectrogram-based audio processing**
* 🌐 **Gradio Web App** for real-time inference

---

## 🏷️ Classes

* Asthma
* COPD
* Heart Failure
* Lung Fibrosis
* Normal
* Pneumonia

---

## 📊 Performance

* **Accuracy:** 81.1%
* **Weighted F1-score:** 0.81
* **Macro F1-score:** 0.37 *(affected by minority classes)*

---

## 🧱 Model Architecture

```
Audio (.wav)
   ↓
Resample + Normalize
   ↓
Segmentation (5 × 2s)
   ↓
Mel Spectrogram (128 bins)
   ↓
Resize (224×224)
   ↓
DenseNet121 (CNN)
   ↓
BiLSTM (Temporal Learning)
   ↓
Fully Connected Layer
   ↓
Softmax (6 classes)
```

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Librosa
* NumPy / Pandas
* Gradio

---

## 📂 Project Structure

```
BreathSense/
├── app/                # Gradio UI
├── src/                # Core ML pipeline
├── notebooks/          # Training & experiments
├── reports/            # Evaluation metrics
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

Datasets used:

* ICBHI 2017 Respiratory Sound Database
* Fraiwan Lung Sound Dataset

⚠️ Dataset not included due to size


---

## 🛠️ Installation

```bash
git clone https://github.com/Tejaswani26/BreathSense.git
cd BreathSense

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## ▶️ Usage

Run the web app:

```bash
python app/gradio_app.py
```

Then open:

```
http://localhost:7860
```

---

## 🧪 Training (Optional)

* Data preparation → `notebooks/01_build_manifest.ipynb`
* Model training → `notebooks/02_00_train_model.ipynb`
* Evaluation → `notebooks/03_evaluate.ipynb`

---

## ⚠️ Limitations

* Class imbalance affects minority class performance
* Sensitive to recording quality
* Limited generalization across devices

---

## 🚀 Future Improvements

* Data augmentation (noise, pitch shift)
* Attention mechanisms
* More balanced dataset
* Cross-validation

---

## 📜 Disclaimer

This is an educational project and **not a medical diagnostic tool**.
Do not use for clinical decisions.

---
