# Lung Sound Classification

**DenseNet121 + BiLSTM (Mel Spectrogram Based)**

---

## Project Overview

This project implements an AI-based Lung Sound Classification system using:

- **Mel Spectrogram** feature extraction
- **DenseNet121** (CNN) for spatial feature learning
- **BiLSTM** for temporal modeling
- **Focal Loss** for class imbalance handling
- **Gradio Web Interface** for live inference

The system classifies lung sounds into **6 respiratory conditions**:

| Class | Description |
|-------|-------------|
| Asthma | Chronic inflammatory disease of airways |
| COPD | Chronic Obstructive Pulmonary Disease |
| Heart Failure | Fluid in lungs due to heart dysfunction |
| Lung Fibrosis | Scarring of lung tissue |
| Normal | Healthy lung sounds |
| Pneumonia | Lung infection with inflammation |

> **Disclaimer:** This is an educational student project and **not** a medical device.

---

## Model Architecture

```
Audio (.wav)
     ↓
Resample to 22,050 Hz + Peak Normalize
     ↓
Split into 5 segments (2s each)
     ↓
Mel Spectrogram (128 mels, n_fft=2048, hop_length=512)
     ↓
Resize to 224×224 + ImageNet Normalization
     ↓
DenseNet121 (Feature Extractor) → 1024-dim features
     ↓
BiLSTM (128 hidden, bidirectional) → Temporal Learning
     ↓
Dropout (0.5) + Fully Connected Layer
     ↓
Softmax (6 classes)
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Sample Rate | 22,050 Hz |
| Segment Duration | 2.0 seconds |
| Number of Segments | 5 |
| Mel Bins | 128 |
| FFT Size | 2048 |
| Hop Length | 512 |
| Image Size | 224×224 |
| LSTM Hidden Units | 128 |
| LSTM Layers | 1 |
| Bidirectional | Yes |
| Dropout | 0.5 |

---

## Project Structure

```
lung_sound_project/
├── app/
│   ├── gradio_app.py          # Main web UI (uses src modules)
│   └── all_gradio_app.py      # Standalone web UI (self-contained)
│
├── src/
│   ├── __init__.py
│   ├── audio_preprocess.py    # Audio loading, padding, segmentation
│   ├── config.py              # Configuration and paths
│   ├── dataset.py             # PyTorch Dataset class
│   ├── features.py            # Mel spectrogram + visualization
│   ├── inference.py           # Model loading + prediction
│   ├── label_map.py           # Label encoding utilities
│   └── model.py               # DenseNet121 + BiLSTM model
│
├── data/
│   ├── raw/
│   │   ├── icbhi/             # ICBHI 2017 dataset
│   │   └── fraiwan/           # Fraiwan dataset
│   └── processed/
│       └── manifests/         # Train/Val/Test CSV splits
│
├── models/
│   ├── best_model.pth         # Trained model checkpoint
│   ├── best_model_focal.pth   # Focal loss trained checkpoint
│   ├── config.json            # Model configuration
│   ├── config_focal.json      # Focal loss configuration
│   ├── label_to_id.json       # Label → ID mapping
│   └── id_to_label.json       # ID → Label mapping
│
├── notebooks/
│   ├── 01_build_manifest.ipynb    # Data preparation
│   ├── 02_00_train_model.ipynb    # Initial training
│   ├── 02_01_train_model_v2.ipynb # Training v2
│   ├── 03_evaluate.ipynb          # Evaluation & analysis
│   ├── 04_train_focal_loss.ipynb  # Focal loss training
│   └── 05_samplea.ipynb           # Sample analysis
│
├── reports/
│   └── test_metrics.json      # Evaluation metrics
│
├── requirements.txt
└── README.md
```

---

## Datasets

### 1. ICBHI 2017 Respiratory Sound Database

- **Source:** [Kaggle - vbookshelf/respiratory-sound-database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)
- **Recordings:** 920 audio files
- **Labels:** Patient-level diagnosis (Healthy, Asthma, COPD, Pneumonia, etc.)
- **Format:** WAV files with annotation TXT files

### 2. Fraiwan Lung Sound Dataset

- **Source:** [Mendeley Data](https://data.mendeley.com/)
- **Recordings:** 336 audio files
- **Labels:** Includes Heart Failure & Lung Fibrosis (not in ICBHI)
- **Format:** WAV files with Excel annotations

### Dataset Preparation

The datasets are merged into a unified manifest with patient-wise splits to prevent data leakage:

| Split | Purpose |
|-------|---------|
| `train.csv` | Model training (~70%) |
| `val.csv` | Hyperparameter tuning (~15%) |
| `test.csv` | Final evaluation (~15%) |

---

## Preprocessing Pipeline

1. **Load Audio:** Load WAV file as mono at 22,050 Hz
2. **Normalize:** Peak normalization to [-1, 1]
3. **Segment:** Center crop/pad to 10 seconds, split into 5 × 2-second segments
4. **Mel Spectrogram:** Convert each segment to mel spectrogram (128 bins)
5. **Resize:** Interpolate to 224×224 for DenseNet
6. **3-Channel:** Duplicate single channel to RGB (3×224×224)
7. **ImageNet Norm:** Normalize with ImageNet mean/std

---

## Training Strategy

### Phase 1: Head Training (10 epochs)
- Freeze DenseNet121 backbone
- Train BiLSTM + classifier only
- Learning Rate: 1e-3
- Uses focal loss with per-class alpha weights

### Phase 2: Fine-tuning (8 epochs)
- Unfreeze last DenseNet dense block
- Fine-tune entire model
- Learning Rate: 1e-5 (backbone), 1e-4 (head)
- Gradient clipping: 1.0

### Loss Function

**Focal Loss** handles class imbalance:

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

- γ (gamma) = 2.0 — focuses on hard examples
- α (alpha) = per-class weights (inverse frequency, clipped)

---

## Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 81.1% |
| Macro F1-Score | 0.37 |
| Weighted F1-Score | 0.81 |

> Note: Macro F1 is limited by minority classes (Heart Failure, Lung Fibrosis) with few test samples.

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training)
- FFmpeg (for audio processing)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd lung_sound_project
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download datasets** (optional, for training):
   - Place ICBHI dataset in `data/raw/icbhi/`
   - Place Fraiwan dataset in `data/raw/fraiwan/`

---

## Usage

### Run Web Application

```bash
# Using modular src (recommended)
python app/gradio_app.py

# Or standalone version
python app/all_gradio_app.py
```

The app launches at `http://localhost:7860` with options to:
- Upload a WAV file
- Record from microphone
- View prediction with probability chart

### Run from Python

```python
from src.inference import load_predictor
from src.config import BEST_MODEL_PATH, CONFIG_JSON_PATH

# Load model
predictor = load_predictor(
    ckpt_path=BEST_MODEL_PATH,
    cfg_path=CONFIG_JSON_PATH
)

# Predict
label, plot_img, table = predictor.predict_file("path/to/audio.wav")
print(f"Prediction: {label}")
```

### Training (Notebooks)

1. **Data Preparation:**
   ```
   notebooks/01_build_manifest.ipynb
   ```

2. **Model Training:**
   ```
   notebooks/02_00_train_model.ipynb      # Basic training
   notebooks/04_train_focal_loss.ipynb    # Focal loss (recommended)
   ```

3. **Evaluation:**
   ```
   notebooks/03_evaluate.ipynb
   ```

---

## File Formats

### Input Audio
- **Format:** WAV (other formats supported via librosa)
- **Sample Rate:** Any (resampled to 22,050 Hz)
- **Duration:** Any (padded/cropped to 10 seconds)
- **Channels:** Mono or Stereo (converted to mono)

### Model Checkpoint (`.pth`)
```python
{
    "state_dict": model.state_dict(),
    "labels": ["Asthma", "COPD", ...],
    "label_to_id": {"Asthma": 0, ...},
    "id_to_label": {0: "Asthma", ...},
    "config": {...}
}
```

---

## Known Limitations

1. **Class Imbalance:** Heart Failure and Lung Fibrosis have limited samples
2. **Recording Quality:** Performance depends on audio quality (noise, artifacts)
3. **Generalization:** Trained on specific datasets; may not generalize to all recording devices
4. **Not Medical Device:** For educational/research purposes only

---

## Future Improvements

- [ ] Add data augmentation (pitch shift, time stretch, noise injection)
- [ ] Implement attention mechanism for segment weighting
- [ ] Explore additional features (MFCC, chromagram)
- [ ] Collect more data for minority classes
- [ ] Cross-validation for more robust evaluation

---

## References

1. ICBHI 2017 Challenge: [https://bhichallenge.med.auth.gr/](https://bhichallenge.med.auth.gr/)
2. DenseNet: Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017)
3. Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)

---

## License

This project is for educational and research purposes only.  
**Not intended for clinical or diagnostic use.**

---

## Disclaimer

This project is an educational student project and is **NOT** a certified medical diagnostic tool. The predictions made by this model should **NOT** be used for medical diagnosis or treatment decisions. Always consult a qualified healthcare professional for medical advice.
