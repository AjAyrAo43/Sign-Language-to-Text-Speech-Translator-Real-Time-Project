# 🤟 SignLiveAI — Real-Time Sign Language to Text & Speech Translator

A deep learning-powered web application that translates **American Sign Language (ASL)** into text and speech in real time. The system captures hand and body movements through a webcam, extracts skeletal landmarks using **Google MediaPipe**, and classifies them into **100 ASL words** using a custom **Transformer** model — all within the browser.

---

## ✨ Key Features

- 🎥 **Live Webcam Translation** — Real-time hand & pose landmark detection and visualization
- 🧠 **Transformer-Based Recognition** — Custom SPOTER architecture trained on 100 ASL word classes
- 🔍 **Explainable AI** — Interpretable attention weights reveal which body landmarks drive predictions
- 📊 **Top-5 Predictions** — Confidence-ranked results for every sign
- 🔊 **Text-to-Speech** — Instantly hear the translated sign using browser-native TTS
- 📝 **Sentence Builder** — Accumulate translated words into full sentences
- 📖 **Searchable Glossary** — Browse all 100 supported ASL signs

---

## 🏗️ Architecture

```
Webcam → MediaPipe (Hands + Pose) → 75 Landmarks/Frame → Recording Buffer (8-30 Frames)
    → FastAPI Backend (/api/predict) → SPOTER Transformer (PyTorch) → Top-5 Predictions
        → Web UI (Text + Confidence + Speech Output)
```

### How It Works

1. **Landmark Extraction** — MediaPipe JS SDK extracts 75 skeletal points per frame (21 left hand + 21 right hand + 33 body pose)
2. **Frame Recording** — The frontend captures 8–30 frames of landmark sequences when the user performs a sign
3. **Prediction API** — Landmark data is sent to the FastAPI backend via REST API
4. **Model Inference** — The SPOTER Transformer processes the sequence and outputs class probabilities
5. **Result Display** — The predicted word, confidence score, and top-5 alternatives are shown with optional speech output

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **ML Model** | PyTorch, nn.Transformer, Custom Decoder Layers |
| **Pose Estimation** | Google MediaPipe (Hands + Pose) |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | HTML5, CSS3, JavaScript, MediaPipe JS SDK |
| **Speech** | Web Speech API, gTTS, pyttsx3 |
| **Data Processing** | NumPy, Pandas, OpenCV, scikit-learn |

---

## 📁 Project Structure

```
├── src/
│   ├── train.py                        # Model training pipeline
│   ├── test.py                         # Evaluation & explainability analysis
│   ├── models/
│   │   ├── spoter_model_original.py    # SPOTER & SPOTERnoPE architectures
│   │   └── ExplainabTransformer.py     # Explainable Transformer variants
│   ├── preProcessing/                  # Video frame & landmark extraction
│   ├── dataloader/                     # Custom PyTorch dataset loaders
│   └── augmentations/                  # Data augmentation utilities
├── web_app/
│   ├── backend/
│   │   ├── app.py                      # FastAPI server & REST endpoints
│   │   ├── inference.py                # Model loading & prediction engine
│   │   └── gloss_labels.py             # 100-class gloss label mapping
│   ├── frontend/
│   │   ├── index.html                  # Main UI
│   │   ├── script.js                   # MediaPipe + camera + API logic
│   │   └── style.css                   # Glassmorphism UI design
│   └── run.py                          # App entry point
├── data/                               # Dataset resources
└── out-checkpoints/                    # Trained model weights
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Conda or virtualenv
- Webcam

### Installation

```bash
# Clone the repo
git clone https://github.com/AjAyrAo43/Sign-Language-to-Text-Speech-Translator-Real-Time-Project.git
cd Sign-Language-to-Text-Speech-Translator-Real-Time-Project

# Create environment & install dependencies
conda create -n sign python=3.11 -y
conda activate sign
pip install -r requirements_models.txt
pip install fastapi uvicorn
```

### Run the Web App

```bash
python web_app/run.py
```

Open **http://localhost:8000** in your browser → Start Camera → Perform a sign → See the translation!

---

## 📊 Model Performance

| Model Variant | Dataset | Test Accuracy |
|---|---|---|
| SPOTER + Positional Encoding | WLASL100 | 60.08% |
| SPOTER (No Positional Encoding) | WLASL100 | 60.08% |
| **Explainable Transformer + Class Query** | **WLASL100** | **62.79%** |
| Explainable Transformer + Class Query | IPNHand | 86.10% |

### Training Configuration

- **Epochs**: 100 | **Learning Rate**: 0.001 | **Optimizer**: Adam
- **Hidden Dim**: 150 | **Attention Heads**: 10 | **Encoder/Decoder Layers**: 6+6
- **Input**: 75 landmarks × 2 coordinates = 150-dim feature vector per frame

---

## 🧠 Model Details

### SPOTER (Sign POse-based TransformER)
A Transformer encoder-decoder that takes flattened landmark coordinates as input. The decoder uses a **Class Query** token (learnable parameter) and a **custom decoder layer** with self-attention removed for efficiency.

### Explainable Transformer (Extended Architecture)
Adds **learnable encoder weights** (`wEnc`) that perform element-wise multiplication with inputs, acting as **feature importance scores** — highlighting which hand/body landmarks are most important for each sign class.

---

## 📌 Dataset

- **WLASL100** — 100 word-level ASL signs from the [WLASL dataset](https://dxli94.github.io/WLASL/)
- **IPNHand** — 13-class hand gesture recognition from [IPNHand](https://gibranbenitez.github.io/IPN_Hand/)

---

## 🔮 Future Improvements

- [ ] Expand vocabulary beyond 100 signs (WLASL300/WLASL1000)
- [ ] Add continuous sign language recognition (sentence-level)
- [ ] Implement sign language video tutorials for each supported word
- [ ] Deploy as a mobile-friendly PWA
- [ ] Add multilingual speech output

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [SPOTER](https://github.com/matyasbohacek/spoter) — Baseline Transformer architecture
- [MediaPipe](https://google.github.io/mediapipe/) — Hand and pose landmark detection
- [WLASL](https://dxli94.github.io/WLASL/) — ASL video dataset
