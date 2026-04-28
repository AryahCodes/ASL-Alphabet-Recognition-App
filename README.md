#  SignApp - Real-Time ASL Alphabet Recognition

NOTE:
Live backend temporarily disabled due to TFLite deployment constraints on free hosting. So while Frontend is deployed on Vercel, it won't connect to the backend and won't work properly. 
Good news: Model runs locally and is fully implemented; conversion & serving code included.


https://signapp-frontend.vercel.app/

An interactive American Sign Language (ASL) learning application using real-time hand tracking and deep learning for letter recognition.


## 🎯 Features

- **Real-time Hand Tracking** with MediaPipe
- **ASL Alphabet Recognition** (A-Y, 24 letters)
- **96.86% Test Accuracy** using professional deep learning model
- **Interactive Learning Modes**
  - Hand Tracking: Real-time letter recognition
  - Training Mode: Collect custom training data
- **Professional ML Pipeline**
  - Z-score normalization for camera-distance invariance
  - Temporal smoothing for stable predictions
  - Frame buffering for robust recognition
  - SigNN-inspired neural network architecture

## 🚀 Tech Stack

### Frontend
- React + TypeScript
- Socket.IO for real-time communication
- MediaPipe Hands (browser-based hand tracking)
- Tailwind CSS

### Backend
- Python + Flask
- TensorFlow/Keras (Deep Learning)
- MediaPipe (Hand landmark detection)
- Socket.IO
- Eventlet (Async support)

### ML Architecture
- **Model:** Deep Neural Network (SigNN-based)
  - 900 → 400 → 200 → 24 neurons
  - Batch normalization + Dropout
  - ReLU and Tanh activations
- **Features:** 72 engineered features
  - Z-score normalized coordinates
  - Finger angles and extension ratios
  - Inter-finger spacing
  - Hand direction and palm size
- **Accuracy:** 96.86% on test set (9,572 samples)

## 📊 Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 96.86% |
| Real-time FPS | 10 FPS |
| Confidence (avg) | 65-75% |
| Letters Supported | 24 (A-Y) |
| Training Samples | 9,572 |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- Webcam

### Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

## 🎮 Usage

### Start Backend
```bash
cd backend
source venv/bin/activate
python server.py
```

### Start Frontend
```bash
cd frontend
npm start
```

Visit `http://localhost:3000`

## 🧠 How It Works

### 1. Hand Tracking
MediaPipe detects 21 hand landmarks in real-time from webcam feed

### 2. Feature Extraction
- Extract 72 features from landmarks
- Apply z-score normalization
- Calculate finger angles and extensions

### 3. Prediction Pipeline
```
Raw Frame → MediaPipe → Landmarks → Feature Extraction → 
Z-Score Norm → Frame Buffer (10 frames) → Model Prediction → 
Temporal Smoothing (7 frames) → Confidence Threshold (50%) → Display
```

### 4. Model Architecture
```python
Input (72 features)
    ↓
Dense(900) + BatchNorm + Dropout(0.15)
    ↓
Dense(400) + BatchNorm + Dropout(0.25)
    ↓
Dense(200) + Dropout(0.4)
    ↓
Dense(24) + Softmax
```

## 📚 Training Your Own Model

### Using Kaggle Data (Recommended)
```bash
cd backend
python train_professional_kaggle.py
```

### Using Custom Data
1. Go to Training Mode in the app
2. Collect 30-50 samples per letter
3. Click "Train Model"

## 🎨 Project Structure
```
SignApp/
├── backend/
│   ├── server.py                      # Flask server + Socket.IO
│   ├── hand_processor.py              # MediaPipe hand tracking
│   ├── feature_extractor.py           # Feature engineering
│   ├── professional_letter_classifier.py  # TensorFlow model
│   ├── train_professional_kaggle.py   # Training script
│   └── models/
│       ├── professional_model.h5      # Trained model
│       └── professional_labels.json   # Label mappings
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── HandTracking.jsx       # Main recognition UI
│   │   │   └── TrainingMode.jsx       # Data collection UI
│   │   └── App.jsx
│   └── package.json
└── README.md
```

## 🔬 Technical Deep Dive

### Z-Score Normalization
Removes camera distance variance by normalizing landmark coordinates:
```python
x_normalized = (x - mean(x)) / std(x)
```

### Temporal Smoothing
Uses sliding window (7 frames) to require consistent predictions:
- Letter must appear in 40%+ of window
- Confidence must average > 50%

### Frame Buffering
Averages landmarks over 10 frames before prediction for stability

## 🐛 Known Issues

- Letters D, M, N, G have lower accuracy (~60-70%)
  - These letters have very similar hand shapes
  - Industry-wide challenge in ASL recognition
- Dynamic letters J and Z not supported (require temporal LSTM)

## 🚀 Future Enhancements

- [ ] Word mode (string multiple letters)
- [ ] Phrase recognition
- [ ] LSTM for dynamic gestures (J, Z)
- [ ] Multi-hand support
- [ ] Mobile app deployment
- [ ] User progress tracking

## 📖 References

- [SigNN Research Paper](https://github.com/AriAlavi/SigNN)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- [FreeCodeCamp ASL Tutorial](https://www.freecodecamp.org/news/create-a-real-time-gesture-to-text-translator/)

## 👨‍💻 Author

[Aryahvishwa Babu](https://github.com/AryahCodes)

## 📄 License

MIT License

## Running Tests

```bash
cd backend
pip install pytest flask flask-socketio flask-cors numpy scikit-learn
pytest tests/ -v
```

The test suite (60 tests) runs without a GPU, webcam, or model file. Heavy dependencies (mediapipe, tensorflow, cv2) are stubbed at import time.

## Metrics

While the server is running, call:

```bash
curl http://localhost:5001/metrics
```

Returns live counters for frames received/processed/dropped, active client count, and inference latency (mean, median, min, max, p95 over the last 200 inferences). See `docs/benchmarking.md` for field definitions and derived metrics.

## Benchmarking

### Direct inference benchmark (no server or webcam needed)

Times `FeatureExtractor.extract_features()` + `model.predict()` directly on saved landmark data:

```bash
python run_real_benchmark.py --n 1000
```

Measured results (3 independent 1,000-sample runs on M-series Mac CPU):

| Metric | Run 1 | Run 2 | Run 3 |
|--------|-------|-------|-------|
| Processed FPS | 48.3 | 48.0 | 47.5 |
| Median latency (ms) | 20.1 | 20.1 | 20.2 |
| P95 latency (ms) | 22.7 | 22.3 | 23.6 |
| Failure rate | 0% | 0% | 0% |

> These are **direct inference** numbers only — no MediaPipe image detection, SocketIO, or frame buffering overhead. Full end-to-end latency will be higher.

Results are saved to `benchmark_results/direct_inference_<timestamp>.json`.

### Full pipeline benchmark (requires running server)

Run the benchmark client against a running server:

```bash
python benchmark_client.py --frames 200 --url http://localhost:5001
```

Results are printed to stdout and by default saved as a timestamped JSON report:

```
benchmark_results/benchmark_20260427T120000Z.json
```

Additional options:

```bash
# Disable saving
python benchmark_client.py --no-save

# Annotate the run with environment notes
python benchmark_client.py --env "M2 MacBook, Python 3.11, gunicorn"
```

Reports include all `/metrics` counters, `processed_fps`, `failure_rate_pct`, and full latency stats (mean, median, min, max, p95). JSON files are excluded from git. See [docs/benchmarking.md](docs/benchmarking.md) for full field reference.

## Smoothing Ablation

Compare the effect of different temporal smoothing window sizes on prediction stability without needing a webcam, model, or server:

```bash
python smoothing_ablation.py
```

Prints a table of `stable_pct`, `flicker_rate`, `changes_per_second`, and `avg_confidence` for these configurations:

| Config | Notes |
|--------|-------|
| `none (raw)` | Raw pass-through above confidence threshold |
| `window=3` | Requires unanimous agreement (very conservative) |
| `window=5` | Moderate smoothing |
| `window=7 (current)` | App default — 40% majority of 7 frames |
| `window=10` | More smoothing, higher latency |

Results are saved to `eval_results/` as a JSON file. No dependencies beyond the standard library.

## Live Demo Testing

Use `docs/demo_testing.md` to record and accumulate repeated live webcam sessions.

**Workflow:**
1. Start the backend and open the frontend in a browser.
2. Run a live session (60–120 seconds) with real hand signs.
3. Run `python benchmark_client.py --env "your machine"` to capture `/metrics` into a timestamped JSON report.
4. Fill in one row of the table in [docs/demo_testing.md](docs/demo_testing.md) per session.

After >= 3 sessions you can cite median/p95 latency and processed FPS as measured values. After >= 5 sessions you can cite stability and flicker behavior.

## Running with Docker

Build the backend image (from repo root):

```bash
docker build -t signapp-backend ./backend
```

Run with model files volume-mounted at runtime:

```bash
docker run -p 5001:5001 \
  -v $(pwd)/backend/models:/app/models \
  signapp-backend
```

The `-v` mount is required because model artifacts are excluded from the image (see `.dockerignore`).

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5001` | Override the listen port via gunicorn `-b` flag |

To bake model files into the image instead of mounting, remove the `models/*.h5`, `models/*.tflite`, and `models/*.pkl` lines from `.dockerignore`.

## What Is Now Measurable

| Claim | Status | How to verify |
|-------|--------|---------------|
| Inference latency (mean, median, p95) | Measurable | Run `benchmark_client.py` during a live webcam session |
| Throughput (processed FPS) | Measurable | Read `processed_fps` from saved JSON report |
| Failure rate | Measurable | Read `failure_rate_pct` from saved JSON report |
| Smoothing flicker reduction | Measurable (synthetic) | Run `smoothing_ablation.py` |
| End-to-end demo stability | Requires real sessions | Follow `docs/demo_testing.md` |
| Model accuracy (96.86%) | From training evaluation | See training script output |

> Only cite numbers from your own runs. Synthetic benchmark frames (blank images) produce no real latency data — hands must be in frame for inference to occur.

## 🙏 Acknowledgments

- Kaggle ASL Alphabet Dataset
- SigNN Research Team
- MediaPipe Team
- FreeCodeCamp Community
