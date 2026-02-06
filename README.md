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
- **Features:** 78 engineered features
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
- Extract 78 features from landmarks
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
Input (78 features)
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

## 🙏 Acknowledgments

- Kaggle ASL Alphabet Dataset
- SigNN Research Team
- MediaPipe Team
- FreeCodeCamp Community
