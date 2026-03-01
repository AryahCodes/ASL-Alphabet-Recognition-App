import eventlet
eventlet.monkey_patch()  # ✅ must be first before importing Flask / SocketIO

from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from hand_processor import HandProcessor
from data_collector import DataCollector
from collections import deque, Counter
import numpy as np
import gc

# -------------------------------------------------
# Frame Buffer Class (TIER 2)
# -------------------------------------------------
class FrameBuffer:
    """Accumulates frames over time for more stable predictions"""
    
    def __init__(self, buffer_size=10, min_frames=5):
        self.buffer_size = buffer_size
        self.min_frames = min_frames
        self.buffer = deque(maxlen=buffer_size)
        self.last_prediction_frame = 0
        self.frame_count = 0
    
    def add_frame(self, landmarks):
        """Add a frame's landmarks to the buffer"""
        self.buffer.append(landmarks)
        self.frame_count += 1
    
    def is_ready(self):
        """Check if we have enough frames to make a prediction"""
        return len(self.buffer) >= self.min_frames
    
    def should_predict(self):
        """
        Decide if we should make a prediction now
        Only predict every 5 frames to reduce computation
        """
        if not self.is_ready():
            return False
        
        frames_since_prediction = self.frame_count - self.last_prediction_frame
        if frames_since_prediction >= 5:  # Predict every 5 frames
            self.last_prediction_frame = self.frame_count
            return True
        return False
    
    def get_average_landmarks(self):
        """
        Average landmarks across all buffered frames
        This creates a more stable representation of the hand pose
        """
        if not self.is_ready():
            return None
        
        # Average each landmark coordinate across all frames
        avg_landmarks = []
        num_frames = len(self.buffer)
        
        for lm_idx in range(21):  # 21 landmarks per hand
            x_sum = sum(frame[lm_idx]['x'] for frame in self.buffer)
            y_sum = sum(frame[lm_idx]['y'] for frame in self.buffer)
            z_sum = sum(frame[lm_idx]['z'] for frame in self.buffer)
            
            avg_landmarks.append({
                'x': x_sum / num_frames,
                'y': y_sum / num_frames,
                'z': z_sum / num_frames
            })
        
        return avg_landmarks
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()

# -------------------------------------------------
# Prediction Smoother Class (TIER 1)
# -------------------------------------------------
class PredictionSmoother:
    """Smooths predictions over time to reduce jitter and improve accuracy"""
    
    def __init__(self, window_size=7, confidence_threshold=0.50):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.frames_since_last_hand = 0
        
    def add_prediction(self, letter, confidence):
        """Add a new prediction to the sliding window"""
        self.predictions.append(letter)
        self.confidences.append(confidence)
        self.frames_since_last_hand = 0
        
    def no_hand_detected(self):
        """Called when no hand is detected in frame"""
        self.frames_since_last_hand += 1
        # Clear predictions if no hand for 3 frames
        if self.frames_since_last_hand >= 3:
            self.predictions.clear()
            self.confidences.clear()
    
    def get_smoothed_prediction(self):
        """
        Get the most common prediction with averaged confidence
        Returns: (letter, confidence) or (None, 0.0) if not enough data
        """
        if len(self.predictions) < 3:  # Need at least 3 predictions
            return None, 0.0
        
        # Count letter occurrences
        letter_counts = Counter(self.predictions)
        most_common_letter, count = letter_counts.most_common(1)[0]
        
        # Must appear in at least 40% of window
        if count < max(3, self.window_size * 0.4):
            return None, 0.0
        
        # Calculate average confidence for the most common letter
        matching_confidences = [
            conf for pred, conf in zip(self.predictions, self.confidences)
            if pred == most_common_letter
        ]
        avg_confidence = sum(matching_confidences) / len(matching_confidences)
        
        # Only return if confidence above threshold
        if avg_confidence < self.confidence_threshold:
            return None, 0.0
        
        return most_common_letter, avg_confidence
    
    def reset(self):
        """Clear all predictions"""
        self.predictions.clear()
        self.confidences.clear()
        self.frames_since_last_hand = 0

# -------------------------------------------------
# App and Socket.IO setup
# -------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    max_http_buffer_size=10 * 1024 * 1024,  # 10MB for images
    ping_interval=10,
    ping_timeout=25
)

# -------------------------------------------------
# Initialize modules & Choose Model
# -------------------------------------------------
hand_processor = HandProcessor()
data_collector = DataCollector()

# ✅ MODEL SELECTION: Professional model disabled to fit in Render free tier (512 MB)
USE_PROFESSIONAL_MODEL = False

print("=" * 60)
print("🚀 Sign Language App - Backend Server")
print("=" * 60)

if USE_PROFESSIONAL_MODEL:
    try:
        from professional_letter_classifier import ProfessionalLetterClassifier
        letter_classifier = ProfessionalLetterClassifier()
        print("🧠 Using: Professional Deep Learning Model (TensorFlow/Keras)")
    except ImportError:
        print("⚠️  professional_letter_classifier.py not found!")
        print("⚠️  Falling back to RandomForest model...")
        from letter_classifier import LetterClassifier
        letter_classifier = LetterClassifier()
        print("🌲 Using: RandomForest Model (Fallback)")
else:
    from letter_classifier import LetterClassifier
    letter_classifier = LetterClassifier()
    print("🌲 Using: RandomForest Model")

# Per-client smoothers and buffers
client_smoothers = {}
client_buffers = {}

# Load model
model_loaded = letter_classifier.load_model()

if letter_classifier.is_trained:
    print(f"✅ Model loaded successfully!")
    print(f"✅ Can recognize: {sorted(letter_classifier.labels)}")
else:
    print("⚠️  No model loaded!")
    if USE_PROFESSIONAL_MODEL:
        print("⚠️  Train the professional model first:")
        print("    python train_professional_kaggle.py")
    else:
        print("⚠️  Use Training Mode to collect data and train")

print("\n✨ Active Enhancements:")
print("   • Tier 1: Temporal smoothing (reduces jitter)")
print("   • Tier 1.5: Z-score normalization (camera-distance invariant)")
print("   • Tier 2: Frame buffering (stable predictions)")
print("=" * 60)

gc.collect()

# -------------------------------------------------
# REST API endpoints
# -------------------------------------------------
@app.route("/test")
def test():
    return {"message": "Backend is working!", "status": "success"}

@app.route("/")
def home():
    model_type = "Professional Deep Learning" if USE_PROFESSIONAL_MODEL else "RandomForest"
    return {
        "message": "Sign Language App API",
        "version": "3.0 - Professional Grade",
        "model_type": model_type,
        "enhancements": ["Temporal Smoothing", "Z-Score Normalization", "Frame Buffering"]
    }

@app.route("/model/status")
def model_status():
    """Get model training status"""
    sample_counts = data_collector.get_sample_counts()
    model_type = "professional" if USE_PROFESSIONAL_MODEL else "randomforest"
    return {
        "is_trained": letter_classifier.is_trained,
        "model_type": model_type,
        "labels": sorted(letter_classifier.labels) if letter_classifier.is_trained else [],
        "sample_counts": sample_counts,
    }

# -------------------------------------------------
# Socket.IO events
# -------------------------------------------------
@socketio.on("connect")
def handle_connect():
    client_id = request.sid
    print(f"✅ Client connected! ID: {client_id}")
    
    # Create smoother and buffer for this client
    client_smoothers[client_id] = PredictionSmoother(
        window_size=7,              # Look at last 7 predictions
        confidence_threshold=0.50    # Only show predictions > 50% confidence
    )
    
    client_buffers[client_id] = FrameBuffer(
        buffer_size=10,  # Accumulate 10 frames (~1 second at 10 FPS)
        min_frames=5     # Need at least 5 frames before predicting
    )
    
    emit("response", {"message": "Connected to backend server!"})

    # Send model status immediately
    sample_counts = data_collector.get_sample_counts()
    model_type = "professional" if USE_PROFESSIONAL_MODEL else "randomforest"
    emit(
        "model_status",
        {
            "is_trained": letter_classifier.is_trained,
            "model_type": model_type,
            "labels": sorted(letter_classifier.labels)
            if letter_classifier.is_trained
            else [],
            "sample_counts": sample_counts,
        },
    )

@socketio.on("disconnect")
def handle_disconnect():
    client_id = request.sid
    print(f"❌ Client disconnected! ID: {client_id}")
    
    # Clean up smoother and buffer
    if client_id in client_smoothers:
        del client_smoothers[client_id]
    if client_id in client_buffers:
        del client_buffers[client_id]

@socketio.on("test_message")
def handle_test_message(data):
    print(f"📩 Received message: {data}")
    emit("response", {"message": f"Echo: {data}"})

@socketio.on("process_frame")
def handle_process_frame(data):
    """
    Process webcam frame with:
    - Frame buffering (Tier 2)
    - Temporal smoothing (Tier 1)
    - Z-score normalization (Tier 1.5)
    """
    try:
        client_id = request.sid
        frame_data = data.get("frame")

        # Ensure client has smoother and buffer
        if client_id not in client_smoothers:
            client_smoothers[client_id] = PredictionSmoother()
        if client_id not in client_buffers:
            client_buffers[client_id] = FrameBuffer()

        smoother = client_smoothers[client_id]
        buffer = client_buffers[client_id]

        # Process the frame
        result = hand_processor.process_frame(frame_data)

        # If hand detected, add to buffer
        if result["success"] and result["hands_detected"] > 0:
            first_hand = result["hands"][0]
            landmarks = first_hand["landmarks"]
            
            # Add frame to buffer
            buffer.add_frame(landmarks)
            
            # Only make prediction if buffer is ready and it's time
            if letter_classifier.is_trained and buffer.should_predict():
                # Get averaged landmarks from buffer
                avg_landmarks = buffer.get_average_landmarks()
                
                if avg_landmarks:
                    # Get raw prediction (will use z-score normalization internally)
                    prediction = letter_classifier.predict(avg_landmarks)
                    
                    if prediction["success"]:
                        # Add to smoother
                        smoother.add_prediction(
                            prediction["letter"],
                            prediction["confidence"]
                        )
                        
                        # Get smoothed prediction
                        smoothed_letter, smoothed_conf = smoother.get_smoothed_prediction()
                        
                        if smoothed_letter:
                            result["letter_prediction"] = {
                                "success": True,
                                "letter": smoothed_letter,
                                "confidence": smoothed_conf,
                                "raw_letter": prediction["letter"],
                                "raw_confidence": prediction["confidence"],
                                "buffer_size": len(buffer.buffer)
                            }
                            
                            print(f'👋 Smoothed: {smoothed_letter} ({smoothed_conf:.1%}) | Raw: {prediction["letter"]} ({prediction["confidence"]:.1%}) | Buffer: {len(buffer.buffer)}')
                        else:
                            # Prediction below threshold or not stable enough
                            result["letter_prediction"] = {
                                "success": False,
                                "message": "Hold steady..."
                            }
                            print(f'👋 Unstable (raw: {prediction["letter"]} {prediction["confidence"]:.1%})')
        else:
            # No hand detected - clear buffer
            buffer.clear()
            smoother.no_hand_detected()
            if result["hands_detected"] == 0:
                print("👋 No hand detected")

        # Send back to frontend
        emit("hand_landmarks", result)

    except Exception as e:
        print(f"❌ Error in process_frame: {str(e)}")
        import traceback
        traceback.print_exc()
        emit(
            "hand_landmarks",
            {"success": False, "error": str(e), "hands_detected": 0, "hands": []},
        )

@socketio.on("save_training_sample")
def handle_save_sample(data):
    """Save a training sample"""
    try:
        landmarks = data.get("landmarks")
        label = data.get("label", "").upper()

        if not landmarks or not label:
            emit("sample_saved", {"success": False, "error": "Missing landmarks or label"})
            return

        filepath = data_collector.save_sample(landmarks, label)
        sample_counts = data_collector.get_sample_counts()

        print(f"💾 Saved training sample: {label} (total: {sample_counts.get(label, 0)})")

        emit(
            "sample_saved",
            {
                "success": True,
                "label": label,
                "filepath": filepath,
                "sample_counts": sample_counts,
            },
        )

    except Exception as e:
        print(f"❌ Error saving sample: {str(e)}")
        emit("sample_saved", {"success": False, "error": str(e)})

@socketio.on("train_model")
def handle_train_model(data):
    """Train the letter classifier (only works with RandomForest model)"""
    try:
        if USE_PROFESSIONAL_MODEL:
            emit("training_complete", {
                "success": False,
                "error": "Professional model training must be done via: python train_professional_kaggle.py"
            })
            print("⚠️  Cannot train professional model from UI. Use: python train_professional_kaggle.py")
            return
        
        print("🎓 Training RandomForest model...")
        landmarks_list, labels_list = data_collector.load_all_samples()

        if len(landmarks_list) == 0:
            emit("training_complete", {"success": False, "error": "No training samples found"})
            return

        success = letter_classifier.train(landmarks_list, labels_list)

        if success:
            letter_classifier.save_model()
            
            # Reset all client smoothers and buffers after retraining
            for smoother in client_smoothers.values():
                smoother.reset()
            for buffer in client_buffers.values():
                buffer.clear()
            
            emit(
                "training_complete",
                {
                    "success": True,
                    "message": f"Model trained on {len(landmarks_list)} samples",
                    "labels": sorted(letter_classifier.labels),
                    "sample_count": len(landmarks_list),
                },
            )
            print(f"✅ Model trained successfully on {len(landmarks_list)} samples!")
            print(f"✅ Can recognize: {sorted(letter_classifier.labels)}")
        else:
            emit("training_complete", {"success": False, "error": "Training failed"})

    except Exception as e:
        print(f"❌ Error training model: {str(e)}")
        import traceback
        traceback.print_exc()
        emit("training_complete", {"success": False, "error": str(e)})

@socketio.on_error_default
def default_error_handler(e):
    print(f"⚠️ Socket.IO Error: {str(e)}")
    import traceback
    traceback.print_exc()

# -------------------------------------------------
# Run server
# -------------------------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    
    print(f"\n🌐 Server starting on port {port}")
    print("📡 Socket.IO enabled for real-time communication")
    print("🤖 MediaPipe hand tracking active")
    print("🎓 Ready for real-time letter recognition")
    print("\nPress CTRL+C to stop\n")

    try:
        socketio.run(
            app,
            host="0.0.0.0",
            port=port,  # ← Use dynamic port
            debug=False,
            use_reloader=False
        )
    finally:
        hand_processor.cleanup()
