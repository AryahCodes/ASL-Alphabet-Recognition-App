import eventlet
eventlet.monkey_patch()  # ✅ must be first before importing Flask / SocketIO

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from hand_processor import HandProcessor
from data_collector import DataCollector
from collections import deque, Counter
import numpy as np
import gc
import os
import sys
import threading
import time

# -------------------------------------------------
# Metrics
# -------------------------------------------------
_server_start_time = time.time()
_metrics_lock = threading.Lock()
_metrics = {
    "frames_received": 0,
    "frames_processed": 0,
    "frames_no_hand": 0,
    "frames_failed": 0,
    "active_clients": 0,
}
_latency_samples = deque(maxlen=200)
_latency_stats = {"count": 0, "sum": 0.0, "min": None, "max": None}

APP_VERSION = os.environ.get("SIGNAPP_VERSION", "3.1")
BUILD_SHA = os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("SIGNAPP_GIT_SHA")
TRAINING_ENABLED = os.environ.get("SIGNAPP_ENABLE_TRAINING", "false").lower() in {
    "1", "true", "yes",
}
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "SIGNAPP_CORS_ORIGINS",
        "http://localhost:3000,https://signapp-frontend.vercel.app",
    ).split(",")
    if origin.strip()
]

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
    
    def __init__(self, window_size=7, confidence_threshold=0.30):
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
        if len(self.predictions) < 2:  # Need at least 3 predictions
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
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

socketio = SocketIO(
    app,
    cors_allowed_origins=ALLOWED_ORIGINS,
    async_mode="eventlet",
    max_http_buffer_size=2 * 1024 * 1024,
    ping_interval=10,
    ping_timeout=25
)

# -------------------------------------------------
# Initialize modules & Choose Model
# -------------------------------------------------
hand_processor = HandProcessor()
data_collector = DataCollector()

# ✅ MODEL SELECTION: Choose which model to use
USE_PROFESSIONAL_MODEL = True

print("=" * 60)
print("Sign Language App - Backend Server")
print("=" * 60)

if USE_PROFESSIONAL_MODEL:
    try:
        from professional_letter_classifier import ProfessionalLetterClassifier
        letter_classifier = ProfessionalLetterClassifier()
        print(f"Using: Professional model ({letter_classifier.backend})")
    except ImportError:
        print("professional_letter_classifier.py not found; falling back to RandomForest model")
        from letter_classifier import LetterClassifier
        letter_classifier = LetterClassifier()
        print("Using: RandomForest Model (Fallback)")
else:
    from letter_classifier import LetterClassifier
    letter_classifier = LetterClassifier()
    print("Using: RandomForest Model")

# Per-client smoothers and buffers
client_smoothers = {}
client_buffers = {}
client_processing = set()

# Load model
letter_classifier.load_model()

if letter_classifier.is_trained:
    print("Model loaded successfully")
    print(f"Can recognize: {sorted(letter_classifier.labels)}")
else:
    print("No recognition model loaded")
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


def _model_status_payload():
    sample_counts = data_collector.get_sample_counts()
    if hasattr(letter_classifier, "status"):
        model = letter_classifier.status()
    else:
        model = {
            "backend": "randomforest",
            "loaded": bool(letter_classifier.is_trained),
            "ready": bool(letter_classifier.is_trained),
            "label_count": len(letter_classifier.labels),
            "labels": sorted(letter_classifier.labels) if letter_classifier.is_trained else [],
            "expected_input_shape": [None, 72],
            "output_shape": [None, len(letter_classifier.labels)] if letter_classifier.labels else None,
            "initialization_error": None if letter_classifier.is_trained else "Model not loaded",
        }

    return {
        "is_trained": bool(letter_classifier.is_trained),
        "ready": bool(letter_classifier.is_trained),
        "recognition_available": bool(letter_classifier.is_trained),
        "model_type": "professional" if USE_PROFESSIONAL_MODEL else "randomforest",
        "sample_counts": sample_counts,
        **model,
    }


def _service_status_payload():
    model = _model_status_payload()
    hand_tracking_ready = bool(getattr(hand_processor, "ready", False))
    ready = bool(model["ready"] and hand_tracking_ready)
    return {
        "service": "signapp-backend",
        "live": True,
        "ready": ready,
        "recognition_available": ready,
        "training_enabled": TRAINING_ENABLED,
        "version": APP_VERSION,
        "git_sha": BUILD_SHA,
        "python_version": sys.version.split()[0],
        "uptime_seconds": round(time.time() - _server_start_time, 2),
        "model": {
            "backend": model["backend"],
            "loaded": model["loaded"],
            "ready": model["ready"],
            "label_count": model["label_count"],
            "expected_input_shape": model["expected_input_shape"],
            "output_shape": model["output_shape"],
            "initialization_error": model["initialization_error"],
        },
        "hand_tracking": {
            "ready": hand_tracking_ready,
            "initialization_error": (
                "Hand tracking initialization failed; check server logs."
                if getattr(hand_processor, "initialization_error", None) else None
            ),
        },
    }


# -------------------------------------------------
# REST API endpoints
# -------------------------------------------------
@app.route("/test")
def test():
    status = _service_status_payload()
    return {
        "message": "Backend process is live. Use /ready for recognition readiness.",
        "status": "success",
        "live": True,
        "ready": status["ready"],
    }

@app.route("/live")
def live():
    return jsonify({"live": True, "service": "signapp-backend"})

@app.route("/ready")
def ready():
    status = _service_status_payload()
    payload = {
        "service": status["service"],
        "ready": status["ready"],
        "recognition_available": status["recognition_available"],
        "model_ready": status["model"]["ready"],
        "hand_tracking_ready": status["hand_tracking"]["ready"],
    }
    return jsonify(payload), 200 if status["ready"] else 503

@app.route("/version")
def version():
    return jsonify(_service_status_payload())

@app.route("/metrics")
def get_metrics():
    with _metrics_lock:
        m = dict(_metrics)
        stats = dict(_latency_stats)
        samples = list(_latency_samples)
    count = stats["count"]
    if count == 0:
        lat = {"count": 0, "mean": None, "min": None, "max": None, "p95": None, "median": None}
    else:
        lat = {
            "count": count,
            "mean": round(stats["sum"] / count, 3),
            "min": round(stats["min"], 3),
            "max": round(stats["max"], 3),
            "p95": round(float(np.percentile(samples, 95)), 3) if samples else None,
            "median": round(float(np.median(samples)), 3) if samples else None,
        }
    return jsonify({
        **m,
        "inference_latency_ms": lat,
        "uptime_seconds": round(time.time() - _server_start_time, 2),
    })

@app.route("/")
def home():
    return {
        "message": "Sign Language App API",
        "version": APP_VERSION,
        "ready": _service_status_payload()["ready"],
        "model_type": "professional" if USE_PROFESSIONAL_MODEL else "randomforest",
        "enhancements": ["Temporal Smoothing", "Z-Score Normalization", "Frame Buffering"]
    }

@app.route("/model/status")
def model_status():
    """Get model training status"""
    return jsonify(_model_status_payload())

# -------------------------------------------------
# Socket.IO events
# -------------------------------------------------
@socketio.on("connect")
def handle_connect():
    client_id = request.sid
    print(f"Client connected: {client_id}")
    
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

    with _metrics_lock:
        _metrics["active_clients"] += 1

    emit("model_status", _model_status_payload())

@socketio.on("disconnect")
def handle_disconnect():
    client_id = request.sid
    print(f"❌ Client disconnected! ID: {client_id}")
    
    # Clean up smoother and buffer
    if client_id in client_smoothers:
        del client_smoothers[client_id]
    if client_id in client_buffers:
        del client_buffers[client_id]
    client_processing.discard(client_id)
    with _metrics_lock:
        _metrics["active_clients"] = max(0, _metrics["active_clients"] - 1)

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
        if client_id in client_processing:
            emit(
                "hand_landmarks",
                {
                    "success": False,
                    "error": "Previous frame is still processing; dropping this frame",
                    "retryable": True,
                    "hands_detected": 0,
                    "hands": [],
                },
            )
            return

        client_processing.add(client_id)
        frame_data = (data or {}).get("frame")
        with _metrics_lock:
            _metrics["frames_received"] += 1

        if not letter_classifier.is_trained:
            emit(
                "hand_landmarks",
                {
                    "success": False,
                    "error": "Recognition model is not ready",
                    "model_status": _model_status_payload(),
                    "hands_detected": 0,
                    "hands": [],
                },
            )
            return

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
            should_predict = buffer.should_predict()
            if letter_classifier.is_trained and should_predict:
                # Get averaged landmarks from buffer
                avg_landmarks = buffer.get_average_landmarks()
                
                if avg_landmarks:
                    # Get raw prediction (will use z-score normalization internally)
                    _t0 = time.perf_counter()
                    prediction = letter_classifier.predict(avg_landmarks)
                    _lat_ms = (time.perf_counter() - _t0) * 1000
                    with _metrics_lock:
                        _metrics["frames_processed"] += 1
                        _latency_stats["count"] += 1
                        _latency_stats["sum"] += _lat_ms
                        _latency_stats["min"] = _lat_ms if _latency_stats["min"] is None else min(_latency_stats["min"], _lat_ms)
                        _latency_stats["max"] = _lat_ms if _latency_stats["max"] is None else max(_latency_stats["max"], _lat_ms)
                        _latency_samples.append(_lat_ms)

                    if prediction["success"]:

                        result["letter_prediction"] = prediction

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
                        else:
                            # Prediction below threshold or not stable enough
                            result["letter_prediction"] = {
                                "success": False,
                                "message": "Hold steady..."
                            }
                            pass
                    else:
                        result["letter_prediction"] = prediction
        else:
            # No hand detected - clear buffer
            smoother.no_hand_detected()
            if result["hands_detected"] == 0:
                with _metrics_lock:
                    _metrics["frames_no_hand"] += 1

        # Send back to frontend
        emit("hand_landmarks", result)

    except Exception as e:
        with _metrics_lock:
            _metrics["frames_failed"] += 1
        print(f"❌ Error in process_frame: {str(e)}")
        import traceback
        traceback.print_exc()
        emit("hand_landmarks", {
            "success": False,
            "error": "Frame processing failed; check server logs",
            "hands_detected": 0,
            "hands": [],
        })
    finally:
        try:
            client_processing.discard(request.sid)
        except Exception:
            pass

@socketio.on("save_training_sample")
def handle_save_sample(data):
    """Save a training sample"""
    try:
        if not TRAINING_ENABLED:
            emit("sample_saved", {"success": False, "error": "Training data collection is disabled"})
            return

        data = data or {}
        landmarks = data.get("landmarks")
        label = data.get("label", "")

        if not landmarks or not label:
            emit("sample_saved", {"success": False, "error": "Missing landmarks or label"})
            return

        data_collector.save_sample(landmarks, label)
        sample_counts = data_collector.get_sample_counts()

        saved_label = data_collector.validate_label(label)
        print(f"Saved training sample: {saved_label} (total: {sample_counts.get(saved_label, 0)})")

        emit(
            "sample_saved",
            {
                "success": True,
                "label": saved_label,
                "sample_counts": sample_counts,
            },
        )

    except ValueError as e:
        print(f"❌ Error saving sample: {str(e)}")
        emit("sample_saved", {"success": False, "error": str(e)})
    except Exception as e:
        print(f"❌ Error saving sample: {str(e)}")
        emit("sample_saved", {"success": False, "error": "Sample could not be saved"})

@socketio.on("train_model")
def handle_train_model(data):
    """Train the letter classifier (only works with RandomForest model)"""
    try:
        if not TRAINING_ENABLED:
            emit("training_complete", {"success": False, "error": "Training is disabled"})
            return

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
        emit("training_complete", {"success": False, "error": "Training failed; check server logs"})

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
