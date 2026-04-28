"""
conftest.py — shared fixtures for the backend test suite.

The server imports eventlet and calls monkey_patch() at module level, which
breaks pytest's threading model.  We neutralise this by injecting a no-op
MagicMock for the entire eventlet package *before* any project module is
imported.  We also stub the heavy ML/CV dependencies (mediapipe, tensorflow,
cv2) so the suite runs without a full GPU/CPU environment.
"""

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Stub every heavy / environment-sensitive dependency before any import
#    from the project.  Order matters: the stubs must land in sys.modules
#    before Python resolves any "import X" inside server.py.
# ---------------------------------------------------------------------------

_STUBS = [
    # eventlet and its sub-modules referenced by flask-socketio
    "eventlet",
    "eventlet.hubs",
    # mediapipe and typical sub-namespaces
    "mediapipe",
    "mediapipe.solutions",
    "mediapipe.solutions.hands",
    "mediapipe.solutions.drawing_utils",
    # OpenCV
    "cv2",
    # TensorFlow / Keras (used by professional_letter_classifier)
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.keras.models",
    "tensorflow.keras.layers",
    "tensorflow.keras.callbacks",
    "tensorflow.keras.utils",
    "tflite_runtime",
    "tflite_runtime.interpreter",
    # scikit-learn (used by letter_classifier)
    "sklearn",
    "sklearn.ensemble",
    "sklearn.model_selection",
    "sklearn.preprocessing",
    "sklearn.metrics",
]

for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Make eventlet.monkey_patch a plain no-op so the call at the top of
# server.py does nothing harmful.
sys.modules["eventlet"].monkey_patch = MagicMock(return_value=None)

# SocketIO validates async_mode against the real engineio at init time.
# Since eventlet is stubbed, that validation would fail.  Replace the
# SocketIO class itself with a MagicMock so server.py's module-level
# `socketio = SocketIO(app, async_mode="eventlet", ...)` becomes a no-op.
import flask_socketio as _fso
from unittest.mock import MagicMock
_fso.SocketIO = MagicMock(return_value=MagicMock())

# flask-cors is a real package in this venv; no stub needed.

# ---------------------------------------------------------------------------
# 2. Now it is safe to import from the project.
# ---------------------------------------------------------------------------

# We need server.py to resolve its own imports from the backend directory.
import os
import sys as _sys
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in _sys.path:
    _sys.path.insert(0, _backend_dir)

import pytest

# Import lazily so collection itself never crashes if something is wrong.
from server import app, FrameBuffer, PredictionSmoother  # noqa: E402


@pytest.fixture
def client():
    """Flask test client with TESTING mode enabled."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
