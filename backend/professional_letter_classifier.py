import json
import os
import pickle
import traceback
from pathlib import Path

import h5py
import numpy as np

from feature_extractor import FeatureExtractor


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "models" / "professional_model.h5"
DEFAULT_LABELS_PATH = APP_DIR / "models" / "class_labels.json"
DEFAULT_MAPPING_PATH = APP_DIR / "models" / "professional_label_mapping.pkl"
EXPECTED_FEATURE_COUNT = 72


class ProfessionalLetterClassifier:
    """
    ASL letter classifier for the professional 72-feature model.

    The production default is a lightweight NumPy+h5py inference path that reads
    the existing Keras H5 weights directly. TensorFlow is still available as an
    explicit local/reference backend via SIGNAPP_INFERENCE_BACKEND=tensorflow.
    """

    def __init__(self, backend=None):
        self.backend = (backend or os.environ.get("SIGNAPP_INFERENCE_BACKEND", "numpy")).lower()
        self.model = None
        self.weights = {}
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.is_trained = False
        self.feature_extractor = FeatureExtractor()
        self.model_path = None
        self.expected_input_shape = [None, EXPECTED_FEATURE_COUNT]
        self.output_shape = None
        self.initialization_error = None
        print(f"ProfessionalLetterClassifier initialized ({self.backend})")

    def load_model(self, model_path=None):
        """Load and validate the configured model artifact."""
        self.model_path = Path(model_path or os.environ.get("SIGNAPP_MODEL_PATH", DEFAULT_MODEL_PATH))
        if not self.model_path.is_absolute():
            self.model_path = APP_DIR / self.model_path

        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self._load_labels()

            if self.backend in ("numpy", "h5_numpy", "h5py"):
                self._load_numpy_model()
            elif self.backend in ("tensorflow", "keras", "tf"):
                self._load_tensorflow_model()
            else:
                raise ValueError(
                    f"Unsupported SIGNAPP_INFERENCE_BACKEND={self.backend!r}. "
                    "Use 'numpy' or 'tensorflow'."
                )

            self._validate_contract()
            self.is_trained = True
            self.initialization_error = None
            print(f"Professional model ready ({self.backend}); labels={len(self.labels)}")
            return True
        except Exception as exc:
            self.model = None
            self.weights = {}
            self.is_trained = False
            self.initialization_error = self._safe_error(exc)
            print(f"Model initialization failed: {self.initialization_error}")
            traceback.print_exc()
            return False

    def _load_labels(self):
        mapping_path = Path(os.environ.get("SIGNAPP_MAPPING_PATH", DEFAULT_MAPPING_PATH))
        if not mapping_path.is_absolute():
            mapping_path = APP_DIR / mapping_path

        labels_path = Path(os.environ.get("SIGNAPP_LABELS_PATH", DEFAULT_LABELS_PATH))
        if not labels_path.is_absolute():
            labels_path = APP_DIR / labels_path

        if mapping_path.exists():
            with open(mapping_path, "rb") as f:
                mappings = pickle.load(f)
            self.label_to_idx = mappings["label_to_idx"]
            self.idx_to_label = mappings["idx_to_label"]
            self.labels = [self.idx_to_label[idx] for idx in sorted(self.idx_to_label)]
            self._validate_json_labels_if_present(labels_path)
            return

        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if not isinstance(labels, list) or not labels:
                raise ValueError(f"Label file is invalid: {labels_path.name}")
            self.labels = [str(label) for label in labels]
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
            return

        raise FileNotFoundError(f"No label mapping found: {labels_path.name} or {mapping_path.name}")

    def _validate_json_labels_if_present(self, labels_path):
        if not labels_path.exists():
            return
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if [str(label) for label in labels] != self.labels:
            raise ValueError(
                f"{labels_path.name} does not match professional label mapping; "
                "remove the stale file or regenerate both artifacts together."
            )

    def _load_numpy_model(self):
        with h5py.File(self.model_path, "r") as f:
            config = self._validate_numpy_architecture(f)
            root = f["model_weights"]
            self.weights = {
                "dense": self._dense(root, "dense"),
                "bn": self._batch_norm(
                    root, "batch_normalization", config["batch_normalization"]["epsilon"]
                ),
                "dense_1": self._dense(root, "dense_1"),
                "bn_1": self._batch_norm(
                    root, "batch_normalization_1", config["batch_normalization_1"]["epsilon"]
                ),
                "dense_2": self._dense(root, "dense_2"),
                "dense_3": self._dense(root, "dense_3"),
            }

        self.output_shape = [None, int(self.weights["dense_3"]["bias"].shape[0])]

    def _load_tensorflow_model(self):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        input_shape = list(self.model.input_shape)
        output_shape = list(self.model.output_shape)
        self.expected_input_shape = [None, int(input_shape[-1])]
        self.output_shape = [None, int(output_shape[-1])]

    @staticmethod
    def _dense(root, name):
        group = root[name][name]
        return {
            "kernel": np.array(group["kernel:0"], dtype=np.float32),
            "bias": np.array(group["bias:0"], dtype=np.float32),
        }

    @staticmethod
    def _batch_norm(root, name, epsilon):
        group = root[name][name]
        return {
            "gamma": np.array(group["gamma:0"], dtype=np.float32),
            "beta": np.array(group["beta:0"], dtype=np.float32),
            "mean": np.array(group["moving_mean:0"], dtype=np.float32),
            "variance": np.array(group["moving_variance:0"], dtype=np.float32),
            "epsilon": float(epsilon),
        }

    @staticmethod
    def _validate_numpy_architecture(h5_file):
        raw_config = h5_file.attrs.get("model_config")
        if raw_config is None:
            raise ValueError("H5 model does not contain a Keras model_config")
        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8")

        layers = json.loads(raw_config)["config"]["layers"]
        expected = [
            ("InputLayer", "input_1", None),
            ("Dense", "dense", {"units": 900, "activation": "relu", "use_bias": True}),
            ("BatchNormalization", "batch_normalization", {"axis": [1], "center": True, "scale": True}),
            ("Dropout", "dropout", {"rate": 0.15}),
            ("Dense", "dense_1", {"units": 400, "activation": "relu", "use_bias": True}),
            ("BatchNormalization", "batch_normalization_1", {"axis": [1], "center": True, "scale": True}),
            ("Dropout", "dropout_1", {"rate": 0.25}),
            ("Dense", "dense_2", {"units": 200, "activation": "tanh", "use_bias": True}),
            ("Dropout", "dropout_2", {"rate": 0.4}),
            ("Dense", "dense_3", {"units": 27, "activation": "softmax", "use_bias": True}),
        ]
        if len(layers) != len(expected):
            raise ValueError("H5 model architecture does not match the supported inference graph")

        configs = {}
        for layer, (expected_class, expected_name, required) in zip(layers, expected):
            layer_config = layer.get("config", {})
            if layer.get("class_name") != expected_class or layer_config.get("name") != expected_name:
                raise ValueError("H5 model layer order does not match the supported inference graph")
            for key, expected_value in (required or {}).items():
                actual = layer_config.get(key)
                matches = np.isclose(actual, expected_value) if isinstance(expected_value, float) else actual == expected_value
                if not matches:
                    raise ValueError(f"Unsupported {expected_name} setting: {key}")
            configs[expected_name] = layer_config

        if configs["input_1"].get("batch_input_shape", [None, None])[-1] != EXPECTED_FEATURE_COUNT:
            raise ValueError("H5 model input does not match the 72-feature extractor")
        return configs

    def _validate_contract(self):
        if self.expected_input_shape[-1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Model expects {self.expected_input_shape[-1]} features; "
                f"feature extractor provides {EXPECTED_FEATURE_COUNT}."
            )
        if self.output_shape is None:
            raise ValueError("Model output shape could not be determined.")
        if self.output_shape[-1] != len(self.labels):
            raise ValueError(
                f"Model outputs {self.output_shape[-1]} classes but label map has {len(self.labels)}."
            )
        expected_indices = list(range(len(self.labels)))
        if sorted(self.idx_to_label.keys()) != expected_indices:
            raise ValueError("Label mapping indices are not contiguous from zero.")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Label mapping contains duplicate labels.")
        if any(self.label_to_idx.get(label) != idx for idx, label in self.idx_to_label.items()):
            raise ValueError("Label mappings are not exact inverses.")

        if self.backend in ("numpy", "h5_numpy", "h5py"):
            expected_shapes = {
                ("dense", "kernel"): (72, 900),
                ("dense", "bias"): (900,),
                ("bn", "gamma"): (900,),
                ("bn", "beta"): (900,),
                ("bn", "mean"): (900,),
                ("bn", "variance"): (900,),
                ("dense_1", "kernel"): (900, 400),
                ("dense_1", "bias"): (400,),
                ("bn_1", "gamma"): (400,),
                ("bn_1", "beta"): (400,),
                ("bn_1", "mean"): (400,),
                ("bn_1", "variance"): (400,),
                ("dense_2", "kernel"): (400, 200),
                ("dense_2", "bias"): (200,),
                ("dense_3", "kernel"): (200, 27),
                ("dense_3", "bias"): (27,),
            }
            for (layer, tensor), expected_shape in expected_shapes.items():
                if self.weights[layer][tensor].shape != expected_shape:
                    raise ValueError(f"Unexpected tensor shape for {layer}.{tensor}")

    def predict(self, landmarks):
        if not self.is_trained:
            return self._prediction_error("Model is not ready")

        try:
            features = self.feature_extractor.extract_features(landmarks)
            if features is None:
                return self._prediction_error("Feature extraction failed")
            if features.shape != (EXPECTED_FEATURE_COUNT,):
                return self._prediction_error(f"Expected 72 features, got {features.shape}")

            probabilities = self._predict_features(features.reshape(1, -1).astype(np.float32))[0]
            predicted_idx = int(np.argmax(probabilities))
            predicted_letter = self.idx_to_label[predicted_idx]
            confidence = float(probabilities[predicted_idx])

            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            return {
                "success": True,
                "letter": predicted_letter,
                "confidence": confidence,
                "top_3": [
                    {"letter": self.idx_to_label[int(idx)], "confidence": float(probabilities[idx])}
                    for idx in top_3_indices
                ],
                "all_probabilities": {
                    self.idx_to_label[i]: float(probabilities[i])
                    for i in range(len(probabilities))
                },
            }
        except Exception:
            traceback.print_exc()
            return self._prediction_error("Prediction failed; check server logs")

    def _predict_features(self, features):
        if self.backend in ("tensorflow", "keras", "tf"):
            return self.model.predict(features, verbose=0)

        x = self._batch_norm_apply(self._relu(self._linear(features, "dense")), "bn")
        x = self._batch_norm_apply(self._relu(self._linear(x, "dense_1")), "bn_1")
        x = np.tanh(self._linear(x, "dense_2"))
        return self._softmax(self._linear(x, "dense_3"))

    def _linear(self, x, key):
        layer = self.weights[key]
        return np.matmul(x, layer["kernel"]) + layer["bias"]

    def _batch_norm_apply(self, x, key):
        bn = self.weights[key]
        return bn["gamma"] * ((x - bn["mean"]) / np.sqrt(bn["variance"] + bn["epsilon"])) + bn["beta"]

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def _softmax(x):
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def _prediction_error(message):
        return {"success": False, "error": message, "letter": None, "confidence": 0.0}

    @staticmethod
    def _safe_error(exc):
        return f"{exc.__class__.__name__}: {exc}"

    def status(self):
        return {
            "backend": self.backend,
            "loaded": self.is_trained,
            "ready": self.is_trained,
            "label_count": len(self.labels),
            "labels": sorted(self.labels) if self.is_trained else [],
            "expected_input_shape": self.expected_input_shape,
            "output_shape": self.output_shape,
            "initialization_error": (
                "Model initialization failed; check server logs."
                if self.initialization_error else None
            ),
        }

    def save_model(self, model_path=None):
        print(f"Professional model artifact remains at {model_path or self.model_path}")
        return True

    def train(self, landmarks_list, labels_list):
        print("Professional model training should be done offline with train_professional_model.py")
        return False
