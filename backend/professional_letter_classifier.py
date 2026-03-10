import numpy as np
import tensorflow as tf
import pickle
import json
from pathlib import Path

class ProfessionalLetterClassifier:
    """
    Professional Deep Learning Letter Classifier
    Uses TensorFlow/Keras model trained on z-score normalized features
    """
    
    def __init__(self):
        self.model = None
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.is_trained = False
        self.feature_extractor = None
        print("✅ ProfessionalLetterClassifier initialized")
    
    def load_model(self, model_path='models/professional_model.h5'):
        """Load the trained professional model"""
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                print(f"⚠️  Model not found at {model_path}")
                return False
            
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print(f"✅ Loaded professional model from {model_path}")
            
            mapping_path = Path('models/professional_label_mapping.pkl')
            if mapping_path.exists():
                with open(mapping_path, 'rb') as f:
                    mappings = pickle.load(f)
                    self.label_to_idx = mappings['label_to_idx']
                    self.idx_to_label = mappings['idx_to_label']
                    self.labels = sorted(self.label_to_idx.keys())
                print(f"✅ Loaded label mappings: {self.labels}")
            else:
                labels_path = Path('models/professional_labels.json')
                if labels_path.exists():
                    with open(labels_path, 'r') as f:
                        self.labels = json.load(f)
                        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
                        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
                    print(f"✅ Loaded labels from JSON: {self.labels}")
                else:
                    print("❌ No label mapping found!")
                    return False
            
            from feature_extractor import FeatureExtractor
            self.feature_extractor = FeatureExtractor()
            
            self.is_trained = True
            print(f"✅ Professional model ready! Can recognize: {self.labels}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading professional model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, landmarks):
        """
        Predict letter from hand landmarks
        
        Args:
            landmarks: List of 21 landmarks with x, y, z coordinates
            
        Returns:
            dict with 'success', 'letter', 'confidence', 'probabilities'
        """
        if not self.is_trained or self.model is None:
            return {
                'success': False,
                'error': 'Model not trained',
                'letter': None,
                'confidence': 0.0
            }
        
        try:
            features = self.feature_extractor.extract_features(landmarks)
            
            if features is None:
                return {
                    'success': False,
                    'error': 'Feature extraction failed',
                    'letter': None,
                    'confidence': 0.0
                }
            
            features = features.reshape(1, -1)
            probabilities = self.model.predict(features.astype(np.float32), verbose=0)[0]
            
            predicted_idx = np.argmax(probabilities)
            predicted_letter = self.idx_to_label[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3 = [
                {
                    'letter': self.idx_to_label[idx],
                    'confidence': float(probabilities[idx])
                }
                for idx in top_3_indices
            ]
            
            return {
                'success': True,
                'letter': predicted_letter,
                'confidence': confidence,
                'top_3': top_3,
                'all_probabilities': {
                    self.idx_to_label[i]: float(probabilities[i])
                    for i in range(len(probabilities))
                }
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'letter': None,
                'confidence': 0.0
            }
    
    def save_model(self, model_path='models/professional_model.h5'):
        """Model is already saved during training"""
        print(f"💾 Professional model saved at {model_path}")
        return True
    
    def train(self, landmarks_list, labels_list):
        """
        Training is done separately via train_professional_kaggle.py
        This method is here for compatibility with the old interface
        """
        print("⚠️  Professional model training should be done via:")
        print("    python train_professional_kaggle.py")
        return False
