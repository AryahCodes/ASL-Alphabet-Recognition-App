import json
import math
import os
from datetime import datetime
from numbers import Real
from pathlib import Path


ALLOWED_LABELS = {
    'A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
    'del', 'space'
}

class DataCollector:
    """Collect and save hand landmark data for training"""
    
    def __init__(self, data_dir='training_data'):
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"DataCollector initialized (saving to {self.data_dir}/)")

    def validate_label(self, label):
        label = str(label or '').strip()
        normalized = label if label in {'Blank', 'del', 'space'} else label.upper()
        if normalized not in ALLOWED_LABELS:
            raise ValueError(f"Unsupported training label: {label}")
        return normalized
    
    def save_sample(self, landmarks, label):
        """Save a single training sample"""
        label = self.validate_label(label)
        self.validate_landmarks(landmarks)
        label_dir = (self.data_dir / label).resolve()
        if self.data_dir not in label_dir.parents:
            raise ValueError("Training label resolved outside the data directory")
        label_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{label}_{timestamp}.json"
        filepath = label_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump({
                'label': label,
                'landmarks': landmarks,
                'timestamp': timestamp
            }, f, indent=2)
        
        return str(filepath)

    @staticmethod
    def validate_landmarks(landmarks):
        if not isinstance(landmarks, list) or len(landmarks) != 21:
            raise ValueError("Training sample must contain exactly 21 landmarks")
        for landmark in landmarks:
            if not isinstance(landmark, dict):
                raise ValueError("Each landmark must contain numeric x, y, and z values")
            for coordinate in ("x", "y", "z"):
                value = landmark.get(coordinate)
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError("Each landmark must contain numeric x, y, and z values")
    
    def load_all_samples(self):
        """Load all training samples"""
        landmarks_list = []
        labels_list = []
        
        if not self.data_dir.exists():
            return landmarks_list, labels_list
        
        for label in os.listdir(self.data_dir):
            label_dir = self.data_dir / label
            
            if not label_dir.is_dir():
                continue
            
            for filename in os.listdir(label_dir):
                if not filename.endswith('.json'):
                    continue
                
                filepath = label_dir / filename
                
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        landmarks_list.append(data['landmarks'])
                        labels_list.append(data['label'])
                except Exception as e:
                    print(f"⚠️  Error loading {filepath}: {e}")
        
        print(f"Loaded {len(landmarks_list)} samples from {self.data_dir}/")
        return landmarks_list, labels_list
    
    def get_sample_counts(self):
        """Get count of samples per label"""
        counts = {}
        
        if not self.data_dir.exists():
            return counts
        
        for label in os.listdir(self.data_dir):
            label_dir = self.data_dir / label
            
            if not label_dir.is_dir():
                continue
            
            json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
            counts[label] = len(json_files)
        
        return counts
