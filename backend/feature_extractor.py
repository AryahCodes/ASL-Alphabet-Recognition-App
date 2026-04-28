import numpy as np

class FeatureExtractor:
    """
    Extract meaningful features from hand landmarks for gesture recognition
    WITH Z-SCORE NORMALIZATION (Tier 1.5 - Professional approach)
    + FINGER ANGLES (Helps distinguish D/M/N/G)
    """
    
    def __init__(self):
        print("✅ FeatureExtractor initialized with Z-score normalization + finger angles")
    
    def calculate_angle(self, p1, p2, p3):
        """
        Calculate angle between three points (in radians)
        p2 is the vertex of the angle
        
        Args:
            p1, p2, p3: numpy arrays of shape (2,) representing 2D points
            
        Returns:
            Angle in radians (0 to π)
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate cosine of angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        
        # Clip to valid range and calculate angle
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return angle
    
    def extract_features(self, landmarks):
        """
        Extract features from 21 hand landmarks with professional z-score normalization
        
        This normalization removes distance-from-camera variance, making the model
        more robust to different hand positions and camera distances.
        
        Args:
            landmarks: List of 21 landmarks with x, y, z coordinates
            
        Returns:
            numpy array of z-score normalized features (72 features total)
        """
        if not landmarks or len(landmarks) != 21:
            return None
        
        # Convert landmarks to numpy array
        points = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        # ========================================
        # TIER 1.5: Z-SCORE NORMALIZATION
        # ========================================
        # Extract x, y coordinates (ignoring z as it's less reliable from MediaPipe)
        x_coords = points[:, 0]  # All x coordinates
        y_coords = points[:, 1]  # All y coordinates
        
        # Calculate mean and std for x and y separately
        x_mean, x_std = np.mean(x_coords), np.std(x_coords)
        y_mean, y_std = np.mean(y_coords), np.std(y_coords)
        
        # Prevent division by zero (if hand is completely still)
        if x_std < 1e-6:
            x_std = 1.0
        if y_std < 1e-6:
            y_std = 1.0
        
        # Apply z-score normalization: (value - mean) / std
        x_normalized = (x_coords - x_mean) / x_std
        y_normalized = (y_coords - y_mean) / y_std
        
        # Create normalized points
        normalized_points = np.column_stack([x_normalized, y_normalized])
        
        # ========================================
        # FEATURE EXTRACTION (on normalized coordinates)
        # ========================================
        features = []
        
        # 1. Flattened normalized positions (42 features: 21 points × 2 coords)
        features.extend(normalized_points.flatten())
        
        # 2. Palm center (normalized)
        palm_indices = [0, 5, 9, 13, 17]
        palm_center = np.mean(normalized_points[palm_indices], axis=0)
        
        # 3. Finger tip distances from palm center (5 features)
        finger_tips = [4, 8, 12, 16, 20]
        for tip_idx in finger_tips:
            distance = np.linalg.norm(normalized_points[tip_idx] - palm_center)
            features.append(distance)
        
        # 4. Distances between consecutive fingertips (4 features)
        for i in range(len(finger_tips) - 1):
            dist = np.linalg.norm(normalized_points[finger_tips[i]] - normalized_points[finger_tips[i+1]])
            features.append(dist)
        
        # 5. Finger extension ratios (5 features)
        finger_chains = [
            [1, 2, 3, 4],      # Thumb
            [5, 6, 7, 8],      # Index
            [9, 10, 11, 12],   # Middle
            [13, 14, 15, 16],  # Ring
            [17, 18, 19, 20]   # Pinky
        ]
        
        for chain in finger_chains:
            finger_length = 0
            for i in range(len(chain) - 1):
                finger_length += np.linalg.norm(normalized_points[chain[i+1]] - normalized_points[chain[i]])
            
            base_to_tip = np.linalg.norm(normalized_points[chain[-1]] - normalized_points[chain[0]])
            extension_ratio = base_to_tip / (finger_length + 1e-6)
            features.append(extension_ratio)
        
        # ========================================
        # NEW: FINGER ANGLES (10 features)
        # This helps distinguish D/M/N/G better!
        # ========================================

        # 6. Finger bend angles per finger (10 features: 5 fingers × 2 angles each)
        for chain in finger_chains:
            # For each finger, calculate angles at the 3 joints
            for i in range(len(chain) - 2):
                angle = self.calculate_angle(
                    normalized_points[chain[i]],
                    normalized_points[chain[i + 1]],
                    normalized_points[chain[i + 2]]
                )
                features.append(angle)
        
        # 7. Hand direction (2 features - normalized)
        hand_direction = normalized_points[9] - normalized_points[0]
        hand_direction_norm = hand_direction / (np.linalg.norm(hand_direction) + 1e-6)
        features.extend(hand_direction_norm)
        
        # 8. Palm size (1 feature)
        palm_size = np.linalg.norm(normalized_points[9] - normalized_points[0])
        features.append(palm_size)
        
        # ========================================
        # NEW: INTER-FINGER ANGLES (3 features)
        # Angles between adjacent fingers - helps with D/M/N
        # ========================================

        # 9. Angles between adjacent finger bases (3 features)
        finger_bases = [5, 9, 13, 17]  # Base of index, middle, ring, pinky
        for i in range(len(finger_bases) - 1):
            angle = self.calculate_angle(
                normalized_points[finger_bases[i]],
                normalized_points[0],  # Wrist as vertex
                normalized_points[finger_bases[i + 1]]
            )
            features.append(angle)
        
        # Total features: 42 + 5 + 4 + 5 + 10 + 2 + 1 + 3 = 72 features
        return np.array(features, dtype=np.float32)
    
    def extract_batch(self, landmarks_list):
        """Extract features from multiple landmark sets"""
        features_list = []
        for landmarks in landmarks_list:
            features = self.extract_features(landmarks)
            if features is not None:
                features_list.append(features)
        
        return np.array(features_list) if features_list else None
    
    def extract_simple_normalized(self, landmarks):
        """
        SIMPLIFIED VERSION: Just normalized x,y coordinates (42 features)
        Use this if the full feature extraction causes issues
        """
        if not landmarks or len(landmarks) != 21:
            return None
        
        points = np.array([[lm['x'], lm['y']] for lm in landmarks])
        
        # Z-score normalize
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        x_mean, x_std = np.mean(x_coords), np.std(x_coords)
        y_mean, y_std = np.mean(y_coords), np.std(y_coords)
        
        if x_std < 1e-6:
            x_std = 1.0
        if y_std < 1e-6:
            y_std = 1.0
        
        x_normalized = (x_coords - x_mean) / x_std
        y_normalized = (y_coords - y_mean) / y_std
        
        # Interleave x and y: [x0, y0, x1, y1, x2, y2, ...]
        features = np.empty(42, dtype=np.float32)
        features[::2] = x_normalized
        features[1::2] = y_normalized
        
        return features