import cv2
import mediapipe as mp
import numpy as np
import base64

class HandProcessor:
    def __init__(self):
        """Initialize MediaPipe Hands"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )
        print("✅ HandProcessor initialized")
    
    def process_frame(self, frame_data):
        """
        Process a frame and extract hand landmarks
        
        Args:
            frame_data: Base64 encoded image string
            
        Returns:
            dict with landmarks and metadata
        """
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB (MediaPipe uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Extract landmarks if hands detected
            if results.multi_hand_landmarks:
                hands_data = []
                
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get handedness (Left/Right)
                    raw_handedness = results.multi_handedness[hand_idx].classification[0].label
                    handedness = 'Left' if raw_handedness == 'Right' else 'Right'

                    
                    # Extract all 21 landmarks
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    
                    hands_data.append({
                        'handedness': handedness,
                        'landmarks': landmarks
                    })
                
                return {
                    'success': True,
                    'hands_detected': len(hands_data),
                    'hands': hands_data
                }
            else:
                return {
                    'success': True,
                    'hands_detected': 0,
                    'hands': []
                }
                
        except Exception as e:
            print(f"❌ Error processing frame: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'hands_detected': 0,
                'hands': []
            }
    
    def cleanup(self):
        """Clean up resources"""
        self.hands.close()
        print("🧹 HandProcessor cleaned up")