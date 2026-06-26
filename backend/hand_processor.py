import cv2
import mediapipe as mp
import numpy as np
import base64
import binascii


MAX_FRAME_BYTES = 1_500_000

class HandProcessor:
    def __init__(self):
        """Initialize MediaPipe Hands"""
        self.mp_hands = None
        self.hands = None
        self.ready = False
        self.initialization_error = None
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.ready = True
            print("HandProcessor initialized")
        except Exception as exc:
            self.initialization_error = f"{exc.__class__.__name__}: {exc}"
            print(f"HandProcessor initialization failed: {self.initialization_error}")
    
    def process_frame(self, frame_data):
        """
        Process a frame and extract hand landmarks
        
        Args:
            frame_data: Base64 encoded image string
            
        Returns:
            dict with landmarks and metadata
        """
        try:
            if not self.ready or self.hands is None:
                raise RuntimeError(f"Hand processor is not ready: {self.initialization_error}")

            if not frame_data or not isinstance(frame_data, str):
                raise ValueError("Missing frame data")
            if "," not in frame_data:
                raise ValueError("Frame must be a data URL")

            encoded = frame_data.split(",", 1)[1]
            estimated_size = (len(encoded) * 3) // 4
            if estimated_size > MAX_FRAME_BYTES:
                raise ValueError("Frame payload is too large")

            try:
                img_bytes = base64.b64decode(encoded, validate=True)
            except binascii.Error as exc:
                raise ValueError("Frame is not valid base64") from exc

            if len(img_bytes) > MAX_FRAME_BYTES:
                raise ValueError("Frame payload is too large")

            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Frame could not be decoded")

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
            print(f"Error processing frame: {str(e)}")
            return {
                'success': False,
                'error': str(e) if isinstance(e, ValueError) else 'Frame processing failed',
                'hands_detected': 0,
                'hands': []
            }
    
    def cleanup(self):
        """Clean up resources"""
        if self.hands is not None:
            self.hands.close()
        print("HandProcessor cleaned up")
