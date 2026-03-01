import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import socket from './socket';

function HandTracking() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [isTracking, setIsTracking] = useState(false);
  const [handsDetected, setHandsDetected] = useState(0);
  const [fps, setFps] = useState(0);
  const [predictedLetter, setPredictedLetter] = useState(null);
  const [predictionConfidence, setPredictionConfidence] = useState(0);
  const intervalRef = useRef(null);
  const fpsCounterRef = useRef({ frames: 0, lastTime: Date.now() });

  // Draw hand landmarks on canvas
  const drawHands = (hands) => {
    const canvas = canvasRef.current;
    const video = webcamRef.current?.video;
    
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw each hand
    hands.forEach((hand) => {
      const landmarks = hand.landmarks;
      
      // Draw connections between landmarks
      const connections = [
        // Thumb
        [0, 1], [1, 2], [2, 3], [3, 4],
        // Index finger
        [0, 5], [5, 6], [6, 7], [7, 8],
        // Middle finger
        [0, 9], [9, 10], [10, 11], [11, 12],
        // Ring finger
        [0, 13], [13, 14], [14, 15], [15, 16],
        // Pinky
        [0, 17], [17, 18], [18, 19], [19, 20],
        // Palm
        [5, 9], [9, 13], [13, 17]
      ];

      // Draw connections
      ctx.strokeStyle = hand.handedness === 'Right' ? '#00FF00' : '#FF00FF';
      ctx.lineWidth = 2;
      
      connections.forEach(([start, end]) => {
        const startPoint = landmarks[start];
        const endPoint = landmarks[end];
        
        ctx.beginPath();
        ctx.moveTo(startPoint.x * canvas.width, startPoint.y * canvas.height);
        ctx.lineTo(endPoint.x * canvas.width, endPoint.y * canvas.height);
        ctx.stroke();
      });

      // Draw landmark points
      landmarks.forEach((landmark, index) => {
        const x = landmark.x * canvas.width;
        const y = landmark.y * canvas.height;
        
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = index === 0 ? '#FF0000' : '#00FFFF'; // Red for wrist, cyan for others
        ctx.fill();
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Draw hand label
      const wrist = landmarks[0];
      ctx.fillStyle = hand.handedness === 'Right' ? '#00FF00' : '#FF00FF';
      ctx.font = 'bold 20px Arial';
      ctx.fillText(
        `${hand.handedness} Hand`,
        wrist.x * canvas.width - 40,
        wrist.y * canvas.height - 10
      );
    });

    // Update FPS counter
    fpsCounterRef.current.frames++;
    const now = Date.now();
    const elapsed = now - fpsCounterRef.current.lastTime;
    
    if (elapsed >= 1000) {
      setFps(fpsCounterRef.current.frames);
      fpsCounterRef.current.frames = 0;
      fpsCounterRef.current.lastTime = now;
    }
  };

  // Listen for hand landmarks from backend
  useEffect(() => {
    socket.on('hand_landmarks', (data) => {
      if (data.success && data.hands_detected > 0) {
        setHandsDetected(data.hands_detected);
        drawHands(data.hands);
        
        // Update letter prediction if available
        if (data.letter_prediction && data.letter_prediction.success) {
          setPredictedLetter(data.letter_prediction.letter);
          setPredictionConfidence(data.letter_prediction.confidence);
        }
      } else {
        setHandsDetected(0);
        setPredictedLetter(null);
        setPredictionConfidence(0);
        // Clear canvas when no hands detected
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
    });

    return () => {
      socket.off('hand_landmarks');
    };
  }, []);

  // Start tracking
  const startTracking = () => {
    setIsTracking(true);
    
    intervalRef.current = setInterval(() => {
      if (webcamRef.current) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          socket.emit('process_frame', { frame: imageSrc });
        }
      }
    }, 100); // Send frame every 100ms (10 FPS)
  };

  // Stop tracking
  const stopTracking = () => {
    setIsTracking(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setHandsDetected(0);
    setFps(0);
    setPredictedLetter(null);
    setPredictionConfidence(0);
    
    // Clear canvas
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return (
    <div style={{ textAlign: 'center', padding: '20px' }}>
      <h1>🤟 Real-Time Hand Tracking & Letter Recognition</h1>
      <p style={{ color: '#666', marginBottom: '20px' }}>
        ASL Alphabet Recognition - 24 Letters (A-Y) with 96.86% Accuracy
      </p>

      <div style={{ position: 'relative', display: 'inline-block' }}>
        <Webcam
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          style={{
            width: '640px',
            height: '480px',
            border: '3px solid #007bff',
            borderRadius: '10px',
            position: 'relative',
            zIndex: 1
          }}
        />
        
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '640px',
            height: '480px',
            zIndex: 2,
            pointerEvents: 'none'
          }}
        />

        {/* Status overlay */}
        <div style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '10px',
          borderRadius: '8px',
          zIndex: 3,
          fontSize: '14px',
          fontFamily: 'monospace'
        }}>
          <div>Status: {isTracking ? '🟢 Tracking' : '🔴 Stopped'}</div>
          <div>Hands: {handsDetected}</div>
          <div>FPS: {fps}</div>
        </div>

        {/* Letter prediction overlay - BIG and prominent */}
        {predictedLetter && (
          <div style={{
            position: 'absolute',
            bottom: '20px',
            left: '50%',
            transform: 'translateX(-50%)',
            backgroundColor: 'rgba(0, 123, 255, 0.9)',
            color: 'white',
            padding: '20px 40px',
            borderRadius: '15px',
            zIndex: 3,
            fontSize: '48px',
            fontWeight: 'bold',
            fontFamily: 'Arial',
            border: '3px solid white',
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
          }}>
            <div>Letter: {predictedLetter}</div>
            <div style={{ fontSize: '24px', marginTop: '10px' }}>
              {(predictionConfidence * 100).toFixed(0)}% confident
            </div>
          </div>
        )}
      </div>

      <br />

      <div style={{ marginTop: '20px' }}>
        {!isTracking ? (
          <button
            onClick={startTracking}
            style={{
              padding: '15px 40px',
              fontSize: '18px',
              fontWeight: 'bold',
              cursor: 'pointer',
              backgroundColor: '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              boxShadow: '0 4px 6px rgba(0,0,0,0.2)',
              transition: 'all 0.3s'
            }}
            onMouseOver={(e) => e.target.style.backgroundColor = '#218838'}
            onMouseOut={(e) => e.target.style.backgroundColor = '#28a745'}
          >
            ▶️ Start Hand Tracking
          </button>
        ) : (
          <button
            onClick={stopTracking}
            style={{
              padding: '15px 40px',
              fontSize: '18px',
              fontWeight: 'bold',
              cursor: 'pointer',
              backgroundColor: '#dc3545',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              boxShadow: '0 4px 6px rgba(0,0,0,0.2)',
              transition: 'all 0.3s'
            }}
            onMouseOver={(e) => e.target.style.backgroundColor = '#c82333'}
            onMouseOut={(e) => e.target.style.backgroundColor = '#dc3545'}
          >
            ⏹️ Stop Tracking
          </button>
        )}
      </div>

      <div style={{
        marginTop: '30px',
        padding: '20px',
        backgroundColor: '#f8f9fa',
        borderRadius: '10px',
        maxWidth: '800px',
        margin: '30px auto'
      }}>
        <h3>📋 Instructions:</h3>
        <ol style={{ textAlign: 'left', lineHeight: '1.8' }}>
          <li>Click "Start Hand Tracking"</li>
          <li>Show your hand to the camera</li>
          <li>Make any ASL letter sign (A-Y, excluding J and Z)</li>
          <li>Watch the blue box appear showing the recognized letter!</li>
          <li>Try different letters and see the confidence percentage</li>
        </ol>
        
        <div style={{
          marginTop: '20px',
          padding: '15px',
          backgroundColor: '#d1ecf1',
          borderRadius: '8px',
          border: '1px solid #bee5eb'
        }}>
          <strong>🎉 Model Info:</strong> Trained on 9,572 Kaggle samples with 96.86% accuracy!
          <br />
          <strong>🔤 Recognizes:</strong> A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y
          <br />
          <strong>📊 Best Performance:</strong> D, F, K, Y (91-94% accuracy)
        </div>

        <div style={{
          marginTop: '15px',
          padding: '15px',
          backgroundColor: '#fff3cd',
          borderRadius: '8px',
          border: '1px solid #ffc107'
        }}>
          <strong>💡 Tip:</strong> Hold your hand steady for 2-3 seconds for best results. 
          Keep your hand centered and well-lit!
        </div>
      </div>
    </div>
  );
}

export default HandTracking;