import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import socket from './socket';

function TrainingMode() {
  const webcamRef = useRef(null);
  const [currentLetter, setCurrentLetter] = useState('A');
  const [sampleCounts, setSampleCounts] = useState({});
  const [isCapturing, setIsCapturing] = useState(false);
  const [message, setMessage] = useState('');
  const [handsDetected, setHandsDetected] = useState(0);
  const [currentHandData, setCurrentHandData] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [modelStatus, setModelStatus] = useState({ is_trained: false, labels: [] });

  const letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'];

  // Listen for hand landmarks
  useEffect(() => {
    socket.on('hand_landmarks', (data) => {
      if (data.success && data.hands_detected > 0) {
        setHandsDetected(data.hands_detected);
        setCurrentHandData(data.hands[0]); // Store first hand
      } else {
        setHandsDetected(0);
        setCurrentHandData(null);
      }
    });

    socket.on('sample_saved', (data) => {
      if (data.success) {
        setSampleCounts(data.sample_counts);
        setMessage(`✅ Saved sample for letter "${data.label}" (Total: ${data.sample_counts[data.label]})`);
        setTimeout(() => setMessage(''), 3000);
      } else {
        setMessage(`❌ Error: ${data.error}`);
      }
    });

    socket.on('training_complete', (data) => {
      setIsTraining(false);
      if (data.success) {
        setMessage(`✅ Training complete! Model can recognize: ${data.labels.join(', ')}`);
        setModelStatus({ is_trained: true, labels: data.labels });
      } else {
        setMessage(`❌ Training failed: ${data.error}`);
      }
    });

    socket.on('model_status', (data) => {
      setModelStatus(data);
      setSampleCounts(data.sample_counts);
    });

    return () => {
      socket.off('hand_landmarks');
      socket.off('sample_saved');
      socket.off('training_complete');
      socket.off('model_status');
    };
  }, []);

  // Start capturing frames
  const startCapturing = () => {
    setIsCapturing(true);
    const interval = setInterval(() => {
      if (webcamRef.current) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          socket.emit('process_frame', { frame: imageSrc });
        }
      }
    }, 100);

    return () => clearInterval(interval);
  };

  useEffect(() => {
    if (isCapturing) {
      const cleanup = startCapturing();
      return cleanup;
    }
  }, [isCapturing]);

  const saveSample = () => {
    if (!currentHandData) {
      setMessage('❌ No hand detected! Show your hand to the camera.');
      setTimeout(() => setMessage(''), 3000);
      return;
    }

    socket.emit('save_training_sample', {
      landmarks: currentHandData.landmarks,
      label: currentLetter
    });
  };

  const trainModel = () => {
    const totalSamples = Object.values(sampleCounts).reduce((a, b) => a + b, 0);
    
    if (totalSamples < 10) {
      setMessage('❌ Need at least 10 total samples to train! (Recommend 20+ per letter)');
      setTimeout(() => setMessage(''), 4000);
      return;
    }

    setIsTraining(true);
    setMessage('🎓 Training model... This may take a few seconds...');
    socket.emit('train_model', {});
  };

  const getSampleCount = (letter) => {
    return sampleCounts[letter] || 0;
  };

  const getTotalSamples = () => {
    return Object.values(sampleCounts).reduce((a, b) => a + b, 0);
  };

  return (
    <div style={{ textAlign: 'center', padding: '20px' }}>
      <h1>🎓 Training Mode - Collect Letter Samples</h1>
      <p style={{ color: '#666', marginBottom: '20px' }}>
        Collect samples of different ASL letters to train the recognition model
      </p>

      {/* Model Status */}
      <div style={{
        backgroundColor: modelStatus.is_trained ? '#d4edda' : '#fff3cd',
        padding: '15px',
        borderRadius: '10px',
        marginBottom: '20px',
        border: `2px solid ${modelStatus.is_trained ? '#28a745' : '#ffc107'}`
      }}>
        <strong>Model Status:</strong> {modelStatus.is_trained ? 
          `✅ Trained (Recognizes: ${modelStatus.labels.join(', ')})` : 
          '⚠️ Not Trained'}
        <br />
        <strong>Total Samples:</strong> {getTotalSamples()}
      </div>

      {/* Webcam */}
      <div style={{ marginBottom: '20px' }}>
        <Webcam
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          style={{
            width: '640px',
            height: '480px',
            border: '3px solid #007bff',
            borderRadius: '10px'
          }}
        />
        
        <div style={{
          marginTop: '10px',
          padding: '10px',
          backgroundColor: handsDetected > 0 ? '#d4edda' : '#f8d7da',
          borderRadius: '8px',
          display: 'inline-block'
        }}>
          {handsDetected > 0 ? '✅ Hand Detected' : '❌ No Hand Detected'}
        </div>
      </div>

      {/* Capture Controls */}
      <div style={{ marginBottom: '20px' }}>
        {!isCapturing ? (
          <button
            onClick={() => setIsCapturing(true)}
            style={{
              padding: '15px 30px',
              fontSize: '18px',
              backgroundColor: '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}
          >
            ▶️ Start Capturing
          </button>
        ) : (
          <button
            onClick={() => setIsCapturing(false)}
            style={{
              padding: '15px 30px',
              fontSize: '18px',
              backgroundColor: '#dc3545',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}
          >
            ⏹️ Stop Capturing
          </button>
        )}
      </div>

      {/* Letter Selection */}
      <div style={{
        backgroundColor: '#f8f9fa',
        padding: '20px',
        borderRadius: '10px',
        marginBottom: '20px'
      }}>
        <h3>Select Letter to Capture:</h3>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(60px, 1fr))',
          gap: '10px',
          maxWidth: '800px',
          margin: '0 auto'
        }}>
          {letters.map(letter => (
            <button
              key={letter}
              onClick={() => setCurrentLetter(letter)}
              style={{
                padding: '15px',
                fontSize: '20px',
                fontWeight: 'bold',
                backgroundColor: currentLetter === letter ? '#007bff' : '#fff',
                color: currentLetter === letter ? 'white' : '#333',
                border: '2px solid #007bff',
                borderRadius: '8px',
                cursor: 'pointer',
                position: 'relative'
              }}
            >
              {letter}
              <div style={{
                fontSize: '10px',
                marginTop: '5px',
                color: currentLetter === letter ? '#fff' : '#666'
              }}>
                {getSampleCount(letter)} samples
              </div>
            </button>
          ))}
        </div>

        <div style={{ marginTop: '20px' }}>
          <button
            onClick={saveSample}
            disabled={!isCapturing || handsDetected === 0}
            style={{
              padding: '20px 40px',
              fontSize: '24px',
              backgroundColor: (isCapturing && handsDetected > 0) ? '#007bff' : '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              cursor: (isCapturing && handsDetected > 0) ? 'pointer' : 'not-allowed',
              fontWeight: 'bold'
            }}
          >
            💾 Save Sample for "{currentLetter}"
          </button>
        </div>
      </div>

      {/* Train Button */}
      <div style={{ marginBottom: '20px' }}>
        <button
          onClick={trainModel}
          disabled={isTraining || getTotalSamples() < 10}
          style={{
            padding: '20px 40px',
            fontSize: '20px',
            backgroundColor: (getTotalSamples() >= 10 && !isTraining) ? '#28a745' : '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '10px',
            cursor: (getTotalSamples() >= 10 && !isTraining) ? 'pointer' : 'not-allowed',
            fontWeight: 'bold'
          }}
        >
          {isTraining ? '⏳ Training...' : '🎓 Train Model'}
        </button>
      </div>

      {/* Message */}
      {message && (
        <div style={{
          padding: '15px',
          backgroundColor: message.includes('✅') ? '#d4edda' : '#f8d7da',
          color: message.includes('✅') ? '#155724' : '#721c24',
          borderRadius: '8px',
          marginTop: '20px',
          fontWeight: 'bold'
        }}>
          {message}
        </div>
      )}

      {/* Instructions */}
      <div style={{
        marginTop: '30px',
        padding: '20px',
        backgroundColor: '#e7f3ff',
        borderRadius: '10px',
        textAlign: 'left',
        maxWidth: '600px',
        margin: '30px auto'
      }}>
        <h3>📋 Instructions:</h3>
        <ol style={{ lineHeight: '2' }}>
          <li>Click "Start Capturing"</li>
          <li>Select a letter button (e.g., "A")</li>
          <li>Make the ASL sign for that letter</li>
          <li>Click "Save Sample" when hand is detected</li>
          <li>Repeat 20-30 times per letter (vary hand position slightly)</li>
          <li>Do this for at least 3-5 different letters</li>
          <li>Click "Train Model" when you have enough samples</li>
          <li>Go to "Hand Tracking" to test recognition!</li>
        </ol>
      </div>
    </div>
  );
}

export default TrainingMode;