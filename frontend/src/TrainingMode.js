import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import socket from './socket';

const letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'];

function TrainingMode({ backend }) {
  const webcamRef = useRef(null);
  const intervalRef = useRef(null);
  const awaitingFrameRef = useRef(false);
  const [currentLetter, setCurrentLetter] = useState('A');
  const [sampleCounts, setSampleCounts] = useState({});
  const [isCapturing, setIsCapturing] = useState(false);
  const [message, setMessage] = useState('');
  const [handsDetected, setHandsDetected] = useState(0);
  const [currentHandData, setCurrentHandData] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [cameraState, setCameraState] = useState('pending');
  const [cameraError, setCameraError] = useState(null);
  const [messageType, setMessageType] = useState('info');
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const trainingEnabled = backend.serviceStatus?.training_enabled === true;

  useEffect(() => {
    const handleHandLandmarks = (data) => {
      awaitingFrameRef.current = false;
      if (data.success && data.hands_detected > 0) {
        setHandsDetected(data.hands_detected);
        setCurrentHandData(data.hands[0]);
      } else {
        setHandsDetected(0);
        setCurrentHandData(null);
      }
    };
    const handleSampleSaved = (data) => {
      if (data.success) {
        setSampleCounts(data.sample_counts);
        setMessage(`Saved sample for ${data.label}. Total: ${data.sample_counts[data.label]}`);
        setMessageType('info');
      } else {
        setMessage(`Sample was not saved: ${data.error}`);
        setMessageType('error');
      }
    };
    const handleTrainingComplete = (data) => {
      setIsTraining(false);
      setMessage(data.success ? `Training complete for ${data.labels.join(', ')}` : `Training unavailable: ${data.error}`);
      setMessageType(data.success ? 'info' : 'error');
    };
    const handleModelStatus = (data) => setSampleCounts(data.sample_counts || {});

    socket.on('hand_landmarks', handleHandLandmarks);
    socket.on('sample_saved', handleSampleSaved);
    socket.on('training_complete', handleTrainingComplete);
    socket.on('model_status', handleModelStatus);

    return () => {
      socket.off('hand_landmarks', handleHandLandmarks);
      socket.off('sample_saved', handleSampleSaved);
      socket.off('training_complete', handleTrainingComplete);
      socket.off('model_status', handleModelStatus);
    };
  }, []);

  useEffect(() => {
    if (!isCapturing) {
      awaitingFrameRef.current = false;
      clearInterval(intervalRef.current);
      intervalRef.current = null;
      return undefined;
    }

    intervalRef.current = setInterval(() => {
      if (!webcamRef.current || !socket.connected || awaitingFrameRef.current) return;
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        awaitingFrameRef.current = true;
        socket.emit('process_frame', { frame: imageSrc });
      }
    }, 180);

    return () => {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    };
  }, [isCapturing]);

  const saveSample = () => {
    if (!currentHandData) {
      setMessage('No hand detected. Show your hand clearly before saving.');
      setMessageType('error');
      return;
    }
    socket.emit('save_training_sample', {
      landmarks: currentHandData.landmarks,
      label: currentLetter,
    });
  };

  const trainModel = () => {
    setIsTraining(true);
    setMessage('Requesting training...');
    setMessageType('info');
    socket.emit('train_model', {});
  };

  const totalSamples = Object.values(sampleCounts).reduce((sum, count) => sum + count, 0);
  const canCapture = trainingEnabled && backend.socketConnected && cameraState === 'ready';

  return (
    <section className="training-layout" aria-labelledby="training-title">
      <div className="panel-copy">
        <p className="eyebrow">Training data</p>
        <h2 id="training-title">Collect labeled landmark samples.</h2>
        <p>
          The deployed professional model is trained offline. This panel is useful for local RandomForest experiments and sample collection.
        </p>
        <div className="notice info">
          Model: {backend.modelReady ? 'ready' : 'not ready'} | Samples: {totalSamples}
        </div>
        {!trainingEnabled && (
          <div className="notice warning" role="status">
            Training data collection is disabled on this deployment.
          </div>
        )}
      </div>

      <div className="training-grid">
        <div className="camera-panel">
          <div className="camera-stage compact">
            {cameraEnabled ? (
              <Webcam
                ref={webcamRef}
                audio={false}
                mirrored
                screenshotFormat="image/jpeg"
                screenshotQuality={0.65}
                videoConstraints={{ facingMode: 'user', width: 640, height: 480 }}
                onUserMedia={() => {
                  setCameraState('ready');
                  setCameraError(null);
                }}
                onUserMediaError={(error) => {
                  setCameraState('error');
                  setCameraError(error.message || 'Camera permission denied.');
                }}
                className="camera-video"
              />
            ) : (
              <div className="camera-placeholder">
                <strong>Camera is off</strong>
                <span>Enable local training to collect samples.</span>
              </div>
            )}
            <div className={`camera-status ${handsDetected > 0 ? 'good' : 'warn'}`}>
              {handsDetected > 0 ? 'Hand detected' : 'No hand'}
            </div>
          </div>

          <div className="control-row">
            <button
              className={isCapturing ? 'danger-action' : 'primary-action'}
              onClick={() => {
                if (!cameraEnabled) {
                  setCameraEnabled(true);
                  setCameraState('pending');
                  return;
                }
                setIsCapturing((value) => !value);
              }}
              disabled={!trainingEnabled || (cameraEnabled && !isCapturing && !canCapture)}
            >
              {isCapturing ? 'Stop capture' : cameraEnabled ? 'Start capture' : 'Enable camera'}
            </button>
            <button className="secondary-action" onClick={saveSample} disabled={!isCapturing || handsDetected === 0}>
              Save {currentLetter}
            </button>
          </div>
          {cameraError && <p className="inline-message error" role="alert">{cameraError}</p>}
        </div>

        <div className="letter-panel">
          <h3>Select a letter</h3>
          <div className="letter-grid">
            {letters.map((letter) => (
              <button
                key={letter}
                type="button"
                className={currentLetter === letter ? 'letter-button active' : 'letter-button'}
                onClick={() => setCurrentLetter(letter)}
              >
                <span>{letter}</span>
                <small>{sampleCounts[letter] || 0}</small>
              </button>
            ))}
          </div>

          <button className="secondary-action full-width" onClick={trainModel} disabled={!trainingEnabled || isTraining || totalSamples < 10}>
            {isTraining ? 'Training...' : 'Train local model'}
          </button>
          {message && (
            <p className={messageType === 'error' ? 'inline-message error' : 'inline-message'} role={messageType === 'error' ? 'alert' : 'status'}>
              {message}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

export default TrainingMode;
