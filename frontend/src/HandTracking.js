import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import socket from './socket';

const connections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

function HandTracking({ backend }) {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const awaitingFrameRef = useRef(false);
  const fpsCounterRef = useRef({ frames: 0, lastTime: Date.now() });

  const [isTracking, setIsTracking] = useState(false);
  const [cameraState, setCameraState] = useState('pending');
  const [cameraError, setCameraError] = useState(null);
  const [handsDetected, setHandsDetected] = useState(0);
  const [fps, setFps] = useState(0);
  const [predictedLetter, setPredictedLetter] = useState(null);
  const [predictionConfidence, setPredictionConfidence] = useState(0);
  const [recognitionMessage, setRecognitionMessage] = useState('Show your hand in the camera frame.');
  const [recognitionError, setRecognitionError] = useState(null);
  const [cameraEnabled, setCameraEnabled] = useState(false);

  const handTrackingError = backend.serviceStatus?.hand_tracking?.initialization_error;
  const readyToTrack = backend.socketConnected && backend.serviceReady && cameraState === 'ready';

  const primaryStatus = useMemo(() => {
    if (!backend.socketConnected) return { label: 'Disconnected', tone: 'bad' };
    if (!backend.modelReady) {
      return backend.modelStatus.initialization_error
        ? { label: 'Model error', tone: 'bad' }
        : { label: 'Model loading', tone: 'warn' };
    }
    if (handTrackingError) return { label: 'Tracking error', tone: 'bad' };
    if (!cameraEnabled) return { label: 'Camera off', tone: 'warn' };
    if (cameraState === 'error') return { label: 'Camera unavailable', tone: 'bad' };
    if (cameraState !== 'ready') return { label: 'Camera pending', tone: 'warn' };
    if (recognitionError) return { label: 'Recognition error', tone: 'bad' };
    return { label: 'Ready', tone: 'good' };
  }, [backend.modelReady, backend.modelStatus.initialization_error, backend.socketConnected, cameraEnabled, cameraState, handTrackingError, recognitionError]);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const drawHands = useCallback((hands) => {
    const canvas = canvasRef.current;
    const video = webcamRef.current?.video;
    if (!canvas || !video || !video.videoWidth) return;

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    hands.forEach((hand) => {
      const landmarks = hand.landmarks;
      ctx.strokeStyle = hand.handedness === 'Right' ? '#0f766e' : '#7c3aed';
      ctx.lineWidth = 3;

      connections.forEach(([start, end]) => {
        const startPoint = landmarks[start];
        const endPoint = landmarks[end];
        ctx.beginPath();
        ctx.moveTo(startPoint.x * canvas.width, startPoint.y * canvas.height);
        ctx.lineTo(endPoint.x * canvas.width, endPoint.y * canvas.height);
        ctx.stroke();
      });

      landmarks.forEach((landmark, index) => {
        const x = landmark.x * canvas.width;
        const y = landmark.y * canvas.height;
        ctx.beginPath();
        ctx.arc(x, y, index === 0 ? 6 : 4, 0, 2 * Math.PI);
        ctx.fillStyle = index === 0 ? '#ef4444' : '#14b8a6';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    });

    fpsCounterRef.current.frames += 1;
    const now = Date.now();
    const elapsed = now - fpsCounterRef.current.lastTime;
    if (elapsed >= 1000) {
      setFps(fpsCounterRef.current.frames);
      fpsCounterRef.current.frames = 0;
      fpsCounterRef.current.lastTime = now;
    }
  }, []);

  const stopTracking = useCallback(() => {
    setIsTracking(false);
    awaitingFrameRef.current = false;
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setHandsDetected(0);
    setFps(0);
    setPredictedLetter(null);
    setPredictionConfidence(0);
    clearCanvas();
  }, [clearCanvas]);

  useEffect(() => {
    const handleHandLandmarks = (data) => {
      awaitingFrameRef.current = false;
      if (!data.success) {
        setRecognitionError(data.error || 'Recognition failed.');
        setRecognitionMessage(data.retryable ? 'The backend is catching up.' : 'Recognition is unavailable right now.');
        setHandsDetected(0);
        setPredictedLetter(null);
        setPredictionConfidence(0);
        clearCanvas();
        return;
      }

      setRecognitionError(null);
      if (data.hands_detected > 0) {
        setHandsDetected(data.hands_detected);
        drawHands(data.hands);

        if (data.letter_prediction?.success) {
          setPredictedLetter(data.letter_prediction.letter);
          setPredictionConfidence(data.letter_prediction.confidence);
          setRecognitionMessage('Letter recognized.');
        } else if (data.letter_prediction?.message) {
          setRecognitionMessage(data.letter_prediction.message);
        } else {
          setRecognitionMessage('Hand detected. Hold steady for a prediction.');
        }
      } else {
        setHandsDetected(0);
        setPredictedLetter(null);
        setPredictionConfidence(0);
        setRecognitionMessage('No hand detected.');
        clearCanvas();
      }
    };

    socket.on('hand_landmarks', handleHandLandmarks);
    return () => socket.off('hand_landmarks', handleHandLandmarks);
  }, [clearCanvas, drawHands]);

  useEffect(() => {
    if (!readyToTrack && isTracking) {
      stopTracking();
    }
  }, [isTracking, readyToTrack, stopTracking]);

  useEffect(() => () => stopTracking(), [stopTracking]);

  const startTracking = () => {
    if (!readyToTrack) return;
    setIsTracking(true);
    setRecognitionError(null);
    setRecognitionMessage('Recognition active.');

    intervalRef.current = setInterval(() => {
      if (!webcamRef.current || !socket.connected || awaitingFrameRef.current) return;
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        awaitingFrameRef.current = true;
        socket.emit('process_frame', { frame: imageSrc });
      }
    }, 140);
  };

  return (
    <section className="recognizer-layout" aria-labelledby="recognizer-title">
      <div className="recognizer-copy">
        <p className="eyebrow">Recognizer</p>
        <h2 id="recognizer-title">Practice ASL letters with live feedback.</h2>
        <p>
          The app checks the backend, model, camera, and recognition loop separately so failures are clear.
        </p>

        <div className="status-grid" aria-live="polite">
          <StatusPill label="Socket" value={backend.socketConnected ? 'Connected' : 'Connecting'} tone={backend.socketConnected ? 'good' : 'warn'} />
          <StatusPill label="Model" value={backend.modelReady ? 'Ready' : 'Unavailable'} tone={backend.modelReady ? 'good' : 'bad'} />
          <StatusPill label="Camera" value={cameraState === 'ready' ? 'Ready' : cameraState === 'error' ? 'Unavailable' : 'Pending'} tone={cameraState === 'ready' ? 'good' : cameraState === 'error' ? 'bad' : 'warn'} />
          <StatusPill label="Recognition" value={primaryStatus.label} tone={primaryStatus.tone} />
        </div>

        {backend.modelStatus.initialization_error && (
          <div className="notice danger" role="alert">
            Model initialization failed: {backend.modelStatus.initialization_error}
          </div>
        )}
        {handTrackingError && (
          <div className="notice danger" role="alert">
            Hand tracking initialization failed: {handTrackingError}
          </div>
        )}
        {backend.socketError && !backend.socketConnected && (
          <div className="notice warning" role="status">
            Backend is not reachable yet. Render may still be waking up.
          </div>
        )}
        {cameraError && (
          <div className="notice danger" role="alert">
            Camera unavailable: {cameraError}
          </div>
        )}
      </div>

      <div className="camera-panel">
        <div className="camera-stage">
          {cameraEnabled ? (
            <Webcam
              ref={webcamRef}
              audio={false}
              mirrored
              screenshotFormat="image/jpeg"
              screenshotQuality={0.72}
              videoConstraints={{ facingMode: 'user', width: 640, height: 480 }}
              onUserMedia={() => {
                setCameraState('ready');
                setCameraError(null);
              }}
              onUserMediaError={(error) => {
                setCameraState('error');
                setCameraError(error.message || 'Permission denied or no camera found.');
              }}
              className="camera-video"
            />
          ) : (
            <div className="camera-placeholder">
              <strong>Camera is off</strong>
              <span>Enable it when you are ready to practice.</span>
            </div>
          )}
          <canvas ref={canvasRef} className="camera-canvas" />
          <div className={`camera-status ${primaryStatus.tone}`}>{primaryStatus.label}</div>
          {predictedLetter && (
            <div className="prediction-card" aria-live="polite">
              <span className="prediction-letter">{predictedLetter}</span>
              <span className="prediction-confidence">{Math.round(predictionConfidence * 100)}% confident</span>
            </div>
          )}
        </div>

        <div className="control-row">
          {!isTracking ? (
            <button
              className="primary-action"
              onClick={() => {
                if (!cameraEnabled) {
                  setCameraEnabled(true);
                  setCameraState('pending');
                  return;
                }
                startTracking();
              }}
              disabled={cameraEnabled && !readyToTrack}
            >
              {cameraEnabled ? 'Start recognition' : 'Enable camera'}
            </button>
          ) : (
            <button className="danger-action" onClick={stopTracking}>
              Stop recognition
            </button>
          )}
          <div className="runtime-stats" aria-live="polite">
            <span>{handsDetected} hand{handsDetected === 1 ? '' : 's'}</span>
            <span>{fps} FPS</span>
          </div>
        </div>

        <p className={recognitionError ? 'inline-message error' : 'inline-message'} role={recognitionError ? 'alert' : 'status'}>
          {recognitionError || recognitionMessage}
        </p>
      </div>
    </section>
  );
}

function StatusPill({ label, value, tone }) {
  return (
    <div className={`status-pill ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default HandTracking;
