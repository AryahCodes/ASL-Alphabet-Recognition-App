import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';

function WebcamCapture() {
  const webcamRef = useRef(null);
  const [imgSrc, setImgSrc] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const [cameraEnabled, setCameraEnabled] = useState(false);

  const capture = () => {
    const imageSrc = webcamRef.current?.getScreenshot();
    setImgSrc(imageSrc || null);
  };

  return (
    <section className="diagnostic-panel" aria-labelledby="camera-test-title">
      <p className="eyebrow">Diagnostics</p>
      <h2 id="camera-test-title">Camera test</h2>
      <div className="camera-stage compact">
        {cameraEnabled ? (
          <Webcam
            ref={webcamRef}
            audio={false}
            mirrored
            screenshotFormat="image/jpeg"
            screenshotQuality={0.75}
            videoConstraints={{ facingMode: 'user', width: 640, height: 480 }}
            onUserMedia={() => setCameraError(null)}
            onUserMediaError={(error) => setCameraError(error.message || 'Camera permission denied.')}
            className="camera-video"
          />
        ) : (
          <div className="camera-placeholder">
            <strong>Camera is off</strong>
            <span>Enable it to test capture.</span>
          </div>
        )}
      </div>

      {cameraError && <p className="inline-message error" role="alert">{cameraError}</p>}
      <button
        className="secondary-action"
        onClick={() => (cameraEnabled ? capture() : setCameraEnabled(true))}
        disabled={Boolean(cameraError)}
      >
        {cameraEnabled ? 'Capture photo' : 'Enable camera'}
      </button>

      {imgSrc && (
        <div className="snapshot">
          <img src={imgSrc} alt="Captured webcam frame" />
        </div>
      )}
    </section>
  );
}

export default WebcamCapture;
