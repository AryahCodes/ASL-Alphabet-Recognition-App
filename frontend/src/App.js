import React, { useState } from 'react';
import HandTracking from './HandTracking';
import TrainingMode from './TrainingMode';
import WebcamCapture from './WebcamCapture';
import SocketTest from './SocketTest';
import { useBackendStatus } from './useBackendStatus';
import './App.css';

const views = [
  { id: 'recognizer', label: 'Recognizer' },
  { id: 'training', label: 'Training' },
  { id: 'diagnostics', label: 'Diagnostics' },
];

function App() {
  const [currentView, setCurrentView] = useState('recognizer');
  const backend = useBackendStatus();

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Real-time ASL alphabet recognition</p>
          <h1>SignApp</h1>
        </div>
        <nav className="view-tabs" aria-label="Application views">
          {views.map((view) => (
            <button
              key={view.id}
              type="button"
              className={currentView === view.id ? 'tab-button active' : 'tab-button'}
              onClick={() => setCurrentView(view.id)}
              aria-current={currentView === view.id ? 'page' : undefined}
            >
              {view.label}
            </button>
          ))}
        </nav>
      </header>

      <main>
        {currentView === 'recognizer' && <HandTracking backend={backend} />}
        {currentView === 'training' && <TrainingMode backend={backend} />}
        {currentView === 'diagnostics' && (
          <div className="diagnostics-grid">
            <SocketTest backend={backend} />
            <WebcamCapture />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
