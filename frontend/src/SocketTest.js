import React, { useEffect, useState } from 'react';
import socket from './socket';

function SocketTest({ backend }) {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');

  useEffect(() => {
    const handleResponse = (data) => setMessages((prev) => [...prev, data.message]);
    socket.on('response', handleResponse);
    return () => socket.off('response', handleResponse);
  }, []);

  const sendMessage = () => {
    if (!inputMessage.trim() || !backend.socketConnected) return;
    socket.emit('test_message', inputMessage.trim());
    setInputMessage('');
  };

  return (
    <section className="diagnostic-panel" aria-labelledby="socket-title">
      <p className="eyebrow">Diagnostics</p>
      <h2 id="socket-title">Backend connection</h2>
      <div className="status-grid">
        <div className={`status-pill ${backend.socketConnected ? 'good' : 'warn'}`}>
          <span>Socket</span>
          <strong>{backend.socketConnected ? 'Connected' : 'Connecting'}</strong>
        </div>
        <div className={`status-pill ${backend.modelReady ? 'good' : 'bad'}`}>
          <span>Model</span>
          <strong>{backend.modelReady ? 'Ready' : 'Unavailable'}</strong>
        </div>
      </div>
      <p className="muted-text">{backend.backendUrl}</p>

      <div className="input-row">
        <input
          type="text"
          value={inputMessage}
          onChange={(event) => setInputMessage(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') sendMessage();
          }}
          placeholder="Send a test message"
          aria-label="Test message"
        />
        <button className="secondary-action" onClick={sendMessage} disabled={!backend.socketConnected}>
          Send
        </button>
      </div>

      <div className="message-log" aria-live="polite">
        {messages.length === 0 ? (
          <p className="muted-text">No backend echo messages yet.</p>
        ) : (
          messages.map((message, index) => <div key={`${message}-${index}`}>{message}</div>)
        )}
      </div>
    </section>
  );
}

export default SocketTest;
