import React, { useEffect, useState } from 'react';
import socket from './socket';

function SocketTest() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const handleConnect = () => {
      setIsConnected(true);
    };
    const handleDisconnect = () => {
      setIsConnected(false);
    };
    const handleResponse = (data) => {
      setMessages(prev => [...prev, data.message]);
    };
    const handleConnectError = (error) => {
      console.error('Connection error:', error.message);
      setIsConnected(false);
    };

    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);
    socket.on('response', handleResponse);
    socket.on('connect_error', handleConnectError);

    setIsConnected(socket.connected);

    return () => {
      socket.off('connect', handleConnect);
      socket.off('disconnect', handleDisconnect);
      socket.off('response', handleResponse);
      socket.off('connect_error', handleConnectError);
    };
  }, []);

  const sendMessage = () => {
    if (inputMessage.trim()) {
      console.log('📤 Sending to backend:', inputMessage);
      socket.emit('test_message', inputMessage);
      setInputMessage('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  return (
    <div style={{ 
      padding: '30px', 
      backgroundColor: 'white', 
      borderRadius: '15px',
      boxShadow: '0 8px 16px rgba(0,0,0,0.1)',
      maxWidth: '600px',
      margin: '20px auto'
    }}>
      <h2 style={{ marginTop: 0 }}>
        🔌 Socket.IO Connection Test
      </h2>
      
      <div style={{ 
        padding: '10px', 
        borderRadius: '8px',
        backgroundColor: isConnected ? '#d4edda' : '#fff3cd',
        color: isConnected ? '#155724' : '#856404',
        marginBottom: '20px',
        fontWeight: 'bold'
      }}>
        Status: {isConnected ? '✅ Connected' : '⏳ Connecting... (backend may be waking up, ~30s)'}
      </div>
      
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type a message..."
          style={{ 
            flex: 1,
            padding: '12px', 
            fontSize: '16px',
            border: '2px solid #ddd',
            borderRadius: '8px',
            outline: 'none'
          }}
        />
        
        <button 
          onClick={sendMessage}
          disabled={!isConnected}
          style={{ 
            padding: '12px 24px',
            fontSize: '16px',
            cursor: isConnected ? 'pointer' : 'not-allowed',
            backgroundColor: isConnected ? '#007bff' : '#ccc',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            fontWeight: 'bold'
          }}
        >
          Send 📤
        </button>
      </div>

      <div style={{ 
        backgroundColor: '#f8f9fa',
        padding: '15px',
        borderRadius: '8px',
        minHeight: '150px',
        maxHeight: '300px',
        overflowY: 'auto'
      }}>
        <h3 style={{ marginTop: 0, fontSize: '16px', color: '#666' }}>
          Messages from Backend:
        </h3>
        {messages.length === 0 ? (
          <p style={{ color: '#999', fontStyle: 'italic' }}>
            No messages yet. Send one to test the connection!
          </p>
        ) : (
          messages.map((msg, index) => (
            <div 
              key={index}
              style={{
                padding: '10px',
                marginBottom: '8px',
                backgroundColor: 'white',
                borderRadius: '6px',
                borderLeft: '4px solid #007bff'
              }}
            >
              {msg}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default SocketTest;