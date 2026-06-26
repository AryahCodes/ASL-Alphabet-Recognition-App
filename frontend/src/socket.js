import { io } from 'socket.io-client';
import BACKEND_URL from './config';

const socket = io(BACKEND_URL, {
  transports: ['websocket', 'polling'],
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  reconnectionAttempts: Infinity,
  timeout: 30000,
  autoConnect: true,
});

socket.on('connect_error', (error) => {
  console.error('Connection error:', error.message);
});

export default socket;
