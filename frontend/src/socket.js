import { io } from 'socket.io-client';
import BACKEND_URL from './config';

const socket = io(BACKEND_URL, {
  transports: ['polling', 'websocket'],
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  reconnectionAttempts: Infinity,
  timeout: 30000,
});

socket.on('connect', () => {
  console.log('Socket connected! ID:', socket.id);
});

socket.on('disconnect', (reason) => {
  console.log('Socket disconnected! Reason:', reason);
});

socket.on('connect_error', (error) => {
  console.error('Connection error:', error.message);
});

export default socket;