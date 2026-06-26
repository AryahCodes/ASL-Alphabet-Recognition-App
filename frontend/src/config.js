const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || (
  process.env.NODE_ENV === 'production'
    ? 'https://signapp-backend-5opq.onrender.com'
    : 'http://localhost:5001'
);

export default BACKEND_URL;
