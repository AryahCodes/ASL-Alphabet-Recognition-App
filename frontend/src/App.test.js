import { render, screen } from '@testing-library/react';
import App from './App';

jest.mock('react-webcam', () => {
  const React = require('react');
  return React.forwardRef((props, ref) => {
    React.useImperativeHandle(ref, () => ({
      getScreenshot: () => 'data:image/jpeg;base64,abc',
      video: { videoWidth: 640, videoHeight: 480 },
    }));
    React.useEffect(() => {
      props.onUserMedia?.();
    }, []);
    return <div data-testid="webcam" className={props.className} />;
  });
});

jest.mock('./socket', () => ({
  connected: false,
  on: jest.fn(),
  off: jest.fn(),
  emit: jest.fn(),
}));

beforeEach(() => {
  global.fetch = jest.fn(() => Promise.resolve({
    json: () => Promise.resolve({
      live: true,
      ready: false,
      model: {
        loaded: false,
        ready: false,
        backend: 'numpy',
        initialization_error: 'Model file not found',
      },
    }),
  }));
});

test('renders recognizer as the default product view', async () => {
  render(<App />);

  expect(screen.getByRole('heading', { name: 'SignApp' })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: /Practice ASL letters/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Enable camera/i })).toBeEnabled();
  expect(screen.queryByTestId('webcam')).not.toBeInTheDocument();
  expect(await screen.findByText(/Model initialization failed/i)).toBeInTheDocument();
});

test('surfaces model initialization failure', async () => {
  render(<App />);

  expect(await screen.findByText(/Model initialization failed/i)).toBeInTheDocument();
  expect(screen.getByText(/Model file not found/i)).toBeInTheDocument();
});
