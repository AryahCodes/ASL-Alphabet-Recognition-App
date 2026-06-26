import { useCallback, useEffect, useMemo, useState } from 'react';
import socket from './socket';
import BACKEND_URL from './config';

const EMPTY_MODEL = {
  ready: false,
  loaded: false,
  labels: [],
  label_count: 0,
  backend: null,
  initialization_error: null,
};

export function useBackendStatus() {
  const [socketState, setSocketState] = useState(socket.connected ? 'connected' : 'connecting');
  const [socketError, setSocketError] = useState(null);
  const [modelStatus, setModelStatus] = useState(EMPTY_MODEL);
  const [serviceStatus, setServiceStatus] = useState(null);
  const [lastCheckedAt, setLastCheckedAt] = useState(null);

  const refreshStatus = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/version`, { cache: 'no-store' });
      const data = await response.json();
      setServiceStatus(data);
      if (data.model) {
        setModelStatus((current) => ({
          ...current,
          ...data.model,
          ready: Boolean(data.model.ready || data.model.loaded),
          loaded: Boolean(data.model.loaded),
          labels: current.labels || [],
        }));
      }
      setLastCheckedAt(new Date());
      setSocketError(null);
      return data;
    } catch (error) {
      setSocketError(error.message);
      setServiceStatus(null);
      return null;
    }
  }, []);

  useEffect(() => {
    const handleConnect = () => {
      setSocketState('connected');
      setSocketError(null);
      refreshStatus();
    };
    const handleDisconnect = (reason) => {
      setSocketState('disconnected');
      setSocketError(reason || 'Disconnected');
    };
    const handleConnectError = (error) => {
      setSocketState('disconnected');
      setSocketError(error.message);
    };
    const handleModelStatus = (data) => {
      setModelStatus({ ...EMPTY_MODEL, ...data });
      setLastCheckedAt(new Date());
    };

    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);
    socket.on('connect_error', handleConnectError);
    socket.on('model_status', handleModelStatus);

    if (socket.connected) {
      handleConnect();
    } else {
      setSocketState('connecting');
      refreshStatus();
    }

    const interval = setInterval(refreshStatus, 15000);

    return () => {
      socket.off('connect', handleConnect);
      socket.off('disconnect', handleDisconnect);
      socket.off('connect_error', handleConnectError);
      socket.off('model_status', handleModelStatus);
      clearInterval(interval);
    };
  }, [refreshStatus]);

  return useMemo(() => {
    const backendLive = Boolean(serviceStatus?.live || socket.connected);
    const modelReady = Boolean(modelStatus.ready || modelStatus.is_trained);
    const serviceReady = Boolean(serviceStatus?.ready ?? modelReady);
    return {
      backendUrl: BACKEND_URL,
      socketState,
      socketConnected: socketState === 'connected',
      backendLive,
      serviceReady,
      modelReady,
      modelStatus,
      serviceStatus,
      socketError,
      lastCheckedAt,
      refreshStatus,
    };
  }, [lastCheckedAt, modelStatus, refreshStatus, serviceStatus, socketError, socketState]);
}
