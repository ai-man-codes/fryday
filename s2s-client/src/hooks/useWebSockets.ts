import { useState, useEffect, useRef, useCallback } from 'react';
import type { WebSocketState, WebSocketMessage, MessageType } from '../types/translation';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (message: WebSocketMessage) => void;
  onError?: (error: Event) => void;
  onOpen?: () => void;
  onClose?: () => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export const useWebSocket = ({
  url,
  onMessage,
  onError,
  onOpen,
  onClose,
  reconnectInterval = 3000,
  maxReconnectAttempts = 5,
}: UseWebSocketOptions) => {
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    connectionId: null,
    lastPing: 0,
    reconnectAttempts: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setState(prev => ({
          ...prev,
          isConnected: true,
          reconnectAttempts: 0,
        }));
        onOpen?.();

        // Start ping-pong for connection health
        pingIntervalRef.current = setInterval(() => {
          sendMessage('ping', {});
          setState(prev => ({ ...prev, lastPing: Date.now() }));
        }, 30000);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          console.log('Received message:', message.type, message);

          if (message.type === 'pong') {
            setState(prev => ({ ...prev, lastPing: Date.now() }));
          }

          onMessage?.(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.(error);
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        setState(prev => ({
          ...prev,
          isConnected: false,
          connectionId: null,
        }));
        onClose?.();

        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
      };
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
    }
  }, [url, onMessage, onError, onOpen, onClose]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendMessage = useCallback(<T = any>(type: MessageType, payload: T) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const message: WebSocketMessage = {
        type,
        payload,
        timestamp: Date.now(),
      };
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, []);

  const sendAudioChunk = useCallback((audioData: Float32Array, sampleRate: number) => {
    // Convert Float32Array to base64 for transmission
    const buffer = audioData.buffer.slice(
      audioData.byteOffset,
      audioData.byteOffset + audioData.byteLength
    );
    const base64Audio = btoa(String.fromCharCode(...new Uint8Array(buffer)));

    return sendMessage('audio_chunk', {
      audio_data: base64Audio,
      sample_rate: sampleRate,
      timestamp: Date.now(),
    });
  }, [sendMessage]);

  const joinRoom = useCallback((roomId: string, languagePair?: { from: string; to: string }) => {
    return sendMessage('join_room', {
      room_id: roomId,
      language_pair: languagePair,
    });
  }, [sendMessage]);

  const leaveRoom = useCallback((roomId: string) => {
    return sendMessage('leave_room', { room_id: roomId });
  }, [sendMessage]);

  // Auto-reconnect logic
  useEffect(() => {
    if (!state.isConnected && state.reconnectAttempts < maxReconnectAttempts) {
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log(`Attempting to reconnect... (${state.reconnectAttempts + 1}/${maxReconnectAttempts})`);
        setState(prev => ({ ...prev, reconnectAttempts: prev.reconnectAttempts + 1 }));
        connect();
      }, reconnectInterval);
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [state.isConnected, state.reconnectAttempts, connect, reconnectInterval, maxReconnectAttempts]);

  // Cleanup on unmount
  useEffect(() => {
    return disconnect;
  }, [disconnect]);

  return {
    state,
    connect,
    disconnect,
    sendMessage,
    sendAudioChunk,
    joinRoom,
    leaveRoom,
    isReady: wsRef.current?.readyState === WebSocket.OPEN,
  };
};
