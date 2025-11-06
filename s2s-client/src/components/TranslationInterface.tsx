import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import {
  Mic,
  MicOff,
  Video,
  VideoOff,
  Phone,
  PhoneOff,
  Settings,
  Globe,
  Users,
  Wifi,
  WifiOff
} from 'lucide-react';
import { useWebRTC } from '../hooks/useWebRTC';
import { useWebSocket } from '../hooks/useWebSockets';
import { useAudioProcessing } from '../hooks/useAudioProcessing';
import {
  type TranslationState,
  type TranslationResult,
  type WebSocketMessage,
  type RoomState,
  type AudioChunk,
  type ConnectedMessage,
  type WebRTCSignalMessage
} from '../types/translation';

const BACKEND_WS_URL = 'ws://localhost:8000/ws/translate'; // Adjust based on your FastAPI server

// Memoized video component for better performance
const RemoteVideo = React.memo<{ stream: MediaStream | null }>(({ stream }) => (
  <video
    ref={(video) => {
      if (video && stream) {
        video.srcObject = stream;
        // Performance optimizations
        video.setAttribute('playsinline', 'true');
        video.setAttribute('webkit-playsinline', 'true');
      }
    }}
    autoPlay
    playsInline
    muted={false}
    controls={false}
    disablePictureInPicture
    className="w-full h-full object-cover"
    style={{
      transform: 'translateZ(0)', // Force hardware acceleration
      willChange: 'transform',
    }}
  />
));

RemoteVideo.displayName = 'RemoteVideo';

// Memoized local video component
const LocalVideo = React.memo<{ stream: MediaStream | null }>(({ stream }) => (
  <div className="absolute top-4 right-4 w-32 h-24 bg-black rounded-lg overflow-hidden">
    <video
      ref={(video) => {
        if (video && stream) {
          video.srcObject = stream;
          video.muted = true;
          // Performance optimizations for local video
          video.setAttribute('playsinline', 'true');
          video.setAttribute('webkit-playsinline', 'true');
        }
      }}
      autoPlay
      playsInline
      muted
      controls={false}
      disablePictureInPicture
      className="w-full h-full object-cover"
      style={{
        transform: 'translateZ(0)', // Force hardware acceleration
        willChange: 'transform',
      }}
    />
  </div>
));

LocalVideo.displayName = 'LocalVideo';

export const TranslationInterface: React.FC = () => {
  // Refs to store hook instances for message handler
  const audioProcessingRef = useRef<any>(null);
  const webRTCRef = useRef<any>(null);

  const [translationState, setTranslationState] = useState<TranslationState>({
    isActive: false,
    isTranslating: false,
    currentTranslation: null,
    error: null,
    targetLanguage: 'fr', // Default to French
  });

  const [roomState, setRoomState] = useState<RoomState | null>(null);
  const [roomId, setRoomId] = useState<string>('');
  const [videoQuality, setVideoQuality] = useState<'low' | 'medium' | 'high'>('medium');

  // Performance monitoring for adaptive quality
  const performanceRef = useRef({
    frameCount: 0,
    lastTime: 0,
    qualityAdjusted: false,
  });

  const decodeBase64Audio = useCallback((base64Data: string): Promise<ArrayBuffer> => {
    return new Promise((resolve) => {
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      resolve(bytes.buffer);
    });
  }, []);

  const handleWebSocketMessage = useCallback(async (message: WebSocketMessage) => {
    switch (message.type) {
      case 'connected': {
        const connectedMsg: ConnectedMessage = message.payload;
        console.log('Connected with ID:', connectedMsg.connection_id);
        // Could store connection_id if needed for future use
        break;
      }

      case 'translation_result': {
        const result: TranslationResult = message.payload;
        setTranslationState(prev => ({
          ...prev,
          currentTranslation: result,
          isTranslating: false,
        }));

        // If there's audio data in the response, play it
        if (message.payload.audio_data && audioProcessingRef.current) {
          try {
            const audioBuffer = await decodeBase64Audio(message.payload.audio_data);
            const audioBufferObj = await audioProcessingRef.current.createAudioBuffer(
              new Float32Array(audioBuffer),
              16000
            );
            audioProcessingRef.current.addToPlaybackQueue(audioBufferObj);
            audioProcessingRef.current.processPlaybackQueue();
          } catch (error) {
            console.error('Error playing translated audio:', error);
          }
        }
        break;
      }

      case 'error':
        setTranslationState(prev => ({
          ...prev,
          error: message.payload.message,
          isTranslating: false,
        }));
        break;

      case 'room_state':
        setRoomState(message.payload);
        break;

      case 'webrtc_signal': {
        const signal: WebRTCSignalMessage = message.payload;
        console.log('Received WebRTC signal:', signal);

        // Handle WebRTC signaling
        if (webRTCRef.current) {
          if (signal.offer) {
            webRTCRef.current.createAnswer(signal.offer);
          } else if (signal.answer) {
            webRTCRef.current.addAnswer(signal.answer);
          } else if (signal.ice_candidate) {
            webRTCRef.current.addIceCandidate(signal.ice_candidate);
          }
        }
        break;
      }

      default:
        console.log('Unhandled message type:', message.type);
    }
  }, [decodeBase64Audio]);

  const ws = useWebSocket({
    url: BACKEND_WS_URL,
    onMessage: handleWebSocketMessage,
    onError: useCallback((error: Event) => {
      console.error('WebSocket error:', error);
      setTranslationState(prev => ({
        ...prev,
        error: 'Connection lost. Attempting to reconnect...',
      }));
    }, []),
    onOpen: useCallback(() => {
      console.log('WebSocket connected');
      setTranslationState(prev => ({ ...prev, error: null }));
    }, []),
    onClose: useCallback(() => {
      console.log('WebSocket disconnected');
      setTranslationState(prev => ({
        ...prev,
        isActive: false,
        error: 'Disconnected from server',
      }));
    }, []),
  });

  const webRTC = useWebRTC();
  webRTCRef.current = webRTC;

  const audioProcessing = useAudioProcessing({
    sampleRate: 16000,
    onAudioChunk: useCallback((chunk: AudioChunk) => {
      if (translationState.isActive && ws.isReady) {
        ws.sendAudioChunk(chunk.data, 16000);
      }
    }, [translationState.isActive, ws]),
    onAudioLevelChange: useCallback(() => {
      // Update UI with audio level if needed
    }, []),
  });
  audioProcessingRef.current = audioProcessing;

  const startTranslation = useCallback(async () => {
    try {
      setTranslationState(prev => ({ ...prev, error: null }));

      // Connect to WebSocket first
      ws.connect();

      // Start WebRTC stream
      const stream = await webRTC.startLocalStream(true, true);
      await webRTC.createPeerConnection();
      await webRTC.addLocalStreamToPeerConnection();

      // Start audio processing
      await audioProcessing.startRecording(stream);

      // Join room if specified and initiate WebRTC signaling
      if (roomId.trim()) {
        ws.joinRoom(roomId.trim(), {
          from: 'en', // Assume English input for now
          to: translationState.targetLanguage,
        });

        // Create and send WebRTC offer after joining room
        setTimeout(async () => {
          try {
            await webRTC.createOffer();
          } catch (error) {
            console.error('Error creating WebRTC offer:', error);
          }
        }, 1000); // Small delay to ensure room join is processed
      }

      setTranslationState(prev => ({ ...prev, isActive: true }));
    } catch (error) {
      console.error('Error starting translation:', error);
      setTranslationState(prev => ({
        ...prev,
        error: 'Failed to start translation. Please check permissions.',
      }));
    }
  }, [webRTC, audioProcessing, ws, roomId, translationState.targetLanguage]);

  const stopTranslation = useCallback(() => {
    // Stop audio processing
    audioProcessing.stopRecording();

    // Stop WebRTC
    webRTC.stopLocalStream();
    webRTC.closePeerConnection();

    // Disconnect WebSocket
    if (roomId.trim()) {
      ws.leaveRoom(roomId.trim());
    }
    ws.disconnect();

    setTranslationState(prev => ({
      ...prev,
      isActive: false,
      isTranslating: false,
      currentTranslation: null,
      error: null,
    }));

    setRoomState(null);
  }, [audioProcessing, webRTC, ws, roomId]);

  // Adaptive video quality based on performance
  useEffect(() => {
    if (!translationState.isActive || !webRTC.state.localStream) return;

    const monitorPerformance = () => {
      const now = performance.now();
      performanceRef.current.frameCount++;

      // Check performance every 5 seconds
      if (now - performanceRef.current.lastTime > 5000) {
        const fps = (performanceRef.current.frameCount * 1000) / (now - performanceRef.current.lastTime);

        // If FPS is below 10, reduce quality
        if (fps < 10 && videoQuality !== 'low' && !performanceRef.current.qualityAdjusted) {
          console.log('Low FPS detected, reducing video quality');
          webRTC.adjustVideoQuality('low');
          setVideoQuality('low');
          performanceRef.current.qualityAdjusted = true;
        }
        // If FPS is good and quality was reduced, try to increase it
        else if (fps > 20 && videoQuality === 'low' && performanceRef.current.qualityAdjusted) {
          console.log('Good FPS detected, increasing video quality');
          webRTC.adjustVideoQuality('medium');
          setVideoQuality('medium');
          performanceRef.current.qualityAdjusted = false;
        }

        performanceRef.current.frameCount = 0;
        performanceRef.current.lastTime = now;
      }

      if (translationState.isActive) {
        requestAnimationFrame(monitorPerformance);
      }
    };

    performanceRef.current.lastTime = performance.now();
    monitorPerformance();

    return () => {
      performanceRef.current.qualityAdjusted = false;
    };
  }, [translationState.isActive, webRTC, videoQuality]);

  const toggleVideo = useCallback(() => {
    webRTC.toggleVideo();
  }, [webRTC]);

  const toggleAudio = useCallback(() => {
    webRTC.toggleAudio();
  }, [webRTC]);

  const changeTargetLanguage = useCallback((language: string) => {
    setTranslationState(prev => ({ ...prev, targetLanguage: language }));
  }, []);

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Real-Time Speech Translator
          </h1>
          <p className="text-gray-600">
            Connect and communicate across languages instantly
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Video Area */}
          <div className="lg:col-span-2">
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Video Call</span>
                  <div className="flex items-center gap-2">
                    <Badge variant={ws.state.isConnected ? "default" : "destructive"}>
                      {ws.state.isConnected ? <Wifi className="w-3 h-3 mr-1" /> : <WifiOff className="w-3 h-3 mr-1" />}
                      {ws.state.isConnected ? 'Connected' : 'Disconnected'}
                    </Badge>
                    {roomState && (
                      <Badge variant="secondary">
                        <Users className="w-3 h-3 mr-1" />
                        {roomState.participants.length} participants
                      </Badge>
                    )}
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="relative aspect-video bg-gray-900 rounded-lg overflow-hidden">
                  {webRTC.state.remoteStream ? (
                    <RemoteVideo stream={webRTC.state.remoteStream} />
                  ) : (
                    <div className="flex items-center justify-center h-full text-white">
                      <div className="text-center">
                        <Video className="w-16 h-16 mx-auto mb-4 opacity-50" />
                        <p>No remote video</p>
                      </div>
                    </div>
                  )}

                  {/* Local video overlay */}
                  {webRTC.state.localStream && (
                    <LocalVideo stream={webRTC.state.localStream} />
                  )}

                  {/* Audio level indicator */}
                  {audioProcessing.state.isRecording && (
                    <div className="absolute bottom-4 left-4 right-4">
                      <div className="bg-black bg-opacity-50 rounded-lg p-2">
                        <div className="flex items-center gap-2 text-white text-sm">
                          <Mic className="w-4 h-4" />
                          <span>Audio Level</span>
                          <Progress
                            value={audioProcessing.state.audioLevel * 100}
                            className="flex-1 h-2"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Controls */}
                <div className="flex justify-center gap-4 mt-4">
                  <Button
                    variant={translationState.isActive ? "destructive" : "default"}
                    size="lg"
                    onClick={translationState.isActive ? stopTranslation : startTranslation}
                    className="px-8"
                  >
                    {translationState.isActive ? (
                      <>
                        <PhoneOff className="w-5 h-5 mr-2" />
                        Stop Translation
                      </>
                    ) : (
                      <>
                        <Phone className="w-5 h-5 mr-2" />
                        Start Translation
                      </>
                    )}
                  </Button>

                  <Button
                    variant="outline"
                    size="lg"
                    onClick={toggleVideo}
                    disabled={!translationState.isActive}
                  >
                    {webRTC.state.isVideoEnabled ? (
                      <Video className="w-5 h-5" />
                    ) : (
                      <VideoOff className="w-5 h-5" />
                    )}
                  </Button>

                  <Button
                    variant="outline"
                    size="lg"
                    onClick={toggleAudio}
                    disabled={!translationState.isActive}
                  >
                    {webRTC.state.isAudioEnabled ? (
                      <Mic className="w-5 h-5" />
                    ) : (
                      <MicOff className="w-5 h-5" />
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Room Settings */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Settings className="w-5 h-5 mr-2" />
                  Room Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Room ID (Optional)</label>
                  <input
                    type="text"
                    value={roomId}
                    onChange={(e) => setRoomId(e.target.value)}
                    placeholder="Enter room ID to join others"
                    className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    disabled={translationState.isActive}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Target Language</label>
                  <select
                    value={translationState.targetLanguage}
                    onChange={(e) => changeTargetLanguage(e.target.value)}
                    className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    disabled={translationState.isActive}
                  >
                    <option value="fr">French (Français)</option>
                    <option value="hi">Hindi (हिन्दी)</option>
                    <option value="es">Spanish (Español)</option>
                    <option value="de">German (Deutsch)</option>
                    <option value="it">Italian (Italiano)</option>
                  </select>
                </div>
              </CardContent>
            </Card>

            {/* Translation Status */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Globe className="w-5 h-5 mr-2" />
                  Translation Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                {translationState.error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
                    <p className="text-red-800 text-sm">{translationState.error}</p>
                  </div>
                )}

                {translationState.isTranslating && (
                  <div className="text-center py-4">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
                    <p className="text-sm text-gray-600">Translating...</p>
                  </div>
                )}

                {translationState.currentTranslation && (
                  <div className="space-y-3">
                    <div>
                      <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Original</p>
                      <p className="text-sm bg-gray-50 p-2 rounded">
                        {translationState.currentTranslation.original_text}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                        Translated ({translationState.targetLanguage.toUpperCase()})
                      </p>
                      <p className="text-sm bg-blue-50 p-2 rounded">
                        {translationState.currentTranslation.translated_text}
                      </p>
                    </div>
                    <div className="text-xs text-gray-500">
                      Confidence: {Math.round(translationState.currentTranslation.confidence * 100)}% |
                      Time: {translationState.currentTranslation.processing_time.toFixed(2)}s
                    </div>
                  </div>
                )}

                {!translationState.isActive && !translationState.currentTranslation && (
                  <div className="text-center py-8 text-gray-500">
                    <Globe className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Start translation to see results here</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};
