// WebSocket Types
export type MessageType =
  | 'ping'
  | 'pong'
  | 'audio_chunk'
  | 'translation_result'
  | 'error'
  | 'connected'
  | 'join_room'
  | 'leave_room'
  | 'room_state'
  | 'webrtc_signal';

export interface WebSocketMessage {
  type: MessageType;
  payload: any;
  timestamp: number;
}

export interface WebSocketState {
  isConnected: boolean;
  connectionId: string | null;
  lastPing: number;
  reconnectAttempts: number;
}

// WebRTC Types
export interface WebRTCState {
  isConnected: boolean;
  localStream: MediaStream | null;
  remoteStream: MediaStream | null;
  peerConnection: RTCPeerConnection | null;
  isVideoEnabled: boolean;
  isAudioEnabled: boolean;
}

export interface WebRTCSignalMessage {
  offer?: RTCSessionDescriptionInit;
  answer?: RTCSessionDescriptionInit;
  ice_candidate?: RTCIceCandidateInit;
}

// Audio Processing Types
export interface AudioProcessingState {
  isRecording: boolean;
  audioLevel: number;
  bufferSize: number;
  playbackQueue: AudioBuffer[];
}

export interface AudioChunk {
  data: Float32Array;
  timestamp: number;
}

// Translation Types
export interface TranslationState {
  isActive: boolean;
  isTranslating: boolean;
  currentTranslation: TranslationResult | null;
  error: string | null;
  targetLanguage: string;
}

export interface TranslationResult {
  original_text: string;
  translated_text: string;
  confidence: number;
  processing_time: number;
  audio_data?: string; // base64 encoded audio
}

// Room Types
export interface RoomState {
  room_id: string;
  participants: Participant[];
  language_pairs: { [participantId: string]: { from: string; to: string } };
}

export interface Participant {
  id: string;
  name?: string;
  is_active: boolean;
}

// Message Payload Types
export interface ConnectedMessage {
  connection_id: string;
  server_version?: string;
}

export interface RoomJoinMessage {
  room_id: string;
  language_pair: { from: string; to: string };
}

export interface AudioChunkMessage {
  audio_data: string; // base64 encoded
  sample_rate: number;
  timestamp: number;
}