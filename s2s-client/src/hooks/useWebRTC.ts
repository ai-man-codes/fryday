import { useState, useEffect, useRef, useCallback } from 'react';
import type { WebRTCState } from '../types/translation';

const ICE_SERVERS = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
  ],
};

export const useWebRTC = () => {
  const [state, setState] = useState<WebRTCState>({
    isConnected: false,
    localStream: null,
    remoteStream: null,
    peerConnection: null,
    isVideoEnabled: true,
    isAudioEnabled: true,
  });

  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const remoteStreamRef = useRef<MediaStream | null>(null);

  const createPeerConnection = useCallback(async () => {
    const pc = new RTCPeerConnection(ICE_SERVERS);

    pc.onicecandidate = (event) => {
      if (event.candidate) {
        // Send ICE candidate to remote peer via signaling server
        console.log('ICE candidate:', event.candidate);
      }
    };

    pc.ontrack = (event) => {
      console.log('Received remote track:', event.track.kind);
      if (event.streams[0]) {
        const remoteStream = event.streams[0];
        remoteStreamRef.current = remoteStream;
        setState(prev => ({ ...prev, remoteStream }));
      }
    };

    pc.onconnectionstatechange = () => {
      console.log('Connection state:', pc.connectionState);
      setState(prev => ({
        ...prev,
        isConnected: pc.connectionState === 'connected'
      }));
    };

    peerConnectionRef.current = pc;
    setState(prev => ({ ...prev, peerConnection: pc }));

    return pc;
  }, []);

  const startLocalStream = useCallback(async (video = true, audio = true) => {
    try {
      const constraints: MediaStreamConstraints = {
        video: video ? {
          width: { ideal: 640, max: 1280 },
          height: { ideal: 480, max: 720 },
          frameRate: { ideal: 15, max: 30 },
          aspectRatio: 4/3
        } : false,
        audio: audio ? {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
          channelCount: 1,
        } : false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      localStreamRef.current = stream;

      setState(prev => ({
        ...prev,
        localStream: stream,
        isVideoEnabled: video,
        isAudioEnabled: audio,
      }));

      return stream;
    } catch (error) {
      console.error('Error accessing media devices:', error);
      throw error;
    }
  }, []);

  const addLocalStreamToPeerConnection = useCallback(async () => {
    if (!peerConnectionRef.current || !localStreamRef.current) return;

    localStreamRef.current.getTracks().forEach(track => {
      peerConnectionRef.current!.addTrack(track, localStreamRef.current!);
    });
  }, []);

  const createOffer = useCallback(async () => {
    if (!peerConnectionRef.current) return null;

    try {
      const offer = await peerConnectionRef.current.createOffer();
      await peerConnectionRef.current.setLocalDescription(offer);
      return offer;
    } catch (error) {
      console.error('Error creating offer:', error);
      throw error;
    }
  }, []);

  const createAnswer = useCallback(async (offer: RTCSessionDescriptionInit) => {
    if (!peerConnectionRef.current) return null;

    try {
      await peerConnectionRef.current.setRemoteDescription(offer);
      const answer = await peerConnectionRef.current.createAnswer();
      await peerConnectionRef.current.setLocalDescription(answer);
      return answer;
    } catch (error) {
      console.error('Error creating answer:', error);
      throw error;
    }
  }, []);

  const addAnswer = useCallback(async (answer: RTCSessionDescriptionInit) => {
    if (!peerConnectionRef.current) return;

    try {
      await peerConnectionRef.current.setRemoteDescription(answer);
    } catch (error) {
      console.error('Error adding answer:', error);
      throw error;
    }
  }, []);

  const addIceCandidate = useCallback(async (candidate: RTCIceCandidateInit) => {
    if (!peerConnectionRef.current) return;

    try {
      await peerConnectionRef.current.addIceCandidate(candidate);
    } catch (error) {
      console.error('Error adding ICE candidate:', error);
      throw error;
    }
  }, []);

  const toggleVideo = useCallback(() => {
    if (localStreamRef.current) {
      const videoTrack = localStreamRef.current.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        setState(prev => ({ ...prev, isVideoEnabled: videoTrack.enabled }));
      }
    }
  }, []);

  const toggleAudio = useCallback(() => {
    if (localStreamRef.current) {
      const audioTrack = localStreamRef.current.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setState(prev => ({ ...prev, isAudioEnabled: audioTrack.enabled }));
      }
    }
  }, []);

  const adjustVideoQuality = useCallback(async (quality: 'low' | 'medium' | 'high' = 'medium') => {
    if (!localStreamRef.current) return;

    const videoTrack = localStreamRef.current.getVideoTracks()[0];
    if (!videoTrack) return;

    const capabilities = videoTrack.getCapabilities?.();
    if (!capabilities) return;

    const constraints: MediaTrackConstraints = {};

    switch (quality) {
      case 'low':
        constraints.width = { ideal: 320, max: 640 };
        constraints.height = { ideal: 240, max: 480 };
        constraints.frameRate = { ideal: 10, max: 15 };
        break;
      case 'medium':
        constraints.width = { ideal: 640, max: 1280 };
        constraints.height = { ideal: 480, max: 720 };
        constraints.frameRate = { ideal: 15, max: 30 };
        break;
      case 'high':
        constraints.width = { ideal: 1280, max: 1920 };
        constraints.height = { ideal: 720, max: 1080 };
        constraints.frameRate = { ideal: 30, max: 30 };
        break;
    }

    try {
      await videoTrack.applyConstraints(constraints);
      console.log(`Video quality adjusted to ${quality}`);
    } catch (error) {
      console.warn('Failed to adjust video quality:', error);
    }
  }, []);

  const stopLocalStream = useCallback(() => {
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => track.stop());
      localStreamRef.current = null;
      setState(prev => ({ ...prev, localStream: null }));
    }
  }, []);

  const closePeerConnection = useCallback(() => {
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
      setState(prev => ({
        ...prev,
        peerConnection: null,
        isConnected: false,
        remoteStream: null,
      }));
      remoteStreamRef.current = null;
    }
  }, []);

  const cleanup = useCallback(() => {
    stopLocalStream();
    closePeerConnection();
  }, [stopLocalStream, closePeerConnection]);

  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  return {
    state,
    startLocalStream,
    addLocalStreamToPeerConnection,
    createPeerConnection,
    createOffer,
    createAnswer,
    addAnswer,
    addIceCandidate,
    toggleVideo,
    toggleAudio,
    stopLocalStream,
    closePeerConnection,
    cleanup,
    adjustVideoQuality,
  };
};
