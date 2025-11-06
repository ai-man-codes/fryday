import { useState, useEffect, useRef, useCallback } from 'react';
import type { AudioProcessingState, AudioChunk } from '../types/translation';

interface UseAudioProcessingOptions {
  sampleRate?: number;
  bufferSize?: number;
  onAudioChunk?: (chunk: AudioChunk) => void;
  onAudioLevelChange?: (level: number) => void;
}

export const useAudioProcessing = ({
  // sampleRate = 16000,
  bufferSize = 1024,
  onAudioChunk,
  onAudioLevelChange,
}: UseAudioProcessingOptions = {}) => {
  const [state, setState] = useState<AudioProcessingState>({
    isRecording: false,
    audioLevel: 0,
    bufferSize,
    playbackQueue: [],
  });

  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null);
  const analyserNodeRef = useRef<AnalyserNode | null>(null);
  const playbackQueueRef = useRef<AudioBuffer[]>([]);
  const animationFrameRef = useRef<number | null>(null);

  const audioWorkletSupported = useRef<boolean>(false);

  // Check for AudioWorklet support
  useEffect(() => {
    audioWorkletSupported.current = 'audioWorklet' in AudioContext.prototype ||
                                   'webkitAudioWorklet' in AudioContext.prototype;
  }, []);

  const initializeAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }

    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }

    return audioContextRef.current;
  }, []);

  const startRecording = useCallback(async (stream: MediaStream) => {
    try {
      const audioContext = await initializeAudioContext();
      mediaStreamRef.current = stream;

      // Create analyser for audio level monitoring
      analyserNodeRef.current = audioContext.createAnalyser();
      analyserNodeRef.current.fftSize = 256;
      analyserNodeRef.current.smoothingTimeConstant = 0.8;

      // Create source node
      sourceNodeRef.current = audioContext.createMediaStreamSource(stream);
      sourceNodeRef.current.connect(analyserNodeRef.current);

      // Create processor node for audio data
      processorNodeRef.current = audioContext.createScriptProcessor(bufferSize, 1, 1);

      processorNodeRef.current.onaudioprocess = (event) => {
        const inputBuffer = event.inputBuffer;
        const inputData = inputBuffer.getChannelData(0);

        // Calculate audio level
        const analyserData = new Uint8Array(analyserNodeRef.current!.frequencyBinCount);
        analyserNodeRef.current!.getByteFrequencyData(analyserData);
        const average = analyserData.reduce((a, b) => a + b) / analyserData.length;
        const audioLevel = average / 255;

        setState(prev => ({ ...prev, audioLevel }));
        onAudioLevelChange?.(audioLevel);

        // Send audio chunk
        if (state.isRecording) {
          const chunk: AudioChunk = {
            data: new Float32Array(inputData),
            timestamp: Date.now(),
          };
          onAudioChunk?.(chunk);
        }
      };

      sourceNodeRef.current.connect(processorNodeRef.current);
      processorNodeRef.current.connect(audioContext.destination);

      setState(prev => ({ ...prev, isRecording: true }));

      // Start audio level monitoring
      const monitorAudioLevel = () => {
        if (analyserNodeRef.current && state.isRecording) {
          animationFrameRef.current = requestAnimationFrame(monitorAudioLevel);
        }
      };
      monitorAudioLevel();

    } catch (error) {
      console.error('Error starting recording:', error);
      throw error;
    }
  }, [bufferSize, state.isRecording, onAudioChunk, onAudioLevelChange]);

  const stopRecording = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (processorNodeRef.current) {
      processorNodeRef.current.disconnect();
      processorNodeRef.current = null;
    }

    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }

    if (analyserNodeRef.current) {
      analyserNodeRef.current.disconnect();
      analyserNodeRef.current = null;
    }

    mediaStreamRef.current = null;
    setState(prev => ({ ...prev, isRecording: false, audioLevel: 0 }));
  }, []);

  const playAudioBuffer = useCallback(async (audioBuffer: AudioBuffer) => {
    try {
      const audioContext = await initializeAudioContext();

      // Create a gain node for smooth playback
      const gainNode = audioContext.createGain();
      gainNode.connect(audioContext.destination);

      // Apply fade in/out to avoid clicks
      const fadeTime = 0.01; // 10ms
      const now = audioContext.currentTime;

      gainNode.gain.setValueAtTime(0, now);
      gainNode.gain.linearRampToValueAtTime(1, now + fadeTime);
      gainNode.gain.setValueAtTime(1, now + audioBuffer.duration - fadeTime);
      gainNode.gain.linearRampToValueAtTime(0, now + audioBuffer.duration);

      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(gainNode);
      source.start();

      return new Promise<void>((resolve) => {
        source.onended = () => resolve();
      });
    } catch (error) {
      console.error('Error playing audio buffer:', error);
      throw error;
    }
  }, []);

  const addToPlaybackQueue = useCallback((audioBuffer: AudioBuffer) => {
    playbackQueueRef.current.push(audioBuffer);
    setState(prev => ({
      ...prev,
      playbackQueue: [...prev.playbackQueue, audioBuffer]
    }));
  }, []);

  const processPlaybackQueue = useCallback(async () => {
    while (playbackQueueRef.current.length > 0) {
      const audioBuffer = playbackQueueRef.current.shift();
      if (audioBuffer) {
        await playAudioBuffer(audioBuffer);
        setState(prev => ({
          ...prev,
          playbackQueue: prev.playbackQueue.slice(1)
        }));
      }
    }
  }, [playAudioBuffer]);

  const decodeAudioData = useCallback(async (arrayBuffer: ArrayBuffer): Promise<AudioBuffer> => {
    const audioContext = await initializeAudioContext();
    return audioContext.decodeAudioData(arrayBuffer);
  }, []);

  const createAudioBuffer = useCallback(async (float32Array: Float32Array, sampleRate: number): Promise<AudioBuffer> => {
    const audioContext = await initializeAudioContext();
    const audioBuffer = audioContext.createBuffer(1, float32Array.length, sampleRate);
    audioBuffer.getChannelData(0).set(float32Array);
    return audioBuffer;
  }, []);

  const cleanup = useCallback(() => {
    stopRecording();

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    playbackQueueRef.current = [];
    setState(prev => ({
      ...prev,
      playbackQueue: [],
      audioLevel: 0,
    }));
  }, [stopRecording]);

  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  return {
    state,
    startRecording,
    stopRecording,
    playAudioBuffer,
    addToPlaybackQueue,
    processPlaybackQueue,
    decodeAudioData,
    createAudioBuffer,
    cleanup,
    audioContext: audioContextRef.current,
  };
};
