import asyncio
import json
import logging
import threading
import time
import base64
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Import translation components from edge.py
import torch
import torch.hub
from faster_whisper import WhisperModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, MarianMTModel, MarianTokenizer
import edge_tts
import sounddevice as sd
import shutil
import subprocess
import queue
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# --- CONFIGURATION (adapted from edge.py)
# --------------------------

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_MODEL_NAME = "base"

TRANSLATOR_CONFIGS = {
    "marian_en_fr": {
        "type": "marian",
        "model_name": "Helsinki-NLP/opus-mt-en-fr",
        "target_lang_code": "fr",
        "tts_voice": "fr-FR-HenriNeural",
    },
    "m2m100_fr": {
        "type": "m2m100",
        "model_name": "facebook/m2m100_418M",
        "target_lang_code": "fr",
        "tts_voice": "fr-FR-HenriNeural",
    },
    "marian_en_hi": {
        "type": "marian",
        "model_name": "Helsinki-NLP/opus-mt-en-hi",
        "target_lang_code": "hi",
        "tts_voice": "hi-IN-MadhurNeural",
    },
    "m2m100_hi": {
        "type": "m2m100",
        "model_name": "facebook/m2m100_418M",
        "target_lang_code": "hi",
        "tts_voice": "hi-IN-MadhurNeural",
    }
}

# --------------------------
# --- DATA STRUCTURES
# --------------------------

@dataclass
class Room:
    room_id: str
    participants: List[str] = None  # List of connection IDs (max 2)
    is_active: bool = False
    language_pairs: Dict[str, str] = None  # connection_id -> target_lang

    def __post_init__(self):
        if self.participants is None:
            self.participants = []
        if self.language_pairs is None:
            self.language_pairs = {}

@dataclass
class Connection:
    websocket: WebSocket
    connection_id: str
    room_id: Optional[str] = None
    target_language: str = "fr"

@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    target_language: str
    confidence: float
    processing_time: float
    audio_data: Optional[str] = None  # base64 encoded

# --------------------------
# --- GLOBAL STATE
# --------------------------

app = FastAPI(title="Speech-to-Speech Translation Server")
rooms: Dict[str, Room] = {}
connections: Dict[str, Connection] = {}
connection_counter = 0

# Audio processing queues per connection
audio_queues: Dict[str, asyncio.Queue] = {}
translation_results: Dict[str, asyncio.Queue] = {}

# --------------------------
# --- CORS MIDDLEWARE
# --------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# --- MODEL LOADING (adapted from edge.py)
# --------------------------

logger.info(f"🚀 Using device: {DEVICE}")

# Global model instances (loaded once)
asr_model = None
translator_tokenizer = None
translator_model = None
vad_model = None
vad_utils = None

def load_models():
    global asr_model, translator_tokenizer, translator_model, vad_model, vad_utils

    logger.info("Loading ASR model...")
    asr_model = WhisperModel(ASR_MODEL_NAME, device=DEVICE, compute_type="int8")

    logger.info("Loading translation models (defaulting to marian_en_fr)...")
    config = TRANSLATOR_CONFIGS["marian_en_fr"]
    if config["type"] == "m2m100":
        translator_tokenizer = M2M100Tokenizer.from_pretrained(config["model_name"])
        translator_model = M2M100ForConditionalGeneration.from_pretrained(config["model_name"]).to(DEVICE)
    else:
        translator_tokenizer = MarianTokenizer.from_pretrained(config["model_name"])
        translator_model = MarianMTModel.from_pretrained(config["model_name"]).to(DEVICE)

    logger.info("Loading VAD model...")
    vad_model, vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                                         model="silero_vad",
                                         force_reload=False,
                                         onnx=False)
    vad_model = vad_model.to("cpu")

    logger.info("✅ All models loaded and ready.")

# --------------------------
# --- AUDIO DECODING (from edge.py)
# --------------------------

def ffmpeg_decode_to_pcm_s16le(input_bytes: bytes, rate: int = SAMPLE_RATE) -> bytes:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise FileNotFoundError("ffmpeg binary not found on PATH. Please install ffmpeg.")

    cmd = [
        ffmpeg_path,
        "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ar", str(rate),
        "-ac", "1",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed. Error: {proc.stderr.decode('utf-8', errors='ignore')}")
    return proc.stdout

# --------------------------
# --- TTS GENERATION (adapted from edge.py)
# --------------------------

async def generate_tts_audio(text: str, voice: str) -> bytes:
    """Generate TTS audio for given text and voice."""
    try:
        comm = edge_tts.Communicate(text, voice)
        bytes_acc = bytearray()
        async for chunk in comm.stream():
            if isinstance(chunk, dict) and chunk.get("type") == "audio" and chunk.get("data"):
                bytes_acc.extend(chunk["data"])

        if bytes_acc:
            return bytes(bytes_acc)
        else:
            logger.warning(f"TTS returned no audio bytes for text: {text[:50]}...")
            return b""
    except Exception as e:
        logger.error(f"TTS generation failed for text: {text[:50]}... Error: {e}")
        raise e

# --------------------------
# --- TRANSLATION PIPELINE
# --------------------------

async def process_translation_pipeline(connection_id: str, target_lang: str = "fr"):
    """Process audio chunks for a specific connection and generate translations."""
    global asr_model, translator_tokenizer, translator_model, vad_model, vad_utils

    if connection_id not in audio_queues:
        return

    audio_queue = audio_queues[connection_id]
    result_queue = translation_results[connection_id]

    # Get translator config for target language
    translator_key = f"marian_en_{target_lang}" if target_lang in ["fr", "hi"] else "marian_en_fr"
    if translator_key not in TRANSLATOR_CONFIGS:
        translator_key = "marian_en_fr"

    config = TRANSLATOR_CONFIGS[translator_key]
    translator_type = config["type"]
    target_lang_code = config["target_lang_code"]
    tts_voice = config["tts_voice"]

    # VAD parameters
    VAD_THRESHOLD = 0.4
    VAD_MIN_SILENCE_MS = 200
    VAD_MAX_PHRASE_S = 5
    VAD_SAMPLES_PER_CHUNK = 512

    phrase_buffer = np.zeros(0, dtype=np.float32)
    buffer = np.zeros(0, dtype=np.float32)
    last_text = ""
    last_speech_time = time.time()

    logger.info(f"🎤 Started translation pipeline for connection {connection_id}")

    try:
        while True:
            try:
                # Get audio chunk with timeout
                audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                if audio_data is None:  # Shutdown signal
                    break

                # Decode base64 audio
                try:
                    binary_data = base64.b64decode(audio_data["audio_data"])
                    audio_array = np.frombuffer(binary_data, dtype=np.float32)
                except Exception as e:
                    logger.error(f"Failed to decode audio data: {e}")
                    continue

                buffer = np.concatenate((buffer, audio_array))

                if len(buffer) < VAD_SAMPLES_PER_CHUNK:
                    continue

                vad_chunk = buffer[:VAD_SAMPLES_PER_CHUNK]
                buffer = buffer[VAD_SAMPLES_PER_CHUNK:]

                # Voice Activity Detection
                speech_prob = vad_model(torch.from_numpy(vad_chunk), SAMPLE_RATE).item()
                is_speech = speech_prob > VAD_THRESHOLD

                if is_speech:
                    phrase_buffer = np.concatenate((phrase_buffer, vad_chunk))
                    last_speech_time = time.time()

                trigger_process = False
                if not is_speech and len(phrase_buffer) > 0:
                    if time.time() - last_speech_time > (VAD_MIN_SILENCE_MS / 1000.0):
                        trigger_process = True

                if len(phrase_buffer) > VAD_MAX_PHRASE_S * SAMPLE_RATE:
                    trigger_process = True
                    logger.info("⚠️ Phrase buffer full. Forcing translation.")

                if trigger_process and len(phrase_buffer) > 0:
                    audio_to_process = phrase_buffer.copy()
                    phrase_buffer = np.zeros(0, dtype=np.float32)

                    if np.mean(np.abs(audio_to_process)) < 1e-4:
                        continue

                    t_start = time.time()

                    # ASR
                    segments, info = asr_model.transcribe(audio_to_process, beam_size=1)
                    text = " ".join([s.text for s in segments]).strip()
                    t_asr = time.time() - t_start

                    if not text or text.lower() == last_text.lower():
                        continue

                    last_text = text
                    logger.info(f"🗣️  [ASR in {t_asr:.2f}s] ({info.language}): {text}")

                    # Translation
                    t_start = time.time()
                    translated = ""
                    try:
                        if translator_type == "m2m100":
                            translator_tokenizer.src_lang = info.language
                            inputs = translator_tokenizer(text, return_tensors="pt").to(DEVICE)
                            bos = translator_tokenizer.get_lang_id(target_lang_code)
                            outputs = translator_model.generate(**inputs, forced_bos_token_id=bos)
                            translated = translator_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                        elif translator_type == "marian":
                            inputs = translator_tokenizer(text, return_tensors="pt").to(DEVICE)
                            outputs = translator_model.generate(**inputs)
                            translated = translator_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                        t_trans = time.time() - t_start
                        logger.info(f"💬  [Translate in {t_trans:.2f}s] ({target_lang_code}): {translated}")

                        # TTS
                        t_start = time.time()
                        tts_bytes = await generate_tts_audio(translated, tts_voice)
                        if tts_bytes:
                            # Convert to base64 for transmission
                            audio_b64 = base64.b64encode(tts_bytes).decode('utf-8')
                            t_tts = time.time() - t_start
                            logger.info(f"🎧 [TTS in {t_tts:.2f}s] Generated {len(tts_bytes)} bytes of audio")
                        else:
                            audio_b64 = None

                        # Create result
                        result = TranslationResult(
                            original_text=text,
                            translated_text=translated,
                            target_language=target_lang_code,
                            confidence=0.8,  # Placeholder - could be improved
                            processing_time=time.time() - t_start,
                            audio_data=audio_b64
                        )

                        # Send to result queue
                        await result_queue.put(result)

                    except Exception as e:
                        logger.error(f"Translation failed: {e}")
                        error_result = TranslationResult(
                            original_text=text,
                            translated_text="Translation failed",
                            target_language=target_lang,
                            confidence=0.0,
                            processing_time=time.time() - t_start
                        )
                        await result_queue.put(error_result)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in translation pipeline: {e}")
                traceback.print_exc()
                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"Translation pipeline crashed for connection {connection_id}: {e}")

    logger.info(f"🛑 Translation pipeline stopped for connection {connection_id}")

# --------------------------
# --- WEBSOCKET HANDLERS
# --------------------------

async def broadcast_to_room(room_id: str, message: dict, exclude_connection_id: str = None):
    """Broadcast message to all participants in a room."""
    if room_id not in rooms:
        return

    room = rooms[room_id]
    for participant_id in room.participants:
        if participant_id != exclude_connection_id and participant_id in connections:
            try:
                await connections[participant_id].websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to {participant_id}: {e}")

async def handle_webrtc_signaling(connection_id: str, message: dict):
    """Handle WebRTC signaling messages."""
    connection = connections.get(connection_id)
    if not connection or not connection.room_id:
        return

    room = rooms.get(connection.room_id)
    if not room:
        return

    # Forward signaling messages to other participant
    for participant_id in room.participants:
        if participant_id != connection_id and participant_id in connections:
            try:
                signaling_message = {
                    "type": "webrtc_signal",
                    "payload": {
                        "from": connection_id,
                        **message
                    },
                    "timestamp": time.time() * 1000
                }
                await connections[participant_id].websocket.send_json(signaling_message)
            except Exception as e:
                logger.error(f"Failed to forward WebRTC signal: {e}")

# --------------------------
# --- WEBSOCKET ENDPOINT
# --------------------------

@app.websocket("/ws/translate")
async def websocket_endpoint(websocket: WebSocket):
    global connection_counter

    await websocket.accept()

    # Generate unique connection ID
    connection_counter += 1
    connection_id = f"conn_{connection_counter}"

    # Create connection object
    connection = Connection(
        websocket=websocket,
        connection_id=connection_id
    )
    connections[connection_id] = connection

    # Create audio and result queues for this connection
    audio_queues[connection_id] = asyncio.Queue()
    translation_results[connection_id] = asyncio.Queue()

    logger.info(f"New connection: {connection_id}")

    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "payload": {"connection_id": connection_id},
            "timestamp": time.time() * 1000
        })

        # Start translation pipeline for this connection
        translation_task = asyncio.create_task(
            process_translation_pipeline(connection_id, connection.target_language)
        )

        # Result broadcaster task
        async def broadcast_results():
            while True:
                try:
                    result = await translation_results[connection_id].get()
                    if result is None:  # Shutdown signal
                        break

                    message = {
                        "type": "translation_result",
                        "payload": asdict(result),
                        "timestamp": time.time() * 1000
                    }
                    await websocket.send_json(message)

                except Exception as e:
                    logger.error(f"Error broadcasting results: {e}")
                    break

        result_task = asyncio.create_task(broadcast_results())

        # Message handling loop
        while True:
            try:
                data = await websocket.receive_json()
                message_type = data.get("type")
                payload = data.get("payload", {})

                if message_type == "join_room":
                    room_id = payload.get("room_id")
                    language_pair = payload.get("language_pair", {})

                    if not room_id:
                        await websocket.send_json({
                            "type": "error",
                            "payload": {"message": "Room ID is required"},
                            "timestamp": time.time() * 1000
                        })
                        continue

                    # Get or create room
                    if room_id not in rooms:
                        rooms[room_id] = Room(room_id=room_id)

                    room = rooms[room_id]

                    # Check if room is full (max 2 participants)
                    if len(room.participants) >= 2:
                        await websocket.send_json({
                            "type": "error",
                            "payload": {"message": "Room is full (maximum 2 participants)"},
                            "timestamp": time.time() * 1000
                        })
                        continue

                    # Add participant to room
                    room.participants.append(connection_id)
                    connection.room_id = room_id
                    connection.target_language = language_pair.get("to", "fr")

                    # Update language pairs
                    room.language_pairs[connection_id] = connection.target_language

                    # If this is the second participant, activate the room
                    if len(room.participants) == 2:
                        room.is_active = True

                    logger.info(f"Connection {connection_id} joined room {room_id}")

                    # Notify all participants of room state
                    room_state_message = {
                        "type": "room_state",
                        "payload": {
                            "room_id": room_id,
                            "participants": room.participants,
                            "is_active": room.is_active,
                            "language_pairs": room.language_pairs
                        },
                        "timestamp": time.time() * 1000
                    }

                    await broadcast_to_room(room_id, room_state_message)

                elif message_type == "leave_room":
                    room_id = connection.room_id
                    if room_id and room_id in rooms:
                        room = rooms[room_id]
                        if connection_id in room.participants:
                            room.participants.remove(connection_id)
                            room.language_pairs.pop(connection_id, None)

                            if len(room.participants) < 2:
                                room.is_active = False

                            # Notify remaining participants
                            room_state_message = {
                                "type": "room_state",
                                "payload": {
                                    "room_id": room_id,
                                    "participants": room.participants,
                                    "is_active": room.is_active,
                                    "language_pairs": room.language_pairs
                                },
                                "timestamp": time.time() * 1000
                            }
                            await broadcast_to_room(room_id, room_state_message)

                            # Clean up empty rooms
                            if len(room.participants) == 0:
                                del rooms[room_id]

                    connection.room_id = None
                    logger.info(f"Connection {connection_id} left room")

                elif message_type == "audio_chunk":
                    # Queue audio for processing
                    await audio_queues[connection_id].put(payload)

                elif message_type == "webrtc_offer":
                    await handle_webrtc_signaling(connection_id, {"offer": payload})

                elif message_type == "webrtc_answer":
                    await handle_webrtc_signaling(connection_id, {"answer": payload})

                elif message_type == "webrtc_ice_candidate":
                    await handle_webrtc_signaling(connection_id, {"ice_candidate": payload})

                elif message_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": time.time() * 1000
                    })

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from {connection_id}")
                continue

    except WebSocketDisconnect:
        logger.info(f"Connection {connection_id} disconnected")

    except Exception as e:
        logger.error(f"Error handling connection {connection_id}: {e}")

    finally:
        # Cleanup
        try:
            # Leave room if in one
            if connection.room_id and connection.room_id in rooms:
                room = rooms[connection.room_id]
                if connection_id in room.participants:
                    room.participants.remove(connection_id)
                    room.language_pairs.pop(connection_id, None)

                    if len(room.participants) < 2:
                        room.is_active = False

                    # Notify remaining participants
                    if room.participants:
                        room_state_message = {
                            "type": "room_state",
                            "payload": {
                                "room_id": connection.room_id,
                                "participants": room.participants,
                                "is_active": room.is_active,
                                "language_pairs": room.language_pairs
                            },
                            "timestamp": time.time() * 1000
                        }
                        await broadcast_to_room(connection.room_id, room_state_message)

                    # Clean up empty rooms
                    if len(room.participants) == 0:
                        del rooms[connection.room_id]

            # Stop translation pipeline
            if connection_id in audio_queues:
                await audio_queues[connection_id].put(None)  # Shutdown signal
                del audio_queues[connection_id]

            if connection_id in translation_results:
                await translation_results[connection_id].put(None)  # Shutdown signal
                del translation_results[connection_id]

            # Remove connection
            if connection_id in connections:
                del connections[connection_id]

        except Exception as e:
            logger.error(f"Error during cleanup for {connection_id}: {e}")

# --------------------------
# --- HTTP ENDPOINTS
# --------------------------

@app.get("/")
async def root():
    return {"message": "Speech-to-Speech Translation Server", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "connections": len(connections),
        "rooms": len(rooms),
        "device": DEVICE
    }

@app.get("/rooms")
async def get_rooms():
    return {
        "rooms": [
            {
                "room_id": room.room_id,
                "participants": len(room.participants),
                "is_active": room.is_active,
                "max_participants": 2
            }
            for room in rooms.values()
        ]
    }

# --------------------------
# --- MAIN
# --------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Speech-to-Speech Translation Server...")
    load_models()
    logger.info("Server startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down server...")

    # Shutdown all translation pipelines
    for conn_id in list(audio_queues.keys()):
        try:
            audio_queues[conn_id].put_nowait(None)
        except:
            pass

    for conn_id in list(translation_results.keys()):
        try:
            translation_results[conn_id].put_nowait(None)
        except:
            pass

    logger.info("Shutdown complete!")

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
