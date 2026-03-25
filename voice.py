"""
Jarvis Voice Client — Talk to Jarvis with your voice!
Connects to your existing WebSocket gateway.

Pipeline:
  [Mic] → Wake Word → Whisper STT → Gateway → TTS → [Speaker]

Install:
  pip install websockets SpeechRecognition edge-tts pygame openai-whisper

  If pyaudio fails on Windows:
    pip install pipwin
    pipwin install pyaudio
  
  OR use the prebuilt wheel:
    pip install PyAudio-0.2.14-cp312-cp312-win_amd64.whl
    (download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

Run:
  python voice.py                     # default: Google free STT
  python voice.py --stt whisper       # use local Whisper (more accurate)
  python voice.py --wake-word jarvis  # custom wake word
  python voice.py --no-wake           # skip wake word, always listen
  python voice.py --voice en-US-GuyNeural   # change TTS voice
"""

import asyncio
import json
import sys
import os
import io
import tempfile
import threading
import queue
import argparse
import time

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

GATEWAY_URL = "ws://127.0.0.1:18789"
PEER_ID = "voice-client"

# TTS voices (Microsoft Edge — free, high quality)
# Some good options:
#   en-US-GuyNeural      — calm male (Jarvis-like)
#   en-US-ChristopherNeural — deeper male
#   en-US-JennyNeural    — female
#   en-GB-RyanNeural     — British male (most Jarvis-like!)
#   en-IN-PrabhatNeural  — Indian English male
DEFAULT_VOICE = "en-GB-RyanNeural"


# ═══════════════════════════════════════════════════════════════
# SPEECH TO TEXT
# ═══════════════════════════════════════════════════════════════

class GoogleSTT:
    """Free Google Speech Recognition (no API key needed)."""

    def __init__(self):
        import speech_recognition as sr
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.mic = sr.Microphone()

        # calibrate for ambient noise
        print("  [Mic] Calibrating for ambient noise...")
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("  [Mic] Ready!")

    def listen(self) -> str | None:
        import speech_recognition as sr
        with self.mic as source:
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                print(f"  [!] Google STT error: {e}")
                return None


class WhisperSTT:
    """Local Whisper model (more accurate, runs offline)."""

    def __init__(self, model_size="base"):
        import speech_recognition as sr
        import whisper

        print(f"  [Whisper] Loading '{model_size}' model...")
        self.model = whisper.load_model(model_size)
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.mic = sr.Microphone()

        print("  [Mic] Calibrating for ambient noise...")
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("  [Whisper] Ready!")

    def listen(self) -> str | None:
        import speech_recognition as sr
        import numpy as np

        with self.mic as source:
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)

                # convert audio to numpy array for Whisper
                wav_data = audio.get_wav_data()
                audio_array = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32) / 32768.0

                result = self.model.transcribe(audio_array, fp16=False)
                text = result["text"].strip()
                return text if text else None

            except sr.WaitTimeoutError:
                return None
            except Exception as e:
                print(f"  [!] Whisper error: {e}")
                return None


# ═══════════════════════════════════════════════════════════════
# TEXT TO SPEECH (Microsoft Edge TTS — free + high quality)
# ═══════════════════════════════════════════════════════════════

class EdgeTTS:
    """High-quality TTS using Microsoft Edge voices (free)."""

    def __init__(self, voice=DEFAULT_VOICE):
        self.voice = voice
        self._init_pygame()

    def _init_pygame(self):
        import pygame
        pygame.mixer.init(frequency=24000, size=-16, channels=1)
        self.pygame = pygame

    async def speak(self, text: str):
        import edge_tts

        # generate audio
        communicate = edge_tts.Communicate(text, self.voice)

        # save to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name

        await communicate.save(temp_path)

        # play audio (in a thread to not block)
        await asyncio.to_thread(self._play_audio, temp_path)

        # cleanup
        try:
            os.unlink(temp_path)
        except:
            pass

    def _play_audio(self, path: str):
        try:
            self.pygame.mixer.music.load(path)
            self.pygame.mixer.music.play()
            while self.pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"  [!] Audio playback error: {e}")

    def stop(self):
        try:
            self.pygame.mixer.music.stop()
        except:
            pass


# ═══════════════════════════════════════════════════════════════
# WAKE WORD DETECTOR
# ═══════════════════════════════════════════════════════════════

class WakeWordDetector:
    """Simple wake word detection using speech recognition."""

    def __init__(self, wake_word: str, stt):
        self.wake_word = wake_word.lower()
        self.stt = stt

    def wait_for_wake_word(self) -> str | None:
        """Listen and return text after wake word, or None."""
        text = self.stt.listen()
        if text is None:
            return None

        text_lower = text.lower()

        # check if wake word is in the text
        if self.wake_word in text_lower:
            # extract everything after the wake word
            idx = text_lower.index(self.wake_word) + len(self.wake_word)
            remaining = text[idx:].strip()

            # strip common filler words after wake word
            for filler in [",", ".", "!", "?", "hey", "ok", "please"]:
                remaining = remaining.lstrip(filler).strip()

            return remaining if remaining else ""

        return None


# ═══════════════════════════════════════════════════════════════
# VOICE CLIENT (ties everything together)
# ═══════════════════════════════════════════════════════════════

class JarvisVoiceClient:
    def __init__(self, args):
        self.args = args
        self.tts = EdgeTTS(voice=args.voice)

        # init STT
        if args.stt == "whisper":
            self.stt = WhisperSTT(model_size=args.whisper_model)
        else:
            self.stt = GoogleSTT()

        # wake word
        self.use_wake = not args.no_wake
        if self.use_wake:
            self.wake = WakeWordDetector(args.wake_word, self.stt)

    async def start(self):
        import websockets

        print()
        print("=" * 52)
        print("  JARVIS VOICE CLIENT")
        print(f"  Gateway:    {GATEWAY_URL}")
        print(f"  STT:        {self.args.stt}")
        print(f"  TTS Voice:  {self.args.voice}")
        print(f"  Wake word:  {'disabled' if not self.use_wake else self.args.wake_word}")
        print("  Status:     LISTENING")
        print("=" * 52)
        print()

        while True:
            try:
                async with websockets.connect(GATEWAY_URL) as ws:
                    # register
                    await ws.send(json.dumps({
                        "type": "status",
                        "peer_id": PEER_ID,
                    }))

                    # read welcome
                    welcome = json.loads(await ws.recv())
                    if welcome.get("type") == "system":
                        print(f"  Jarvis: {welcome['text']}")
                        await self.tts.speak(welcome["text"])

                    # main loop
                    await self._listen_loop(ws)

            except ConnectionRefusedError:
                print("  [!] Gateway not running. Retrying in 3s...")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"  [!] Connection error: {e}. Retrying in 3s...")
                await asyncio.sleep(3)

    async def _listen_loop(self, ws):
        while True:
            text = None

            if self.use_wake:
                # wait for wake word
                print(f"\n  Listening for '{self.args.wake_word}'...")
                result = await asyncio.to_thread(
                    self.wake.wait_for_wake_word
                )

                if result is None:
                    continue

                if result == "":
                    # wake word detected but no command — listen again
                    print("  [*] Wake word detected! Listening for command...")

                    # play a small chime-like feedback
                    await self.tts.speak("Yes, sir?")

                    text = await asyncio.to_thread(self.stt.listen)
                    if text is None:
                        print("  [*] Didn't catch that.")
                        continue
                else:
                    # wake word + command in one phrase
                    text = result
            else:
                # no wake word — always listening
                print("\n  Listening...")
                text = await asyncio.to_thread(self.stt.listen)
                if text is None:
                    continue

            # we have text — send to gateway
            print(f"  You: {text}")

            await ws.send(json.dumps({
                "type": "message",
                "text": text,
                "peer_id": PEER_ID,
            }))

            # wait for response (skip typing indicator)
            while True:
                raw = await ws.recv()
                data = json.loads(raw)

                if data["type"] == "typing":
                    print("  Jarvis is thinking...")
                    continue

                if data["type"] == "response":
                    reply = data["text"]
                    print(f"  Jarvis: {reply}")
                    await self.tts.speak(reply)
                    break

                if data["type"] == "error":
                    print(f"  [!] Error: {data['text']}")
                    await self.tts.speak("Sorry sir, I encountered an error.")
                    break

                if data["type"] in ("system", "status"):
                    print(f"  [System] {data.get('text', '')}")
                    break


# ═══════════════════════════════════════════════════════════════
# ENTRY
# ═══════════════════════════════════════════════════════════════

def list_voices():
    """Print available Edge TTS voices."""
    print("\n  Fetching available voices...\n")

    async def _list():
        import edge_tts
        voices = await edge_tts.list_voices()
        en_voices = [v for v in voices if v["Locale"].startswith("en")]
        for v in en_voices:
            gender = v["Gender"]
            name = v["ShortName"]
            locale = v["Locale"]
            print(f"    {name:35s}  {gender:8s}  {locale}")

    asyncio.run(_list())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jarvis Voice Client")
    parser.add_argument(
        "--stt", choices=["google", "whisper"], default="google",
        help="Speech-to-text engine (default: google free)",
    )
    parser.add_argument(
        "--whisper-model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--voice", default=DEFAULT_VOICE,
        help=f"TTS voice name (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--wake-word", default="jarvis",
        help="Wake word to activate (default: jarvis)",
    )
    parser.add_argument(
        "--no-wake", action="store_true",
        help="Disable wake word — always listen",
    )
    parser.add_argument(
        "--list-voices", action="store_true",
        help="List available TTS voices and exit",
    )

    args = parser.parse_args()

    if args.list_voices:
        list_voices()
        sys.exit(0)

    client = JarvisVoiceClient(args)

    try:
        asyncio.run(client.start())
    except KeyboardInterrupt:
        print("\n  Voice client stopped. Goodbye, sir.")