"""
60db Python wrapper — sits inside elevenlabslib alongside the ElevenLabs classes.

Mirrors the User → Voice object graph that elevenlabslib uses for ElevenLabs:

    user = SixtydbUser("sk_live_...")
    voice = user.get_voice("fbb75ed2-975a-40c7-9e06-38e30524a9a1")
    voice.generate_play_audio_v2("Hello.", playbackOptions=PlaybackOptions(runInBackground=False))

Reuses elevenlabslib's PlaybackOptions + play_audio_v2 so the in-process
playback magic (device selection, start/end callbacks, runInBackground) works
identically for 60db audio.

Three TTS surfaces, all returning mp3 bytes for parity with elevenlabslib:
  - generate_audio_v3 (REST one-shot)       POST /tts-synthesize
  - stream_audio_v3   (NDJSON streaming)    POST /tts-stream
  - generate_websocket_audio (WS session)   wss://api.60db.ai/ws/tts

Docs: https://docs.60db.ai
"""

from __future__ import annotations

import base64
import concurrent.futures
import dataclasses
import io
import json
import threading
import uuid
from concurrent.futures import Future
from typing import Iterator, Optional, Tuple, Union

import requests

from .helpers import PlaybackOptions, play_audio_v2

SIXTYDB_REST_URL = "https://api.60db.ai/tts-synthesize"
SIXTYDB_STREAM_URL = "https://api.60db.ai/tts-stream"
SIXTYDB_WS_URL = "wss://api.60db.ai/ws/tts"
SIXTYDB_DEFAULT_VOICE_ID = "fbb75ed2-975a-40c7-9e06-38e30524a9a1"
SIXTYDB_CHAT_BASE_URL = "https://api.60db.ai/v1"
SIXTYDB_DEFAULT_CHAT_MODEL = "60db-tiny"


@dataclasses.dataclass
class SixtydbGenerationOptions:
    """Generation options for 60db TTS.

    Parameters:
        speed (float): 0.5 - 2.0. Defaults to 1.0.
        stability (float): 0 - 100. Lower = expressive, higher = consistent. Defaults to 50.
        similarity (float): 0 - 100. Defaults to 75.
        enhance (bool): Enable audio enhancement. Defaults to True.
        output_format (str): mp3 | wav | ogg | flac. Defaults to mp3 so the
            return bytes match elevenlabslib's mp3-default convention.
        language (str | None): Optional language hint. Auto-detect if None.
    """
    speed: float = 1.0
    stability: float = 50.0
    similarity: float = 75.0
    enhance: bool = True
    output_format: str = "mp3"
    language: Optional[str] = None

    def __post_init__(self):
        if not (0.5 <= self.speed <= 2.0):
            raise ValueError("speed must be between 0.5 and 2.0")
        if not (0 <= self.stability <= 100):
            raise ValueError("stability must be between 0 and 100")
        if not (0 <= self.similarity <= 100):
            raise ValueError("similarity must be between 0 and 100")
        if self.output_format not in ("mp3", "wav", "ogg", "flac"):
            raise ValueError("output_format must be one of: mp3, wav, ogg, flac")


class SixtydbUser:
    """60db API client. Parallel to elevenlabslib.User.

    Holds the api key and acts as a factory for SixtydbVoice instances.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("60db api_key is required")
        self._api_key = api_key

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def get_voice(self, voice_id: str = SIXTYDB_DEFAULT_VOICE_ID) -> "SixtydbVoice":
        """Return a SixtydbVoice bound to this user. 60db has no /voices
        listing in the public docs, so this is a direct constructor by ID."""
        return SixtydbVoice(self, voice_id)

    def get_chat(self, model: str = SIXTYDB_DEFAULT_CHAT_MODEL) -> "SixtydbChat":
        """Return a SixtydbChat bound to this user for LLM calls.

        60db's chat completions endpoint is drop-in OpenAI-compatible
        (POST /v1/chat/completions). This is a new feature in this library
        — elevenlabslib has no equivalent because ElevenLabs is pure TTS.
        """
        return SixtydbChat(self, model)


class SixtydbVoice:
    """60db TTS for a specific voice. Parallel to elevenlabslib.Voice.

    Methods mirror elevenlabslib's Voice surface where the semantics match:
        - generate_audio_v3        REST one-shot
        - stream_audio_v3          NDJSON streaming
        - generate_websocket_audio WebSocket session
        - generate_play_audio_v2   generate + play via PlaybackOptions
    """

    def __init__(self, linked_user: SixtydbUser, voice_id: str):
        self._linked_user = linked_user
        self._voice_id = voice_id

    @property
    def linkedUser(self) -> SixtydbUser:
        return self._linked_user

    @property
    def voiceID(self) -> str:
        return self._voice_id

    # ----- payload helper -------------------------------------------------

    def _build_body(self, prompt: str, gen: SixtydbGenerationOptions) -> dict:
        body = {
            "text": prompt,
            "voice_id": self._voice_id,
            "speed": gen.speed,
            "stability": gen.stability,
            "similarity": gen.similarity,
            "enhance": gen.enhance,
        }
        # output_format only applies to /tts-synthesize; /tts-stream ignores it.
        body["output_format"] = gen.output_format
        if gen.language:
            body["language"] = gen.language
        return body

    # ----- REST one-shot --------------------------------------------------

    def generate_audio_v3(
        self,
        prompt: str,
        generation_options: SixtydbGenerationOptions = SixtydbGenerationOptions(),
    ) -> Tuple[Future, Future]:
        """REST POST /tts-synthesize. Returns (audio_future, info_future)
        matching elevenlabslib.Voice.generate_audio_v3's shape.

        audio_future resolves to mp3 bytes (or wav/ogg/flac per output_format).
        info_future resolves to a dict { sample_rate, duration_seconds, encoding }.
        """
        body = self._build_body(prompt, generation_options)
        audio_future: Future = concurrent.futures.Future()
        info_future: Future = concurrent.futures.Future()

        def worker():
            try:
                resp = requests.post(
                    SIXTYDB_REST_URL,
                    headers=self._linked_user.headers,
                    json=body,
                    timeout=60,
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"60db /tts-synthesize {resp.status_code}: {resp.text}"
                    )
                data = resp.json()
                audio_b64 = data.get("audio_base64")
                if not audio_b64:
                    raise RuntimeError("60db response missing audio_base64")
                audio_future.set_result(base64.b64decode(audio_b64))
                info_future.set_result({
                    "sample_rate": data.get("sample_rate"),
                    "duration_seconds": data.get("duration_seconds"),
                    "encoding": data.get("encoding"),
                    "output_format": data.get("output_format"),
                })
            except Exception as e:
                # Guard against re-setting an already-resolved future. The
                # success path sets audio_future first; if info_future.set_result
                # then raised, we must NOT re-set audio_future or set_exception
                # will hit InvalidStateError.
                if not audio_future.done():
                    audio_future.set_exception(e)
                if not info_future.done():
                    info_future.set_exception(e)

        threading.Thread(target=worker, daemon=True).start()
        return audio_future, info_future

    # ----- NDJSON streaming -----------------------------------------------

    def stream_audio_v3(
        self,
        prompt: str,
        generation_options: SixtydbGenerationOptions = SixtydbGenerationOptions(),
    ) -> Iterator[bytes]:
        """REST POST /tts-stream. Yields decoded audio chunks (one per NDJSON line)
        as they arrive. Caller can concatenate or feed into a player frame-by-frame.

        Stops on {"type": "complete"}. Raises on {"type": "error"}.
        """
        body = self._build_body(prompt, generation_options)
        # /tts-stream doesn't accept output_format in the docs; drop it.
        body.pop("output_format", None)

        resp = requests.post(
            SIXTYDB_STREAM_URL,
            headers=self._linked_user.headers,
            json=body,
            stream=True,
            timeout=120,
        )
        # try/finally guarantees the underlying HTTP connection is released
        # even if the caller abandons the generator mid-stream (e.g. break,
        # exception in the consumer loop).
        try:
            if resp.status_code != 200:
                raise RuntimeError(f"60db /tts-stream {resp.status_code}: {resp.text}")

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = evt.get("type")
                if t == "chunk":
                    ac = (evt.get("result") or {}).get("audioContent")
                    if ac:
                        yield base64.b64decode(ac)
                elif t == "complete":
                    break
                elif t == "error":
                    raise RuntimeError(f"60db stream error: {evt.get('message')}")
        finally:
            resp.close()

    # ----- WebSocket session ---------------------------------------------

    def generate_websocket_audio(
        self,
        prompt: str,
        generation_options: SixtydbGenerationOptions = SixtydbGenerationOptions(),
        sample_rate: int = 16000,
    ) -> bytes:
        """Per-text WS session against wss://api.60db.ai/ws/tts.

        Collects LINEAR16 PCM frames into one wav-headered blob. Synchronous
        wrapper around asyncio so it composes with the rest of elevenlabslib.

        Returns wav bytes (with a RIFF header) so callers can pipe straight
        into play_audio_v2(..., audioFormat="mp3_44100_128") — actually wav
        decodes natively via soundfile too.
        """
        # Use the SYNC websockets API to match elevenlabslib's existing style
        # (Voice._generate_websocket uses websockets.sync.connection) AND to
        # avoid asyncio.run() failing when called from inside a running event
        # loop (e.g. async web handlers, Jupyter, etc.).
        try:
            from websockets.sync.client import connect as ws_connect
        except ImportError as e:
            raise RuntimeError(
                "websockets >= 12.0 is required for SixtydbVoice WS mode"
            ) from e

        url = f"{SIXTYDB_WS_URL}?apiKey={self._linked_user.api_key}"
        context_id = str(uuid.uuid4())
        parts: list[bytes] = []

        with ws_connect(url) as ws:
            first = json.loads(ws.recv())
            # Check key presence, NOT truthiness — connection_established
            # may legitimately be an empty {} on some workspace configs.
            if "connection_established" not in first:
                raise RuntimeError(f"60db WS unexpected first frame: {first}")

            ws.send(json.dumps({"create_context": {
                "context_id": context_id,
                "voice_id": self._voice_id,
                "audio_config": {
                    "audio_encoding": "LINEAR16",
                    "sample_rate_hertz": sample_rate,
                },
                "speed": generation_options.speed,
                "stability": generation_options.stability,
                "similarity": generation_options.similarity,
            }}))
            ctx = json.loads(ws.recv())
            if "context_created" not in ctx:
                raise RuntimeError(f"60db WS expected context_created, got: {ctx}")

            ws.send(json.dumps({"send_text": {"context_id": context_id, "text": prompt}}))
            ws.send(json.dumps({"flush_context": {"context_id": context_id}}))

            while True:
                evt = json.loads(ws.recv())
                if evt.get("audio_chunk", {}).get("audioContent"):
                    parts.append(base64.b64decode(evt["audio_chunk"]["audioContent"]))
                # Key-presence checks, not truthiness — 60db may send
                # {"flush_completed": {}} / {"error": {}} (empty subobjects);
                # `{}` is falsy in Python so .get() would silently skip them
                # and the loop would spin forever waiting for the next frame.
                elif "flush_completed" in evt:
                    break
                elif "error" in evt:
                    msg = (evt.get("error") or {}).get("message", "unknown")
                    raise RuntimeError(f"60db WS error: {msg}")

            try:
                ws.send(json.dumps({"close_context": {"context_id": context_id}}))
            except Exception:
                pass

        return _pcm_to_wav(b"".join(parts), sample_rate)

    # ----- convenience: generate + play -----------------------------------

    def generate_play_audio_v2(
        self,
        prompt: str,
        playbackOptions: PlaybackOptions = PlaybackOptions(),
        generation_options: SixtydbGenerationOptions = SixtydbGenerationOptions(),
    ):
        """Generate via REST one-shot, then play through elevenlabslib's
        in-process playback (device selection, callbacks, runInBackground).

        Matches elevenlabslib.Voice.generate_play_audio_v2's call shape so
        consumers can swap between providers without changing call sites.
        """
        # play_audio_v2 expects elevenlabslib-style format strings ('mp3_44100_128',
        # 'pcm_44100', 'ulaw_8000') — it does NOT understand bare 'wav'/'ogg'/'flac'.
        # Force mp3 here so playback always works, regardless of what the caller
        # set on their generation_options. Users who want a different container
        # should call generate_audio_v3() + save the bytes themselves.
        if generation_options.output_format != "mp3":
            generation_options = dataclasses.replace(generation_options, output_format="mp3")
        audio_future, _info_future = self.generate_audio_v3(prompt, generation_options)
        audio_bytes = audio_future.result()
        return play_audio_v2(audio_bytes, playbackOptions, audioFormat="mp3_44100_128")


class SixtydbChat:
    """60db LLM client — wraps the OpenAI-compatible chat completions endpoint.

    No direct elevenlabslib parallel; this is opt-in for users who want to
    pair TTS with text generation inside the same library.

    Docs: https://docs.60db.ai/api-reference/llm/chat-completion
    """

    def __init__(self, linked_user: SixtydbUser, model: str = SIXTYDB_DEFAULT_CHAT_MODEL):
        self._linked_user = linked_user
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: list,
        model: Optional[str] = None,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        base_url: str = SIXTYDB_CHAT_BASE_URL,
    ) -> Union[str, Iterator[str]]:
        """Send a chat-completion request.

        Args:
            messages: OpenAI-style [{"role":"system|user|assistant","content":"..."}].
            model: Override the model set on this SixtydbChat (defaults to constructor's).
            stream: If True, returns an iterator of content deltas (SSE chunks).
                    If False, returns the full assistant text as a single string.
            max_tokens / temperature: Optional model knobs.
            base_url: Override the API base if 60db ever splits chat onto a
                      different host.

        Returns:
            str (non-streaming) OR Iterator[str] (streaming).
        """
        payload: dict = {
            "model": model or self._model,
            "messages": messages,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        url = f"{base_url}/chat/completions"

        if not stream:
            resp = requests.post(
                url,
                headers=self._linked_user.headers,
                json=payload,
                timeout=120,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"60db /chat/completions {resp.status_code}: {resp.text}")
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as e:
                raise RuntimeError(
                    f"60db response shape unexpected: {str(data)[:200]}"
                ) from e

        return self._stream_chat(url, payload)

    def _stream_chat(self, url: str, payload: dict) -> Iterator[str]:
        """SSE iterator. Yields content deltas as they arrive.

        60db's SSE has two non-OpenAI envelope frames (chat_id leading, done
        trailing) plus the OpenAI-style data: {choices:[{delta:{content}}]}
        lines. We skip anything without a delta.content field.
        """
        resp = requests.post(
            url,
            headers=self._linked_user.headers,
            json=payload,
            stream=True,
            timeout=120,
        )
        # try/finally guarantees the connection is released even if the caller
        # abandons the generator mid-stream.
        try:
            if resp.status_code != 200:
                raise RuntimeError(f"60db /chat/completions {resp.status_code}: {resp.text}")

            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                body = raw[len("data: "):]
                if body.strip() == "[DONE]":
                    break
                try:
                    evt = json.loads(body)
                except json.JSONDecodeError:
                    continue
                # Skip 60db's envelope frames (chat_id / done) — they have no choices.
                choices = evt.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content
        finally:
            resp.close()


# ----- internal helpers ---------------------------------------------------

def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Wrap raw LINEAR16 PCM in a RIFF/WAVE header so soundfile/sounddevice
    can decode it without any extra dep."""
    byte_rate = (sample_rate * channels * bits_per_sample) // 8
    block_align = (channels * bits_per_sample) // 8
    header = io.BytesIO()
    header.write(b"RIFF")
    header.write((36 + len(pcm)).to_bytes(4, "little"))
    header.write(b"WAVE")
    header.write(b"fmt ")
    header.write((16).to_bytes(4, "little"))
    header.write((1).to_bytes(2, "little"))  # PCM
    header.write(channels.to_bytes(2, "little"))
    header.write(sample_rate.to_bytes(4, "little"))
    header.write(byte_rate.to_bytes(4, "little"))
    header.write(block_align.to_bytes(2, "little"))
    header.write(bits_per_sample.to_bytes(2, "little"))
    header.write(b"data")
    header.write(len(pcm).to_bytes(4, "little"))
    return header.getvalue() + pcm
