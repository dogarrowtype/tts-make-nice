#!/usr/bin/env python3
"""
tts_proxy.py — A proxy between a client and an OpenAI-compatible TTS API server.

Sits in front of any OpenAI-compatible /v1/audio/speech endpoint and applies
the quality-of-life text & audio handling enhancements originally developed in
testing/tts_server_fast_validation.py:

  * Text normalization (fancy quotes -> ASCII, dashes, ellipsis, bullets,
    emoji stripping, optional all-caps lowercasing)
  * Pipe character (|) splitting for manual segment control
  * Semantic text splitting on a character threshold
  * Per-segment synthesis against the upstream server
  * Trailing/leading silence trimming per segment
  * Random natural pause (default 300-500ms) inserted between segments
  * Retry logic per segment with graceful silence fallback
  * Output re-encoding to mp3/opus/flac/aac/wav with graceful WAV fallback
    when ffmpeg is unavailable

Usage:
    python tts_proxy.py --upstream http://localhost:9991 --port 9992
"""

import argparse
import asyncio
import logging
import os
import platform
import random
import re
import shutil
import subprocess
import time
from datetime import datetime
from io import BytesIO

import demoji
import httpx
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from semantic_text_splitter import TextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Global configuration (populated from CLI args in main)
# ---------------------------------------------------------------------------

UPSTREAM_URL = None          # Base URL of the upstream OpenAI-compatible TTS server
UPSTREAM_API_KEY = None      # Optional API key override for upstream
UPSTREAM_TIMEOUT = 600.0     # Per-segment upstream request timeout (seconds)

PIPE_SPLIT_ENABLED = True
SEGMENTATION_THRESHOLD = 380
ALLOW_ALLCAPS = False
EMOTION_TAGS = ""             # prepended to every segment sent upstream (e.g. "[ upbeat, friendly ]")

PAUSE_MIN_MS = 300
PAUSE_MAX_MS = 500
MAX_RETRY_ATTEMPTS = 3
TRIM_ENABLED = True
TRIM_TOP_DB = 30

# Arbitrary parameters merged into every upstream /v1/audio/speech request.
# api_defaults apply only when the client did not supply the key;
# api_overrides always win over the client.
API_DEFAULTS = {}
API_OVERRIDES = {}

# Voice cloning presets (mlx-audio ref_audio/ref_text contract).
# Named presets are injected per-request via the "voice_clone" control field.
VOICE_CLONING_ENABLED = False
VOICE_CLONING_DEFAULT = None   # preset name applied when client doesn't opt out
VOICE_PRESETS = {}             # name -> dict of upstream params (ref_audio, ref_text, ...)

# Granular QoL feature toggles. Every enhancement can be switched off
# individually via the config file's "features" object.
FEATURES = {
    # --- text normalization rules ---
    "normalize_double_quotes": True,   # fancy double quotes -> "
    "normalize_single_quotes": True,   # fancy single quotes/apostrophes -> '
    "en_dash_to_hyphen": True,         # – -> -
    "ellipsis_to_dots": True,          # … -> ...
    "strip_bullets": True,             # •‣⁃* removed
    "em_dash_padding": True,           # — -> " — " (natural pause)
    "strip_emoji": True,               # remove emoji via demoji
    "lowercase_allcaps": True,         # SHOUTED words -> shouted (models choke on caps)
    # --- segmentation ---
    "pipe_split": True,                # | forces a segment boundary
    "semantic_split": True,            # TextSplitter chunking on seg_threshold
    "manual_pauses": True,             # {pause 1500} injects exact silence (ms)
    "emotion_tags": True,              # prepend configured emotion tags to every segment
    # --- audio post-processing ---
    "trim_silence": True,              # per-segment leading/trailing silence removal
    "segment_pauses": True,            # random pause inserted between segments
}

FFMPEG_AVAILABLE = False

# Media types for supported response formats
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/L16",
}

app = FastAPI(title="TTS Make Nice Proxy",
              description="QoL-enhancing proxy for OpenAI-compatible TTS servers")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Text handling (ported from tts_server_fast_validation.py)
# ---------------------------------------------------------------------------

def normalize_text(text, allow_allcaps=False):
    """Standardize punctuation and characters that TTS models commonly choke on.

    Each rule is gated on FEATURES so it can be toggled individually.
    """
    if FEATURES["normalize_double_quotes"]:
        text = re.sub(r'[“”„‟«»❝❞〝〞〟＂]', '"', text)   # Fancy double quotes
    if FEATURES["normalize_single_quotes"]:
        text = re.sub(r'[‘’‚‛‹›❛❜`´]', "'", text)         # Fancy single quotes/apostrophes
    if FEATURES["en_dash_to_hyphen"]:
        text = re.sub(r'[–]', '-', text)                    # En dash to hyphen
    if FEATURES["ellipsis_to_dots"]:
        text = re.sub(r'…', '...', text)                    # Ellipsis
    if FEATURES["strip_bullets"]:
        text = re.sub(r'[•‣⁃*]', '', text)                  # Bullets to none
    if FEATURES["em_dash_padding"]:
        text = re.sub(r'—', ' — ', text)                    # Em dash padding
    if FEATURES["strip_emoji"]:
        text = demoji.replace(text, "")

    if FEATURES["lowercase_allcaps"] and not allow_allcaps:
        # Convert all-caps words to lowercase (models choke on all caps)
        def lowercase_all_caps(match):
            word = match.group(0)
            if word.isupper() and len(word) > 1:
                return word.lower()
            return word
        text = re.sub(r'\b[A-Z][A-Z]+\b', lowercase_all_caps, text)

    return text


PAUSE_MARKER_RE = re.compile(r'\{\s*pause\s+(\d+)\s*\}', re.IGNORECASE)


def split_on_pipes(text):
    """Split text on pipe characters (|), stripping empties."""
    return [seg.strip() for seg in text.split('|') if seg.strip()]


def split_preserve_pauses(text):
    """Split text into parts, keeping {pause N} markers as ("pause", ms) tuples."""
    parts = []
    pos = 0
    for m in PAUSE_MARKER_RE.finditer(text):
        if m.start() > pos:
            parts.append(text[pos:m.start()])
        parts.append(("pause", int(m.group(1))))  # manual pauses are honored exactly
        pos = m.end()
    if pos < len(text):
        parts.append(text[pos:])
    return parts


def segment_text(text):
    """Split text into synthesis segments.

    Returns a list whose items are either strings (text to synthesize) or
    ("pause", ms) tuples (exact silence to insert). {pause N} markers are
    hard boundaries: text is split around them first, then pipe splitting and
    semantic segmentation apply to the text pieces.
    """
    raw_parts = split_preserve_pauses(text) if FEATURES["manual_pauses"] else [text]
    n_pauses = sum(1 for p in raw_parts if isinstance(p, tuple))
    if n_pauses:
        logging.info(f"Found {n_pauses} manual pause marker(s)")

    all_segments = []
    for part in raw_parts:
        if isinstance(part, tuple):
            all_segments.append(part)
            continue

        if FEATURES["pipe_split"] and PIPE_SPLIT_ENABLED:
            piece_segments = split_on_pipes(part)
        else:
            piece_segments = [part]

        if not FEATURES["semantic_split"]:
            all_segments.extend(seg for seg in piece_segments if seg and seg.strip())
            continue

        for piece in piece_segments:
            if piece.strip():
                splitter = TextSplitter(SEGMENTATION_THRESHOLD)
                all_segments.extend(splitter.chunks(piece.strip()))

    n_text = sum(1 for s in all_segments if not isinstance(s, tuple))
    logging.info(f"Total segments: {n_text} text, {n_pauses} pause(s)")
    return all_segments


def apply_emotion_tags(segments, tags):
    """Prepend emotion tags to every text segment; pause markers pass through.
    No-op when the feature is off or tags are empty."""
    if not FEATURES["emotion_tags"] or not tags or not tags.strip():
        return segments
    prefix = tags.strip()
    tagged = [f"{prefix} {seg}" if isinstance(seg, str) else seg for seg in segments]
    logging.info(f"Emotion tags prepended: {prefix}")
    return tagged


# ---------------------------------------------------------------------------
# Audio handling
# ---------------------------------------------------------------------------

def trim_silence(wav_data, sample_rate, top_db=TRIM_TOP_DB, frame_length=1024, hop_length=512):
    """
    Trim leading and trailing silence using an energy-based end-pointer
    (numpy implementation mirroring librosa.effects.trim).
    Returns (trimmed_audio, seconds_trimmed).
    """
    if wav_data is None or len(wav_data) == 0:
        return wav_data, 0.0
    try:
        # Frame-based RMS energy
        n_frames = 1 + max(0, (len(wav_data) - frame_length) // hop_length)
        if n_frames <= 1:
            return wav_data, 0.0
        frames = np.lib.stride_tricks.as_strided(
            wav_data,
            shape=(n_frames, frame_length),
            strides=(wav_data.strides[0] * hop_length, wav_data.strides[0]),
        )
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        db = 20.0 * np.log10(rms / (np.max(rms) + 1e-12) + 1e-12)

        keep = np.nonzero(db > -top_db)[0]
        if len(keep) == 0:
            return wav_data, 0.0  # Entirely silent — keep as-is

        start = keep[0] * hop_length
        end = min(len(wav_data), (keep[-1] + 1) * hop_length + frame_length)
        trimmed = wav_data[start:end]
        seconds_trimmed = (len(wav_data) - len(trimmed)) / sample_rate
        logging.info(f"Trimmed {seconds_trimmed:.2f}s of silence")
        return trimmed, seconds_trimmed
    except Exception as e:
        logging.warning(f"Error trimming silence: {e}")
        return wav_data, 0.0


def decode_audio(audio_bytes, content_type=""):
    """
    Decode audio bytes to (mono float32 numpy array, sample_rate).
    Tries soundfile directly, falls back to ffmpeg for compressed formats.
    """
    # Try soundfile first (wav, flac, ogg)
    try:
        data, sr = sf.read(BytesIO(audio_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except Exception:
        pass

    if not FFMPEG_AVAILABLE:
        raise RuntimeError(
            f"Cannot decode upstream audio (content-type: {content_type}) "
            "and ffmpeg is not available to convert it."
        )

    # Fallback: pipe through ffmpeg into WAV
    suffix = ".bin"
    if "mpeg" in content_type or "mp3" in content_type:
        suffix = ".mp3"
    elif "opus" in content_type:
        suffix = ".opus"
    elif "aac" in content_type:
        suffix = ".aac"

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as in_f:
        in_f.write(audio_bytes)
        in_path = in_f.name
    out_path = in_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-ac", "1", out_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60,
        )
        data, sr = sf.read(out_path, dtype="float32")
        return data, sr
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.remove(p)


def encode_audio(wav_data, sample_rate, response_format):
    """
    Encode audio to the requested format.
    Returns (audio_bytes, actual_format). Falls back to WAV when conversion fails.
    """
    response_format = (response_format or "mp3").lower()

    # WAV: encode directly with soundfile, no ffmpeg needed
    if response_format in ("wav", "pcm"):
        buf = BytesIO()
        subtype = "PCM_16" if response_format == "wav" else "FLOAT"
        sf.write(buf, wav_data, sample_rate, format="WAV", subtype=subtype)
        return buf.getvalue(), response_format

    if not FFMPEG_AVAILABLE:
        logging.warning(f"FFmpeg not available; returning WAV instead of {response_format}")
        buf = BytesIO()
        sf.write(buf, wav_data, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "wav"

    # libopus only accepts 8/12/16/24/48 kHz input and prefers 48 kHz; it also
    # requires an explicit ogg container extension handled by ffmpeg. Resampling
    # to 48 kHz up front makes opus encoding work regardless of upstream rate.
    encode_sr = 48000 if response_format == "opus" else sample_rate

    codec_args = {
        "mp3":  ["-codec:a", "libmp3lame", "-ar", str(sample_rate), "-b:a", "320k"],
        "opus": ["-f", "ogg", "-codec:a", "libopus", "-ar", "48000", "-b:a", "128k"],
        "flac": ["-codec:a", "flac", "-ar", str(sample_rate), "-compression_level", "12"],
        "aac":  ["-codec:a", "aac", "-ar", str(sample_rate), "-b:a", "128k"],
    }.get(response_format)
    if codec_args is None:
        logging.warning(f"Unsupported response_format '{response_format}'; returning WAV")
        buf = BytesIO()
        sf.write(buf, wav_data, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "wav"

    # Resample if the encoder needs a different rate (keeps WAV header honest)
    if encode_sr != sample_rate:
        wav_data = resample_linear(wav_data, sample_rate, encode_sr)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_f:
        sf.write(in_f, wav_data, encode_sr, format="WAV", subtype="PCM_16")
        in_path = in_f.name
    out_path = f"{in_path}.{response_format}"
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, *codec_args, out_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        )
        if proc.returncode != 0:
            logging.error(f"ffmpeg failed for {response_format}: {proc.stderr.decode(errors='replace')}")
            raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
        with open(out_path, "rb") as f:
            return f.read(), response_format
    except Exception as e:
        logging.warning(f"Error converting to {response_format}: {e}. Returning WAV instead.")
        buf = BytesIO()
        sf.write(buf, wav_data, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "wav"
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.remove(p)


# ---------------------------------------------------------------------------
# Upstream communication
# ---------------------------------------------------------------------------

class UpstreamClientError(Exception):
    """Upstream returned a 4xx — a request problem that retries cannot fix."""

    def __init__(self, status_code, detail):
        super().__init__(f"Upstream returned {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


async def synthesize_segment(client, segment, base_body):
    """
    Send one text segment to the upstream TTS server.
    Returns (float32 numpy array, sample_rate).
    """
    body = dict(base_body)
    body["input"] = segment
    # Always request WAV from upstream so we can decode/trim/stitch without loss
    body["response_format"] = "wav"

    headers = {}
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"

    response = await client.post(
        f"{UPSTREAM_URL}/v1/audio/speech",
        json=body,
        headers=headers,
    )
    if response.status_code != 200:
        detail = response.text[:500]
        if 400 <= response.status_code < 500:
            raise UpstreamClientError(response.status_code, detail)
        raise RuntimeError(f"Upstream returned {response.status_code}: {detail}")
    content_type = response.headers.get("content-type", "")
    return decode_audio(response.content, content_type)


async def generate_speech(text, base_body, emotion_tags=None):
    """
    Full QoL pipeline: normalize -> segment -> tag -> synthesize per segment ->
    trim & stitch with natural pauses.
    Returns (float32 numpy array, sample_rate).
    """
    text = normalize_text(text, allow_allcaps=ALLOW_ALLCAPS)
    segments = segment_text(text)
    segments = apply_emotion_tags(segments, emotion_tags)

    if not segments:
        raise HTTPException(status_code=400, detail="No speakable text after preprocessing")

    wavs = []
    sample_rate = None

    timeout = httpx.Timeout(UPSTREAM_TIMEOUT, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for i, seg in enumerate(segments):
            # Manual pause marker: exact silence, replaces the random auto-pause
            if isinstance(seg, tuple):
                if sample_rate is None:
                    sample_rate = 24000  # nothing synthesized yet; assumption corrected later
                logging.info(f"Inserting manual pause: {seg[1]}ms")
                wavs.append(np.zeros(int(sample_rate * seg[1] / 1000.0), dtype=np.float32))
                continue

            logging.info(f"Processing one segment:\n{seg}")
            retry_count = 0
            wav = None

            while retry_count < MAX_RETRY_ATTEMPTS:
                try:
                    wav, sr = await synthesize_segment(client, seg, base_body)
                    if sample_rate is None:
                        sample_rate = sr
                        # Fix any manual pause(s) inserted before the first segment
                        wavs = [resample_linear(w, 24000, sample_rate)
                                if len(w) and isinstance(w, np.ndarray) else w for w in wavs]
                    elif sr != sample_rate:
                        # Keep a consistent sample rate across segments
                        wav = resample_linear(wav, sr, sample_rate)
                    break
                except UpstreamClientError:
                    raise  # 4xx: retrying can't help; let it reach the client
                except Exception as e:
                    logging.error(f"Error synthesizing segment: {e}")
                    retry_count += 1
                    if retry_count >= MAX_RETRY_ATTEMPTS:
                        logging.warning("Max retries reached. Using silence for this segment.")
                        if sample_rate is None:
                            sample_rate = 24000  # sane default if nothing succeeded yet
                        wav = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
                    await asyncio.sleep(0.5)

            if wav is not None:
                if TRIM_ENABLED and FEATURES["trim_silence"]:
                    wav, _ = trim_silence(wav, sample_rate)
                wavs.append(wav)
                # Random auto-pause between segments — skipped when the next
                # item is an explicit pause marker, which takes over the gap
                next_is_pause = (i + 1 < len(segments)
                                 and isinstance(segments[i + 1], tuple))
                if FEATURES["segment_pauses"] and not next_is_pause:
                    pause_s = random.uniform(PAUSE_MIN_MS / 1000.0, PAUSE_MAX_MS / 1000.0)
                    wavs.append(np.zeros(int(sample_rate * pause_s), dtype=np.float32))
                logging.info(
                    f"Processed one segment"
                    f"{' after ' + str(retry_count) + ' retries' if retry_count > 0 else ''}."
                )

    if not wavs:
        raise HTTPException(status_code=502, detail="Upstream produced no audio")

    final = np.concatenate(wavs, axis=0).astype(np.float32)
    return final, sample_rate


def resample_linear(audio, in_sr, out_sr):
    """Lightweight linear-interpolation resampler (avoids heavy DSP deps)."""
    if in_sr == out_sr or len(audio) == 0:
        return audio
    duration = len(audio) / in_sr
    n_out = int(round(duration * out_sr))
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/audio/speech")
async def tts(request: Request):
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
        elif "application/x-www-form-urlencoded" in content_type:
            body = dict(await request.form())
        else:
            body = await request.json()

        client_ip = request.client.host if request.client else "unknown"
        print("\n==== TTS REQUEST ====")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Client IP: {client_ip}")
        print(f"Content-Type: {content_type}")
        print("Request data:")
        import json as _json
        print(_json.dumps(body, indent=2))
        print("====================\n")

        input_text = body.get("input")
        if not input_text:
            raise HTTPException(status_code=400, detail="Input text is required")

        response_format = (body.get("response_format") or "mp3").lower()

        # "voice_clone" / "emotion_tags" are proxy control fields, never forwarded upstream
        voice_clone = body.get("voice_clone", None)
        # Per-request override; an explicit "" disables the configured default
        emotion_tags = body.get("emotion_tags", None)
        if emotion_tags is None:
            emotion_tags = EMOTION_TAGS

        # Everything except input/response_format (and proxy control fields)
        # is forwarded upstream, with config-file defaults/overrides merged in
        base_body = {k: v for k, v in body.items()
                     if k not in ("input", "response_format", "voice_clone", "emotion_tags")}
        for k, v in API_DEFAULTS.items():
            base_body.setdefault(k, v)
        base_body.update(API_OVERRIDES)

        client_has_ref = "ref_audio" in body  # client's own fields always win

        if voice_clone is False:
            # Per-request opt-out: strip cloning params the config may have added
            if not client_has_ref:
                base_body.pop("ref_audio", None)
                base_body.pop("ref_text", None)
        elif isinstance(voice_clone, str) and voice_clone:
            # Explicit preset selection for this request
            if not VOICE_CLONING_ENABLED:
                raise HTTPException(status_code=400,
                    detail="Voice cloning presets are disabled in the proxy config")
            preset = VOICE_PRESETS.get(voice_clone)
            if preset is None:
                raise HTTPException(status_code=400,
                    detail=f"Unknown voice cloning preset '{voice_clone}'. "
                           f"Available: {sorted(VOICE_PRESETS)}")
            base_body.update(preset)
        elif voice_clone in (None, True):
            # No explicit choice: apply the configured default preset unless the
            # request/config already specifies a reference
            if (VOICE_CLONING_ENABLED and VOICE_CLONING_DEFAULT
                    and "ref_audio" not in base_body):
                base_body.update(VOICE_PRESETS[VOICE_CLONING_DEFAULT])

        wav_data, sample_rate = await generate_speech(input_text, base_body, emotion_tags)
        audio_bytes, actual_format = encode_audio(wav_data, sample_rate, response_format)

        media_type = MEDIA_TYPES.get(actual_format, "audio/wav")
        return Response(content=audio_bytes, media_type=media_type)

    except HTTPException:
        raise
    except UpstreamClientError as e:
        logging.error(f"Upstream rejected the request: {e.detail}")
        return JSONResponse(status_code=e.status_code,
                            content={"error": "Upstream rejected the request",
                                     "upstream_detail": e.detail})
    except Exception as e:
        logging.error(f"Error in TTS endpoint: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.options("/v1/audio/speech")
async def tts_options():
    return {}


PLAYGROUND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "playground.html")


@app.get("/", include_in_schema=False)
@app.get("/playground.html", include_in_schema=False)
async def playground():
    """Serve the audio playground page."""
    try:
        with open(PLAYGROUND_PATH, encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="playground.html not found")


@app.post("/v1/preview")
async def preview(request: Request):
    """Show exactly how the proxy will process input text: the normalized text
    and the segments that would be sent upstream. Nothing is synthesized."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required")
    input_text = body.get("input")
    if not input_text:
        raise HTTPException(status_code=400, detail="Input text is required")

    emotion_tags = body.get("emotion_tags", None)
    if emotion_tags is None:
        emotion_tags = EMOTION_TAGS

    normalized = normalize_text(input_text, allow_allcaps=ALLOW_ALLCAPS)
    segments = segment_text(normalized)
    segments = apply_emotion_tags(segments, emotion_tags)
    display = [f"⏸ pause {s[1]}ms" if isinstance(s, tuple) else s for s in segments]
    return {
        "original_length": len(input_text),
        "normalized": normalized,
        "segments": display,
        "segment_count": len(segments),
        "emotion_tags": emotion_tags if FEATURES["emotion_tags"] else "",
        "features": {k: v for k, v in FEATURES.items()},
    }


@app.get("/v1/models")
async def list_models():
    """Pass through the upstream model list, with a sensible fallback."""
    try:
        headers = {}
        if UPSTREAM_API_KEY:
            headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{UPSTREAM_URL}/v1/models", headers=headers)
            if r.status_code == 200:
                return r.json()
    except Exception as e:
        logging.warning(f"Could not fetch upstream models: {e}")

    return {
        "object": "list",
        "data": [
            {"id": "tts-1", "object": "model", "created": int(time.time()), "owned_by": "upstream"},
            {"id": "tts-1-hd", "object": "model", "created": int(time.time()), "owned_by": "upstream"},
        ],
    }


@app.get("/v1/audio/voices")
async def list_voices(model: str = None):
    """Pass through the upstream voice list (mlx-audio voice-pack convention)."""
    headers = {}
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    try:
        params = {"model": model} if model else {}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{UPSTREAM_URL}/v1/audio/voices", params=params, headers=headers)
            return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception as e:
        logging.warning(f"Could not fetch upstream voices: {e}")
        return {"object": "list", "model": model, "data": []}


@app.get("/v1/config")
async def proxy_config_info():
    """Public proxy configuration for clients/playgrounds: default model and
    voice cloning presets (names only — ref paths stay server-side)."""
    return {
        "model_default": API_DEFAULTS.get("model"),
        "emotion_tags": EMOTION_TAGS if FEATURES["emotion_tags"] else "",
        "api_defaults": {k: v for k, v in API_DEFAULTS.items() if k != "model"},
        "voice_cloning": {
            "enabled": VOICE_CLONING_ENABLED,
            "default": VOICE_CLONING_DEFAULT,
            "presets": [
                {"name": name, "has_ref_text": bool(params.get("ref_text"))}
                for name, params in VOICE_PRESETS.items()
            ],
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "TTS make-nice proxy is running",
        "upstream": UPSTREAM_URL,
        "config": {
            "pipe_split": PIPE_SPLIT_ENABLED,
            "segmentation_threshold": SEGMENTATION_THRESHOLD,
            "allow_allcaps": ALLOW_ALLCAPS,
            "emotion_tags": EMOTION_TAGS,
            "trim": TRIM_ENABLED,
            "pause_ms": [PAUSE_MIN_MS, PAUSE_MAX_MS],
            "max_retries": MAX_RETRY_ATTEMPTS,
            "api_defaults": API_DEFAULTS,
            "api_overrides": API_OVERRIDES,
            "features": FEATURES,
            "voice_cloning": {
                "enabled": VOICE_CLONING_ENABLED,
                "default": VOICE_CLONING_DEFAULT,
                "presets": sorted(VOICE_PRESETS),
            },
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def is_ffmpeg_available():
    is_windows = platform.system().lower() == "windows"
    for cmd in (["ffmpeg.exe", "ffmpeg"] if is_windows else ["ffmpeg"]):
        if shutil.which(cmd) is not None:
            return True
    return False


def strip_json_comments(text):
    """Remove // and /* */ comments without touching string literals
    (URLs contain //, so a naive regex strip is unsafe)."""
    out, i, n = [], 0, len(text)
    in_string, escaped = False, False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
            i += 1
        elif c == '"':
            in_string = True
            out.append(c)
            i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def main():
    global UPSTREAM_URL, UPSTREAM_API_KEY, UPSTREAM_TIMEOUT
    global PIPE_SPLIT_ENABLED, SEGMENTATION_THRESHOLD, ALLOW_ALLCAPS, EMOTION_TAGS
    global PAUSE_MIN_MS, PAUSE_MAX_MS, MAX_RETRY_ATTEMPTS
    global TRIM_ENABLED, TRIM_TOP_DB, FFMPEG_AVAILABLE
    global API_DEFAULTS, API_OVERRIDES
    global VOICE_CLONING_ENABLED, VOICE_CLONING_DEFAULT, VOICE_PRESETS

    # First pass: locate a config file so it can supply defaults for the real parse.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    # Auto-load proxy_config.json when no --config is given: prefer the working
    # directory, then the script's directory.
    config_path = pre_args.config
    if config_path is None:
        candidates = [
            os.path.abspath("proxy_config.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy_config.json"),
        ]
        config_path = next((c for c in dict.fromkeys(candidates) if os.path.exists(c)), None)
        if config_path:
            logging.info(f"Auto-loading default config: {config_path}")

    config = {}
    if config_path:
        import json as _json
        try:
            with open(config_path) as f:
                config = _json.loads(strip_json_comments(f.read()))
            if not isinstance(config, dict):
                raise ValueError("config root must be a JSON object")
            logging.info(f"Loaded config from {config_path}")
        except Exception as e:
            logging.error(f"Failed to load config file {config_path}: {e}")
            raise SystemExit(1)

    parser = argparse.ArgumentParser(
        description="QoL-enhancing proxy for OpenAI-compatible TTS API servers. "
                    "Every flag can also be set in a JSON config file (--config) using "
                    "the same underscore names; CLI flags take precedence over the file. "
                    "The file may additionally define 'api_defaults' and 'api_overrides' "
                    "objects whose keys are injected into every upstream request."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON config file. Defaults to proxy_config.json "
                             "in the working or script directory when present")
    parser.add_argument("--upstream", type=str, required=not config.get("upstream"),
                        help="Base URL of the upstream OpenAI-compatible TTS server "
                             "(e.g. http://localhost:9991)")
    parser.add_argument("--model", type=str, default=None,
                        help="Default model sent upstream when the client omits it")
    parser.add_argument("--emotion_tags", type=str, default=None,
                        help="Emotion tags prepended to every segment sent upstream, "
                             "e.g. '[ upbeat, friendly ]'")
    parser.add_argument("--port", type=int, default=9992, help="Port to run the proxy on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key to send to the upstream server (overrides client's)")
    parser.add_argument("--upstream_timeout", type=float, default=600.0,
                        help="Timeout in seconds for upstream synthesis requests")

    parser.add_argument("--seg_threshold", type=int, default=380,
                        help="Character limit for semantic text segments")
    parser.add_argument("--disable_pipe_split", action="store_true",
                        help="Disable automatic text splitting on pipe characters (|)")
    parser.add_argument("--allow_allcaps", action="store_true",
                        help="Allow words with all capital letters")

    parser.add_argument("--no_trim", action="store_true",
                        help="Disable per-segment silence trimming")
    parser.add_argument("--trim_top_db", type=float, default=30.0,
                        help="dB threshold below peak treated as silence when trimming")
    parser.add_argument("--pause_min_ms", type=int, default=300,
                        help="Minimum random pause between segments (ms)")
    parser.add_argument("--pause_max_ms", type=int, default=500,
                        help="Maximum random pause between segments (ms)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max synthesis attempts per segment before silence fallback")

    # Apply config-file values as parser defaults (CLI flags still win).
    # Sub-dicts like api_defaults/api_overrides are handled separately below.
    known_settings = {
        "upstream", "port", "host", "api_key", "upstream_timeout", "model",
        "emotion_tags",
        "seg_threshold", "disable_pipe_split", "allow_allcaps",
        "no_trim", "trim_top_db", "pause_min_ms", "pause_max_ms", "max_retries",
    }
    parser.set_defaults(**{k: v for k, v in config.items() if k in known_settings})
    reserved_sections = ("api_defaults", "api_overrides", "features", "voice_cloning")
    for key in config:
        if key not in known_settings and key not in reserved_sections:
            logging.warning(f"Ignoring unknown config key: {key}")

    args = parser.parse_args()

    UPSTREAM_URL = args.upstream.rstrip("/")
    UPSTREAM_API_KEY = args.api_key or os.environ.get("UPSTREAM_API_KEY")
    UPSTREAM_TIMEOUT = args.upstream_timeout

    PIPE_SPLIT_ENABLED = not args.disable_pipe_split
    SEGMENTATION_THRESHOLD = args.seg_threshold
    ALLOW_ALLCAPS = args.allow_allcaps

    TRIM_ENABLED = not args.no_trim
    TRIM_TOP_DB = args.trim_top_db
    PAUSE_MIN_MS = args.pause_min_ms
    PAUSE_MAX_MS = args.pause_max_ms
    MAX_RETRY_ATTEMPTS = args.max_retries
    if args.emotion_tags:
        EMOTION_TAGS = args.emotion_tags
        logging.info(f"Emotion tags: {EMOTION_TAGS}")

    API_DEFAULTS = dict(config.get("api_defaults", {}))
    API_OVERRIDES = dict(config.get("api_overrides", {}))
    if args.model and "model" not in API_DEFAULTS:
        API_DEFAULTS["model"] = args.model
    if API_DEFAULTS:
        logging.info(f"API defaults from config: {sorted(API_DEFAULTS)}")
    if API_OVERRIDES:
        logging.info(f"API overrides from config (always win): {sorted(API_OVERRIDES)}")

    # Voice cloning presets
    vc_cfg = config.get("voice_cloning", {})
    if vc_cfg and not isinstance(vc_cfg, dict):
        logging.error('"voice_cloning" config section must be an object; ignoring')
        vc_cfg = {}
    VOICE_CLONING_ENABLED = bool(vc_cfg.get("enabled", False))
    VOICE_CLONING_DEFAULT = vc_cfg.get("default")
    VOICE_PRESETS = dict(vc_cfg.get("presets", {}))
    for name, params in list(VOICE_PRESETS.items()):
        if not isinstance(params, dict) or not params.get("ref_audio"):
            logging.warning(f"Voice cloning preset '{name}' has no ref_audio; dropping it")
            del VOICE_PRESETS[name]
            continue
        if not os.path.exists(params["ref_audio"]):
            logging.warning(
                f"Voice cloning preset '{name}': ref_audio not found locally "
                f"({params['ref_audio']}) — it must exist on the machine running "
                "the upstream server or requests will fail with HTTP 400."
            )
    if VOICE_CLONING_DEFAULT and VOICE_CLONING_DEFAULT not in VOICE_PRESETS:
        logging.warning(f"voice_cloning.default '{VOICE_CLONING_DEFAULT}' is not a "
                        f"defined preset; ignoring")
        VOICE_CLONING_DEFAULT = None
    if VOICE_CLONING_ENABLED:
        logging.info(f"Voice cloning presets: {sorted(VOICE_PRESETS)}"
                     + (f" (default: {VOICE_CLONING_DEFAULT})" if VOICE_CLONING_DEFAULT else ""))

    # Merge granular feature toggles from config
    features_cfg = config.get("features", {})
    for key, value in features_cfg.items():
        if key in FEATURES:
            FEATURES[key] = bool(value)
        else:
            logging.warning(f"Ignoring unknown feature toggle: {key} "
                            f"(valid: {sorted(FEATURES)})")
    disabled = sorted(k for k, v in FEATURES.items() if not v)
    if disabled:
        logging.info(f"QoL features disabled: {disabled}")

    # Voice cloning sanity check: mlx-audio expects ref_audio to be a
    # filesystem path readable by the *upstream* server. Warn early when the
    # configured path is missing locally (same-machine setups) instead of
    # failing with a 400 on every request.
    for source, params in (("api_defaults", API_DEFAULTS), ("api_overrides", API_OVERRIDES)):
        ref_audio = params.get("ref_audio")
        if ref_audio:
            if os.path.exists(ref_audio):
                logging.info(f"Voice cloning enabled via {source}: ref_audio={ref_audio}")
            else:
                logging.warning(
                    f"{source}.ref_audio not found locally: {ref_audio} — "
                    "it must exist on the machine running the upstream server "
                    "or requests will fail with HTTP 400."
                )

    FFMPEG_AVAILABLE = is_ffmpeg_available()
    logging.info(f"FFmpeg is {'available' if FFMPEG_AVAILABLE else 'not available'}")
    logging.info(f"Upstream TTS server: {UPSTREAM_URL}")
    logging.info(f"Starting proxy on http://{args.host}:{args.port}/v1/audio/speech")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
