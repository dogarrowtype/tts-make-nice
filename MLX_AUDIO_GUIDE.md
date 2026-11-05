# Using tts_proxy.py with mlx-audio

Guide for running `tts_proxy.py` in front of an [mlx-audio](https://github.com/Blaizzy/mlx-audio) server (`mlx_audio/server.py`), with notes on the server-specific behaviors verified from the source, including voice cloning.

## Setup

Start the mlx-audio server (upstream), then the proxy:

```bash
# Upstream (default port 8000)
mlx_audio.server

# Proxy — point clients here instead
python tts_proxy.py --upstream http://localhost:8000 --port 9992
```

With a config file:

```bash
python tts_proxy.py --config proxy_config.json
```

```json
{
  "upstream": "http://localhost:8000",
  "port": 9992,
  "api_defaults": {
    "model": "mlx-community/Spark-TTS-0.5B-8bit"
  }
}
```

## The `/v1/audio/speech` request schema

mlx-audio's `SpeechRequest` accepts these JSON fields. The proxy forwards every field except `input` and `response_format` untouched, so all of them work through the proxy (from the client, or injected via `api_defaults` / `api_overrides`):

| Field | Default | Notes |
|---|---|---|
| `model` | — (required) | HF repo id or local path sampled on first use |
| `input` | — (required) | Rewritten by the proxy (normalization + segmentation) |
| `voice` | `null` | Model-specific voice name |
| `instruct` | `null` | Instruction prompt for instruct-capable models |
| `speed` | `1.0` | |
| `gender` | `"male"` | Applied upstream even if the client omits it |
| `pitch` | `1.0` | |
| `lang_code` | `"a"` | |
| `ref_audio` | `null` | **Path on the upstream machine** — see voice cloning below |
| `ref_text` | `null` | Transcript of `ref_audio` |
| `temperature` | `0.7` | Explicit request value wins, then the model's `generate_config`, then this default |
| `top_p` | `0.95` | Same resolution order as `temperature` |
| `top_k` | `40` | Same resolution order |
| `repetition_penalty` | `1.0` | Same resolution order |
| `max_tokens` | `1200` | Generation budget per request — the proxy's segmentation means each segment gets this budget |
| `response_format` | `"mp3"` | The proxy always requests `wav` upstream and re-encodes client-side |
| `stream` | `false` | **Not supported through the proxy** — do not enable |
| `streaming_interval` | `2.0` | Only relevant when streaming |
| `verbose` | `false` | |

## Voice cloning

mlx-audio clones from two JSON fields:

- **`ref_audio`** — a **filesystem path readable by the machine running mlx-audio**. Not an upload, not base64. The server validates it with `os.path.exists()` and returns HTTP 400 (`Reference audio file not found`) if missing. At inference time it is loaded with `load_audio(ref_audio, sample_rate=model.sample_rate, ...)`, resampled to the model's native rate automatically; Spark-type models additionally get volume normalization.
- **`ref_text`** — the transcript of what is spoken in `ref_audio` (optional, improves clone quality).

Configure it once in the proxy config so every request uses the same cloned voice:

```json
{
  "api_overrides": {
    "ref_audio": "/absolute/path/on/upstream/machine/reference.wav",
    "ref_text": "Exact transcript of the reference clip."
  }
}
```

Or let clients choose per-request by sending `ref_audio` / `ref_text` in their own JSON body — those fields pass straight through the proxy.

### Voice cloning caveats (verified from source)

- **The path must exist on the upstream machine.** The proxy logs a warning at startup if a configured `ref_audio` is missing locally; if proxy and upstream run on different machines, ensure they share a filesystem or ignore the warning.
- **Cloning requests bypass batching.** Any request carrying `ref_audio` or `ref_text` is forced onto mlx-audio's serial generation path. Throughput is unaffected for a single user, but don't expect concurrent cloned requests to batch.
- **Sampling params resolve through the model's `generate_config`** when not explicitly set. To pin behavior (recommended for consistent clones), set them in `api_overrides`, e.g. `"temperature": 0.7, "top_p": 0.95, "top_k": 40`.
- **`ref_text` accuracy matters.** A mismatched transcript degrades clone quality more than a slightly imperfect clip.

## Proxy behaviors that matter with mlx-audio

- **Segment budget**: the proxy splits long input into segments (~380 chars by default) and sends each as its own request. Each segment gets the full upstream `max_tokens` budget (default 1200), so very long inputs work, but total latency is the sum of per-segment serial generation.
- **`model` is required per segment**: clients must send `model`, or supply it via `api_defaults` in the config.
- **`gender` default**: mlx-audio applies `gender: "male"` when the client omits it. If your model ignores gender this is harmless.
- **Unknown extra fields** in the client body are forwarded to the upstream; FastAPI silently ignores fields not in `SpeechRequest`, so stray params don't cause errors (but silently do nothing).
- **Upstream format**: the proxy always asks for `wav` (lossless) and re-encodes to the client's requested format after stitching, so `response_format: "opus"/"mp3"/"flac"/"aac"` requires ffmpeg on the proxy machine (falls back to `wav` with a logged warning).
