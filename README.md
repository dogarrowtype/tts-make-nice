# tts-make-nice

A quality-of-life proxy that sits between a client and any OpenAI-compatible
TTS API server (tested against [mlx-audio](https://github.com/Blaizzy/mlx-audio)).
It cleans up text before synthesis, splits long input into natural segments,
stitches the audio back together with natural pausing, and re-encodes to the
format the client asked for — without the client needing to know any of it
happened.

```
Client → tts_proxy (normalize → split → tag → per-segment synth → trim/stitch → encode) → upstream /v1/audio/speech
```

It also ships a browser **playground** for experimenting, served by the proxy itself.

## Features

### Text handling (all granularly toggleable)
- Fancy quotes → ASCII, en-dash → hyphen, `…` → `...`, bullet stripping,
  em-dash padding for natural pauses, emoji removal, ALL-CAPS lowercasing
- **Pipe splitting** — `|` in the text forces a segment boundary
- **Manual pauses** — `{pause 1500}` injects exactly 1500 ms of silence at that
  spot (hard split, replaces the random inter-segment pause there)
- **Semantic segmentation** — long text chunked at ~380 chars on sentence boundaries
- **Emotion tags** — a configurable string (e.g. `[ upbeat, friendly ]`) prepended
  to every segment, per-model prompt prefixing without editing the text
- `POST /v1/preview` shows exactly what will be sent upstream, before you spend
  generation time

### Voice cloning (mlx-audio `ref_audio`/`ref_text`)
- Named **presets** in the config (`voice_cloning.presets`) with startup validation
- Per-request control field: `"voice_clone": "preset-name"`, `true` (server default),
  or `false` (opt-out). Never forwarded upstream
- Playground UI with enable checkbox + preset picker, driven by `GET /v1/config`

### Audio handling
- Per-segment synthesis with retry + silence fallback; **4xx upstream errors fail
  fast** and are propagated to the client (no pointless retries)
- Per-segment leading/trailing silence trimming
- Randomized 300–500 ms pauses between segments
- Always requests lossless WAV upstream, then re-encodes to mp3/opus/flac/aac/wav
  via ffmpeg (graceful WAV fallback with logged ffmpeg errors)
- Opus path resamples to 48 kHz correctly (no pitch distortion)

### Compatibility
- OpenAI-compatible `POST /v1/audio/speech` (JSON or form bodies), forward/client
  wins precedence for unknown fields
- `GET /v1/models` and `GET /v1/audio/voices` passthrough, `/health`, CORS enabled
- Arbitrary upstream params via `api_defaults` (fill-if-absent) and
  `api_overrides` (always win) in the config file

## Setup

```bash
pip install -r requirements.txt
cp proxy_config.example.json proxy_config.json   # optional; edit to taste
python tts_proxy.py                              # auto-loads proxy_config.json if present
# or explicitly:
python tts_proxy.py --config proxy_config.json
# or CLI-only:
python tts_proxy.py --upstream http://localhost:8000 --port 9992 --model mlx-community/Spark-TTS-0.5B-8bit
```

Then open the playground at `http://127.0.0.1:9992/`.

Config precedence: **CLI flag > config file > built-in default**. The config file
is comment-tolerant JSON (`//` and `/* */` are stripped; string values are safe).

See `proxy_config.example.json` for the annotated full format, and
`MLX_AUDIO_GUIDE.md` for mlx-audio specifics (voice cloning paths, batching,
sampling-param resolution).

## Proxy control fields

Fields the proxy interprets and **never forwards** upstream:

| Field | Values | Effect |
|---|---|---|
| `voice_clone` | `"<preset>"` / `true` / `false` | Select a cloning preset, use server default, or opt out |
| `emotion_tags` | `"<tags>"` / `""` | Override the configured emotion tags for this request, or disable |

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/speech` | TTS with the full QoL pipeline (OpenAI-compatible) |
| `POST /v1/preview` | Normalized text + final segments (with tags), no synthesis |
| `GET /v1/models` | Upstream model list passthrough (fallback list) |
| `GET /v1/audio/voices?model=...` | Upstream voice list passthrough |
| `GET /v1/config` | Public proxy config for clients (model default, presets, emotion tags) |
| `GET /health` | Status + effective configuration |
| `GET /` , `/playground.html` | The playground page |

## Files

- `tts_proxy.py` — the proxy (single file)
- `playground.html` — no-scroll landscape playground, served by the proxy
- `proxy_config.example.json` — annotated full config format
- `requirements.txt` — Python dependencies
- `MLX_AUDIO_GUIDE.md` — using the proxy with mlx-audio, incl. voice cloning