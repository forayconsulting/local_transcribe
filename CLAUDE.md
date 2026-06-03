# local_transcribe

Local audio transcription + speaker diarization CLI for Apple Silicon. Chains mlx-whisper, speechbrain ECAPA-TDNN embeddings, and spectral clustering into a deterministic pipeline that runs entirely on-device.

## Tech Stack
- **Python 3.13** managed by `uv`
- **mlx-whisper**: Whisper transcription via Apple MLX framework
- **speechbrain**: ECAPA-TDNN speaker embeddings (192-dim, no HF auth required)
- **scikit-learn**: Spectral clustering for speaker assignment
- **click**: CLI framework
- **ffmpeg**: System dependency for audio format conversion

## Architecture Overview

Single-package CLI tool. All source lives in `src/transcribe/`. The pipeline flows: audio file → ffmpeg preprocessing → mlx-whisper transcription → hallucination cleanup → speaker embedding extraction → clustering → markdown output.

## Directory Map
| Directory | Purpose | Key Entry Points |
|-----------|---------|-----------------|
| `src/transcribe/` | All pipeline modules and CLI | `cli.py`, `pipeline.py` |

## Development
- **Install**: `uv sync`
- **Run**: `uv run transcribe <audio_file>`
- **Test**: `uv run transcribe <file> --names "A,B"` (no test suite yet)

## Conventions
- Determinism enforced everywhere: `temperature=(0.0,)` for Whisper, `random_state=42` for clustering
- Model weights cached at `~/.cache/praecipio_local_transcribe/` (speechbrain) and HuggingFace hub default (mlx-whisper)
- All tunable thresholds live in `src/transcribe/config.py`

## Warnings
- Apple Silicon only (MLX framework requirement)
- First run downloads ~1.6GB of model weights
- `torchaudio.load()` requires `torchcodec` in recent versions, so audio loading uses `soundfile` instead
