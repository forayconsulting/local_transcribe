from pathlib import Path

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

MODEL_ALIASES = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
}

DEFAULT_LANGUAGE = "en"
SAMPLE_RATE = 16000

SPEECHBRAIN_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SPEECHBRAIN_CACHE = Path.home() / ".cache" / "praecipio_local_transcribe" / "speechbrain"
EMBEDDING_DIM = 192

SEGMENT_MIN_DURATION = 0.5
PARAGRAPH_GAP_THRESHOLD = 2.0
MAX_SPEAKERS_SWEEP = 10
MIN_SPEAKERS = 2

HALLUCINATION_WORD_REPEAT_THRESHOLD = 0.6
