import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

from .config import SAMPLE_RATE


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")


def preprocess_audio(input_path: Path) -> Path:
    check_ffmpeg()
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = Path(tmp.name)

    result = subprocess.run(
        [
            "ffmpeg", "-i", str(input_path),
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav",
            "-y", str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-500:]}")

    return out_path


def get_audio_duration(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return info.duration
