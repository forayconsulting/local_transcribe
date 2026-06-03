from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WordInfo:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[WordInfo] = field(default_factory=list)
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    language: str
    duration: float


def transcribe_audio(
    wav_path: Path,
    model: str,
    language: str,
) -> TranscriptionResult:
    import mlx_whisper

    raw = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=model,
        language=language,
        word_timestamps=True,
        temperature=(0.0,),
        condition_on_previous_text=True,
        hallucination_silence_threshold=2.0,
        verbose=False,
    )

    segments = []
    for i, seg in enumerate(raw.get("segments", [])):
        words = []
        for w in seg.get("words", []):
            words.append(WordInfo(
                word=w.get("word", ""),
                start=w.get("start", 0.0),
                end=w.get("end", 0.0),
                probability=w.get("probability", 0.0),
            ))
        segments.append(Segment(
            id=i,
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            text=seg.get("text", "").strip(),
            words=words,
            no_speech_prob=seg.get("no_speech_prob", 0.0),
            avg_logprob=seg.get("avg_logprob", 0.0),
        ))

    duration = segments[-1].end if segments else 0.0
    return TranscriptionResult(
        segments=segments,
        language=raw.get("language", language),
        duration=duration,
    )
