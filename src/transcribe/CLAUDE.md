# src/transcribe/

The entire pipeline lives here. Entry point is `cli.py` → `pipeline.py` which orchestrates the five stages in order.

## Contents
| File | Purpose | Key Exports |
|------|---------|-------------|
| `cli.py` | Click CLI definition, argument parsing, model alias resolution | `cli()` |
| `pipeline.py` | Orchestrator wiring all stages with progress reporting and temp file cleanup | `run_pipeline()` |
| `config.py` | All constants: model paths, thresholds, cache dirs | `DEFAULT_MODEL`, `MODEL_ALIASES`, `SPEECHBRAIN_MODEL`, `PARAGRAPH_GAP_THRESHOLD` |
| `preprocess.py` | ffmpeg wrapper converting any audio format to 16kHz mono WAV | `preprocess_audio()`, `get_audio_duration()` |
| `transcriber.py` | mlx-whisper integration with deterministic settings and word timestamps | `transcribe_audio()`, `Segment`, `TranscriptionResult` |
| `hallucination.py` | Detects and strips Whisper artifacts (repeated words, consecutive identical segments) | `clean_hallucinations()` |
| `diarizer.py` | Speaker embedding extraction, auto-detection of speaker count, spectral clustering, label assignment | `diarize()`, `DiarizedSegment` |
| `formatter.py` | Groups diarized segments into speaker-turn paragraphs and writes timestamped markdown | `format_markdown()`, `write_output()` |

## Pipeline Flow
`preprocess` → `transcriber` → `hallucination` → `diarizer` → `formatter`

## Relationships
- **Parent**: `src/` (just a package container)
- **Depends on**: `config.py` is imported by every other module for thresholds and model paths

## Conventions
- Dataclasses for all intermediate data: `Segment`, `TranscriptionResult`, `DiarizedSegment`
- `diarizer.py` loads audio via `soundfile` (not `torchaudio`) to avoid the `torchcodec` dependency
- Speaker names assigned by first-appearance order in the transcript for determinism
- The `progress` callback pattern lets `pipeline.py` inject `click.echo` without other modules depending on click

## Warnings
- `diarizer.py` is the most complex module. The speaker count auto-detection uses eigenvalue gap on the Laplacian of the cosine similarity matrix, validated by silhouette score sweep. If both methods disagree by more than 1, silhouette wins.
- Segments shorter than 0.5s get skipped during embedding extraction and inherit the nearest neighbor's speaker label.
- `hallucination.py` patterns were adapted from `jvl_whisper`. The re-transcription recovery step was dropped since we're running locally (just discard bad segments).
