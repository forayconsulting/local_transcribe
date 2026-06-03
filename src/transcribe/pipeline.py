from pathlib import Path

import click

from .diarizer import diarize
from .formatter import format_markdown, format_timestamp, write_output
from .hallucination import clean_hallucinations
from .preprocess import get_audio_duration, preprocess_audio
from .transcriber import transcribe_audio


def run_pipeline(
    input_path: Path,
    output_path: Path | None,
    speakers: int | None,
    names: list[str] | None,
    model: str,
    language: str,
) -> Path:
    if not input_path.exists():
        raise click.ClickException(f"File not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".md")

    def progress(msg: str):
        click.echo(click.style(f"  {msg}", dim=True))

    click.echo(click.style("Preprocessing audio...", bold=True))
    wav_path = preprocess_audio(input_path)

    try:
        duration = get_audio_duration(wav_path)
        progress(f"Duration: {format_timestamp(duration)}")

        click.echo(click.style("Transcribing...", bold=True))
        result = transcribe_audio(wav_path, model, language)
        progress(f"{len(result.segments)} segments transcribed")

        click.echo(click.style("Cleaning hallucinations...", bold=True))
        clean_segments, removed_ranges = clean_hallucinations(result.segments)
        if removed_ranges:
            progress(f"Removed {len(removed_ranges)} hallucinated sections")
        else:
            progress("No hallucinations detected")

        click.echo(click.style("Diarizing speakers...", bold=True))
        diarized = diarize(
            wav_path, clean_segments,
            n_speakers=speakers, names=names, progress=progress,
        )

        speaker_set = list(dict.fromkeys(seg.speaker for seg in diarized))

        click.echo(click.style("Formatting output...", bold=True))
        md = format_markdown(
            segments=diarized,
            source_name=input_path.name,
            duration=result.duration,
            speakers=speaker_set,
            model=model,
            language=language,
            removed_ranges=removed_ranges if removed_ranges else None,
        )

        result_path = write_output(md, output_path)
        return result_path

    finally:
        wav_path.unlink(missing_ok=True)
