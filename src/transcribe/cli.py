from pathlib import Path

import click

from .config import DEFAULT_LANGUAGE, MODEL_ALIASES
from .pipeline import run_pipeline


@click.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--speakers", "-s", type=int, default=None,
    help="Number of speakers (auto-detect if omitted)",
)
@click.option(
    "--names", "-n", type=str, default=None,
    help="Comma-separated speaker names (implies --speakers)",
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), default=None,
    help="Output file path (default: <input>.md)",
)
@click.option(
    "--model", "-m", type=str, default="large-v3-turbo",
    help="Whisper model alias or HF repo (default: large-v3-turbo)",
)
@click.option(
    "--language", "-l", type=str, default=DEFAULT_LANGUAGE,
    help=f"Language code (default: {DEFAULT_LANGUAGE})",
)
def cli(
    audio_file: Path,
    speakers: int | None,
    names: str | None,
    output: Path | None,
    model: str,
    language: str,
):
    """Transcribe and diarize an audio file locally on Apple Silicon."""
    resolved_model = MODEL_ALIASES.get(model, model)

    name_list = None
    if names:
        name_list = [n.strip() for n in names.split(",")]
        if speakers is None:
            speakers = len(name_list)
        elif speakers != len(name_list):
            raise click.BadParameter(
                f"--speakers={speakers} but --names has {len(name_list)} names"
            )

    result_path = run_pipeline(
        input_path=audio_file,
        output_path=output,
        speakers=speakers,
        names=name_list,
        model=resolved_model,
        language=language,
    )

    click.echo(click.style(f"\nDone! Output: {result_path}", fg="green", bold=True))
