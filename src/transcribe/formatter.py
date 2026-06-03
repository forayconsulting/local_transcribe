from pathlib import Path

from .config import PARAGRAPH_GAP_THRESHOLD
from .diarizer import DiarizedSegment


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_markdown(
    segments: list[DiarizedSegment],
    source_name: str,
    duration: float,
    speakers: list[str],
    model: str,
    language: str,
    removed_ranges: list[tuple[float, float]] | None = None,
) -> str:
    lines = [
        f"# Transcription: {Path(source_name).stem}",
        "",
        f"**Duration:** {format_timestamp(duration)}",
        f"**Source:** {source_name}",
        f"**Speakers:** {', '.join(speakers)}",
        f"**Model:** {model}",
        f"**Language:** {language}",
        "",
    ]

    if removed_ranges:
        lines.append(
            f"*Note: {len(removed_ranges)} sections of garbled/hallucinated audio "
            f"were omitted from the transcript.*"
        )
        lines.append("")

    lines.append("---")
    lines.append("")

    if not segments:
        lines.append("*No speech detected.*")
        return "\n".join(lines)

    current_speaker = None
    current_words = []
    paragraph_start = None
    last_end = 0.0

    def flush_paragraph():
        nonlocal current_words, paragraph_start
        if current_words and current_speaker and paragraph_start is not None:
            text = " ".join(current_words)
            ts = format_timestamp(paragraph_start)
            lines.append(f"**{current_speaker} [{ts}]:** {text}")
            lines.append("")
        current_words = []
        paragraph_start = None

    for seg in segments:
        if not seg.text.strip():
            continue

        speaker_changed = seg.speaker != current_speaker
        gap = seg.start - last_end if last_end > 0 else 0
        long_pause = gap > PARAGRAPH_GAP_THRESHOLD

        if speaker_changed or long_pause:
            flush_paragraph()
            current_speaker = seg.speaker
            paragraph_start = seg.start

        if paragraph_start is None:
            paragraph_start = seg.start

        current_words.append(seg.text.strip())
        last_end = seg.end

    flush_paragraph()

    if removed_ranges:
        for start, end in removed_ranges:
            ts_start = format_timestamp(start)
            ts_end = format_timestamp(end)
            insert_line = f"*[{ts_start}–{ts_end}: inaudible/garbled audio omitted]*"
            insert_pos = None
            for i, line in enumerate(lines):
                if line.startswith("**") and "[" in line:
                    try:
                        bracket = line.index("[") + 1
                        colon = line.index("]", bracket)
                        ts_str = line[bracket:colon]
                        parts = ts_str.split(":")
                        if len(parts) == 3:
                            t = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                        else:
                            t = int(parts[0]) * 60 + int(parts[1])
                        if t > start:
                            insert_pos = i
                            break
                    except (ValueError, IndexError):
                        continue
            if insert_pos is not None:
                lines.insert(insert_pos, insert_line)
                lines.insert(insert_pos + 1, "")

    return "\n".join(lines)


def write_output(content: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
