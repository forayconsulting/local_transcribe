import re

from .transcriber import Segment


def is_hallucination(seg: Segment) -> bool:
    text = seg.text.strip()

    if seg.no_speech_prob > 0.8 and len(text) < 5:
        return True

    words = text.split()
    if len(words) >= 6:
        unique = set(w.strip(",. ").lower() for w in words)
        if len(unique) <= 2:
            return True

    stripped = re.sub(r"[,\s]+", " ", text).strip()
    tokens = stripped.split()
    if len(tokens) >= 4:
        if len(set(tokens)) == 1:
            return True

    return False


def detect_repeated_segments(segments: list[Segment], min_repeats: int = 3) -> set[int]:
    if len(segments) < min_repeats:
        return set()

    bad = set()
    i = 0
    while i < len(segments):
        text_i = segments[i].text.strip()
        if not text_i:
            i += 1
            continue

        run = [i]
        j = i + 1
        while j < len(segments):
            if segments[j].text.strip() == text_i:
                run.append(j)
                j += 1
            else:
                break

        if len(run) >= min_repeats:
            bad.update(run)

        i = j if j > i + 1 else i + 1

    return bad


def clean_hallucinations(
    segments: list[Segment],
) -> tuple[list[Segment], list[tuple[float, float]]]:
    bad_single = {i for i, seg in enumerate(segments) if is_hallucination(seg)}
    bad_repeated = detect_repeated_segments(segments)
    all_bad = bad_single | bad_repeated

    clean = []
    removed_ranges = []
    hall_start = None
    hall_end = None

    for i, seg in enumerate(segments):
        if i in all_bad:
            if hall_start is None:
                hall_start = seg.start
            hall_end = seg.end
        else:
            if hall_start is not None:
                removed_ranges.append((hall_start, hall_end))
                hall_start = None
                hall_end = None
            clean.append(seg)

    if hall_start is not None:
        removed_ranges.append((hall_start, hall_end))

    return clean, removed_ranges
