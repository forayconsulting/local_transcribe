from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from speechbrain.inference.speaker import EncoderClassifier

from .config import (
    EMBEDDING_DIM,
    MAX_SPEAKERS_SWEEP,
    MIN_SPEAKERS,
    SAMPLE_RATE,
    SEGMENT_MIN_DURATION,
    SPEECHBRAIN_CACHE,
    SPEECHBRAIN_MODEL,
)
from .transcriber import Segment, WordInfo


@dataclass
class DiarizedSegment:
    id: int
    start: float
    end: float
    text: str
    words: list[WordInfo] = field(default_factory=list)
    speaker: str = ""


def load_speaker_encoder() -> EncoderClassifier:
    SPEECHBRAIN_CACHE.mkdir(parents=True, exist_ok=True)
    return EncoderClassifier.from_hparams(
        source=SPEECHBRAIN_MODEL,
        savedir=str(SPEECHBRAIN_CACHE),
        run_opts={"device": "cpu"},
    )


def extract_embeddings(
    wav_path: Path,
    segments: list[Segment],
    encoder: EncoderClassifier,
) -> tuple[np.ndarray, list[bool]]:
    audio_np, sr = sf.read(str(wav_path), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=SAMPLE_RATE)
    waveform = torch.from_numpy(audio_np).unsqueeze(0)

    embeddings = np.zeros((len(segments), EMBEDDING_DIM), dtype=np.float32)
    valid = [False] * len(segments)

    for i, seg in enumerate(segments):
        duration = seg.end - seg.start
        if duration < SEGMENT_MIN_DURATION:
            continue

        start_sample = int(seg.start * SAMPLE_RATE)
        end_sample = int(seg.end * SAMPLE_RATE)
        chunk = waveform[:, start_sample:end_sample]

        if chunk.shape[1] < int(SEGMENT_MIN_DURATION * SAMPLE_RATE):
            continue

        with torch.no_grad():
            emb = encoder.encode_batch(chunk)
        embeddings[i] = emb.squeeze().numpy()
        valid[i] = True

    return embeddings, valid


def estimate_num_speakers(embeddings: np.ndarray, max_k: int = MAX_SPEAKERS_SWEEP) -> int:
    n = len(embeddings)
    max_k = min(max_k, n - 1)
    if max_k < MIN_SPEAKERS:
        return MIN_SPEAKERS

    sim = cosine_similarity(embeddings)
    sim = np.maximum(sim, 0)

    degree = np.diag(sim.sum(axis=1))
    laplacian = degree - sim
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.sort(eigenvalues)

    gaps = np.diff(eigenvalues[: max_k + 1])
    eigen_k = int(np.argmax(gaps) + 1)
    eigen_k = max(MIN_SPEAKERS, min(eigen_k, max_k))

    best_k, best_score = MIN_SPEAKERS, -1.0
    for k in range(MIN_SPEAKERS, max_k + 1):
        try:
            agg = AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average",
            )
            labels = agg.fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels, metric="cosine")
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue

    if abs(eigen_k - best_k) > 1:
        return best_k
    return eigen_k


def cluster_speakers(
    embeddings: np.ndarray,
    valid_mask: list[bool],
    n_speakers: int | None = None,
) -> np.ndarray:
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    valid_embeddings = embeddings[valid_indices]

    if len(valid_embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)

    if n_speakers is None:
        n_speakers = estimate_num_speakers(valid_embeddings)

    n_speakers = min(n_speakers, len(valid_embeddings))

    sim = cosine_similarity(valid_embeddings)
    sim = np.maximum(sim, 0)
    np.fill_diagonal(sim, 1.0)

    sc = SpectralClustering(
        n_clusters=n_speakers,
        affinity="precomputed",
        random_state=42,
    )
    valid_labels = sc.fit_predict(sim)

    labels = np.full(len(embeddings), -1, dtype=int)
    for idx, vi in enumerate(valid_indices):
        labels[vi] = valid_labels[idx]

    for i in range(len(labels)):
        if labels[i] == -1:
            nearest = -1
            for offset in range(1, len(labels)):
                if i - offset >= 0 and labels[i - offset] != -1:
                    nearest = labels[i - offset]
                    break
                if i + offset < len(labels) and labels[i + offset] != -1:
                    nearest = labels[i + offset]
                    break
            labels[i] = nearest if nearest != -1 else 0

    return labels


def assign_speaker_names(
    labels: np.ndarray,
    names: list[str] | None = None,
) -> list[str]:
    order = []
    seen = set()
    for lbl in labels:
        if lbl not in seen:
            order.append(lbl)
            seen.add(lbl)

    label_to_name = {}
    for i, lbl in enumerate(order):
        if names and i < len(names):
            label_to_name[lbl] = names[i]
        else:
            label_to_name[lbl] = f"Speaker {i + 1}"

    return [label_to_name[lbl] for lbl in labels]


def diarize(
    wav_path: Path,
    segments: list[Segment],
    n_speakers: int | None = None,
    names: list[str] | None = None,
    progress=None,
) -> list[DiarizedSegment]:
    if progress:
        progress("Loading speaker encoder...")
    encoder = load_speaker_encoder()

    if progress:
        progress("Extracting speaker embeddings...")
    embeddings, valid_mask = extract_embeddings(wav_path, segments, encoder)

    if progress:
        n_valid = sum(valid_mask)
        progress(f"Extracted {n_valid} embeddings from {len(segments)} segments")

    if progress:
        progress("Clustering speakers...")
    labels = cluster_speakers(embeddings, valid_mask, n_speakers)

    n_detected = len(set(labels))
    if progress:
        progress(f"Detected {n_detected} speakers")

    speaker_names = assign_speaker_names(labels, names)

    result = []
    for i, seg in enumerate(segments):
        result.append(DiarizedSegment(
            id=seg.id,
            start=seg.start,
            end=seg.end,
            text=seg.text,
            words=seg.words,
            speaker=speaker_names[i],
        ))

    return result
