# local_transcribe

A CLI tool that transcribes and diarizes audio files entirely on-device. No cloud APIs, no data leaving your machine. Built for Apple Silicon.

It takes any audio file and produces a timestamped, speaker-labeled markdown transcript. The pipeline detects how many speakers are in the recording, identifies who said what, and cleans up common transcription artifacts automatically.

## How it works

The tool chains three local models together in a deterministic pipeline.

First, `mlx-whisper` (running natively on Apple Silicon via MLX) transcribes the audio with word-level timestamps. This is fast. A 90-minute recording finishes in about 4-5 minutes on an M-series Mac.

Then, `speechbrain`'s ECAPA-TDNN model extracts a voice embedding for each transcript segment. These 192-dimensional vectors capture the vocal characteristics of whoever is speaking. The embeddings get clustered using spectral clustering to figure out which segments belong to the same person. If you don't tell it how many speakers to expect, it estimates the count automatically using an eigenvalue gap heuristic validated against silhouette scores.

Finally, a cleanup pass catches Whisper's hallucination patterns (the "thank you thank you thank you" loops, repeated phrases during silence, etc.) and strips them from the output.

Same input always produces the same output. Temperature is pinned to 0.0 for greedy decoding and the clustering uses a fixed random seed.

## Install

You need Python 3.11-3.13, `uv`, and `ffmpeg`.

```bash
brew install ffmpeg  # if you don't have it

git clone https://github.com/forayconsulting/local_transcribe.git
cd local_transcribe
uv sync
```

The first run will download the Whisper model (~1.5GB) and the speaker embedding model (~100MB). These are cached locally after that.

## Usage

The simplest invocation auto-detects everything:

```bash
uv run transcribe recording.m4a
```

This writes `recording.md` next to the input file.

If you know who's in the recording, name them:

```bash
uv run transcribe meeting.m4a --names "Alice,Bob,Carol"
```

To specify the number of speakers without naming them:

```bash
uv run transcribe interview.wav --speakers 2
```

You can also set the output path, whisper model, and language:

```bash
uv run transcribe call.m4a -o ~/Desktop/notes.md --model large-v3 --language es
```

Available model aliases: `large-v3-turbo` (default, fastest), `large-v3` (highest quality), `small`, `base`.

## Output format

The tool produces a markdown file that looks like this:

```markdown
# Transcription: meeting

**Duration:** 1:23:45
**Source:** meeting.m4a
**Speakers:** Alice, Bob
**Model:** mlx-community/whisper-large-v3-turbo
**Language:** en

---

**Alice [00:02]:** So I was thinking we should revisit the timeline.

**Bob [00:08]:** Yeah, the original estimate was way too aggressive.

**Alice [00:14]:** Agreed. What if we pushed the demo to next Friday?
```

Paragraphs break on speaker changes or when there's a pause longer than 2 seconds within the same speaker's turn.

## Limitations

This only runs on Apple Silicon Macs. The transcription engine (`mlx-whisper`) requires the MLX framework which is specific to Apple's hardware.

Speaker diarization works best when speakers have distinct voices. Two people with very similar vocal characteristics will be harder to separate. The embedding model runs on CPU (not the Neural Engine) so the diarization step is slower than the transcription step for very long recordings.

Auto-detection of speaker count is good but not perfect. If you know how many people are in the recording, passing `--speakers N` will give you better results than letting it guess.

## License

MIT
