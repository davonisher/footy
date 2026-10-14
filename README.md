# Footy

**Handy’s “foot”** — a small macOS app for **batch-transcribing old audio and video files** locally.

[Handy](https://github.com/cjpais/Handy) is excellent for live dictation. What I personally missed was an easy way to drop older recordings (Voice Memos, meetings, archives) onto something in the same spirit and get a transcript back. So I built Footy for my own use: same local-first idea, focused on files, an archive, and shortcuts into folders like Dictafoon / Handy recordings.

> Handy = live. Footy = files.

This is a personal side project. It is **not** affiliated with or endorsed by Handy.

## Credits & inspiration

All credit for the inspiration goes to **[CJ Pais](https://github.com/cjpais)** and the [Handy](https://github.com/cjpais/Handy) project ([handy.computer](https://handy.computer)).

Footy’s name, UI cues, and local-transcription mindset are inspired by Handy. If you want live speech-to-text into any text field, use Handy. Footy only fills the “transcribe files I already have” gap I ran into myself.

Also thanks to the ecosystems Footy leans on:

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) / Whisper models
- [FluidAudio](https://github.com/FluidInference/FluidAudio) (Parakeet on Apple Neural Engine + offline diarization)
- [Ollama](https://ollama.com) / Qwen (optional post-processing)

## Features

- Native macOS window (Cocoa + WKWebView)
- Drop audio/video → local transcript
- Model choice: FluidAudio Parakeet (ANE), Whisper Large V3 Turbo, Parakeet ggml
- Optional timestamps, speaker labels, Qwen post-process via Ollama
- Simple archive + open Dictafoon / Handy / transcripts folders
- Output: `~/Documents/Transcripts/`

## Requirements (macOS)

- Apple Silicon recommended (especially for FluidAudio)
- Python 3.11+
- `ffmpeg` and (for Whisper) `whisper-cli` on your `PATH`
- Optional: Handy installed (reuses its Whisper model), FluidAudio CLI, Ollama + `qwen3.8`

## Run from source

```bash
cd batch-transcriber   # or: footy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5055` or build the native shell:

```bash
swiftc -O -framework Cocoa -framework WebKit native/main.swift -o Footy
./Footy
```

The Swift app expects the project at `~/Code/batch-transcriber` by default (edit `native/main.swift` if your path differs).

## Disclaimer

Built for personal use. Expect rough edges. Not a product, not a Handy fork — just a thank-you-shaped companion for file batch jobs.
