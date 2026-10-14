#!/usr/bin/env python3
"""Footy — local batch transcription for Mac."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask.json.provider import DefaultJSONProvider

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
WORK_DIR = APP_DIR / "work"
OUTPUT_DIR = Path.home() / "Documents" / "Transcripts"
VOICEMEMOS_DIR = (
    Path.home()
    / "Library"
    / "Group Containers"
    / "group.com.apple.VoiceMemos.shared"
    / "Recordings"
)
HANDY_DIR = (
    Path.home() / "Library" / "Application Support" / "com.pais.handy" / "recordings"
)
HANDY_MODELS = (
    Path.home() / "Library" / "Application Support" / "com.pais.handy" / "models"
)
EG2_APP = Path("/Applications/Embedding Gemma Transcribe.app")

WHISPER_MODEL = HANDY_MODELS / "ggml-large-v3-turbo.bin"
PARAKEET_MODEL = HANDY_MODELS / "ggml-parakeet-tdt-0.6b-v3-q4_0.bin"

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
WHISPER = shutil.which("whisper-cli") or "/opt/homebrew/bin/whisper-cli"
PARAKEET = shutil.which("parakeet-cli") or "/opt/homebrew/bin/parakeet-cli"
FLUIDAUDIO = (
    shutil.which("fluidaudiocli")
    or shutil.which("fluidaudio")
    or str(Path.home() / ".local/bin/fluidaudiocli")
)
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

CHUNK_TARGET_MB = 25
CHUNK_SECONDS = 10 * 60

ALLOWED_EXT = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
}

POST_PROCESS_PROMPT = """Clean this transcript:
1. Fix spelling, capitalization, and punctuation
2. Convert number words to digits where natural
3. Remove filler words (um, uh, euh, als filler)
4. Keep the original language
5. Preserve meaning and word order; do not paraphrase

Return only the cleaned transcript.

Transcript:
{text}
"""

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


class SafeJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("allow_nan", False)
        return super().dumps(obj, **kwargs)


app = Flask(__name__)
app.json = SafeJSONProvider(app)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _clean_text(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text or "")


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\- ]+", "", stem, flags=re.UNICODE).strip()
    return stem[:80] or "transcript"


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _update_job(job_id: str, **kwargs) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.update(kwargs)


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _run_with_env(cmd: list[str], env: dict) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            (proc.stderr or proc.stdout or f"Command failed: {' '.join(cmd)}")[-2000:]
        )
    return proc


def _probe_duration_seconds(path: Path) -> float:
    cmd = [
        FFPROBE,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        return float(_run(cmd).stdout.strip())
    except Exception:
        return 0.0


def _convert_to_wav(src: Path, dst: Path, stereo: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG,
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "2" if stereo else "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    _run(cmd)


def _split_wav(wav_path: Path, chunk_dir: Path, max_seconds: int) -> list[Path]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pattern = chunk_dir / "chunk_%03d.wav"
    _run(
        [
            FFMPEG,
            "-y",
            "-i",
            str(wav_path),
            "-f",
            "segment",
            "-segment_time",
            str(max_seconds),
            "-c",
            "copy",
            str(pattern),
        ]
    )
    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    return chunks if chunks else [wav_path]


def _parse_srt_timestamp(ts: str) -> float:
    # 00:01:02,345
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) != 3:
        return 0.0
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


def _shift_srt(srt_text: str, offset_seconds: float) -> str:
    if not srt_text.strip() or offset_seconds <= 0:
        return srt_text

    def shift_line(line: str) -> str:
        if "-->" not in line:
            return line
        left, right = [p.strip() for p in line.split("-->")]
        return f"{_seconds_to_srt(_parse_srt_timestamp(left) + offset_seconds)} --> {_seconds_to_srt(_parse_srt_timestamp(right) + offset_seconds)}"

    return "\n".join(shift_line(line) for line in srt_text.splitlines())


def _seconds_to_srt(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    td = timedelta(seconds=seconds)
    total = int(td.total_seconds())
    ms = int(round((seconds - total) * 1000))
    if ms == 1000:
        total += 1
        ms = 0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _srt_to_timestamped_text(srt_text: str) -> str:
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    lines: list[str] = []
    for block in blocks:
        parts = [p for p in block.splitlines() if p.strip()]
        if len(parts) < 2:
            continue
        # index, timing, text...
        timing = next((p for p in parts if "-->" in p), None)
        if not timing:
            continue
        start = timing.split("-->")[0].strip()
        sec = _parse_srt_timestamp(start)
        text = " ".join(p.strip() for p in parts if "-->" not in p and not p.strip().isdigit())
        text = text.strip()
        if text:
            lines.append(f"[{_fmt_seconds(sec)}] {text}")
    return "\n".join(lines)


def _fluidaudio_bin() -> str | None:
    path = Path(FLUIDAUDIO)
    return str(path) if path.is_file() and os.access(path, os.X_OK) else None


def _available_models() -> list[dict]:
    models = []
    if _fluidaudio_bin():
        models.append(
            {
                "id": "fluidaudio",
                "label": "FluidAudio Parakeet (ANE)",
                "available": True,
                "speakers": True,
            }
        )
    if WHISPER_MODEL.exists() and Path(WHISPER).exists():
        models.append(
            {
                "id": "whisper-turbo",
                "label": "Whisper Large V3 Turbo",
                "available": True,
                "speakers": "stereo-only",
            }
        )
    if Path(PARAKEET).exists():
        models.append(
            {
                "id": "parakeet",
                "label": "Parakeet TDT 0.6B v3 (ggml)",
                "available": PARAKEET_MODEL.exists(),
                "speakers": False,
            }
        )
    return models


def _pick_ollama_model() -> str | None:
    preferred = [
        "qwen3.8:latest",
        "hf.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M",
        "clef-flash:9b",
        "clef:27b",
        "tev1:4b",
    ]
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as resp:
            data = json.loads(resp.read().decode())
        names = {m.get("name") for m in data.get("models", [])}
        for name in preferred:
            if name in names:
                return name
        return next(iter(names), None)
    except Exception:
        return None


def _post_process_text(text: str, model: str | None = None) -> str:
    model = model or _pick_ollama_model()
    if not model or not text.strip():
        return text
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0.1},
        "messages": [
            {
                "role": "user",
                "content": POST_PROCESS_PROMPT.format(text=text.strip()),
            }
        ],
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    out = (data.get("message") or {}).get("content") or ""
    return _clean_text(out).strip() or text


def _transcribe_whisper(
    wav_path: Path,
    out_prefix: Path,
    language: str,
    diarize: bool,
) -> tuple[str, str]:
    cmd = [
        WHISPER,
        "-m",
        str(WHISPER_MODEL),
        "-f",
        str(wav_path),
        "-otxt",
        "-osrt",
        "-oj",
        "-of",
        str(out_prefix),
        "-np",
        "-l",
        language or "auto",
    ]
    if diarize:
        # Stereo channel diarization only (not true multi-speaker clustering).
        cmd.append("-di")
    _run_with_env(cmd, os.environ.copy())
    txt = out_prefix.with_suffix(".txt")
    srt = out_prefix.with_suffix(".srt")
    text = _clean_text(txt.read_text(encoding="utf-8", errors="replace")).strip() if txt.exists() else ""
    srt_text = srt.read_text(encoding="utf-8", errors="replace") if srt.exists() else ""
    return text, srt_text


def _transcribe_parakeet(wav_path: Path, out_prefix: Path) -> tuple[str, str]:
    if not PARAKEET_MODEL.exists():
        raise RuntimeError(
            f"Parakeet-model ontbreekt: {PARAKEET_MODEL}. Download ggml-parakeet-tdt-0.6b-v3-q4_0.bin."
        )
    cmd = [
        PARAKEET,
        "-m",
        str(PARAKEET_MODEL),
        "-f",
        str(wav_path),
        "-otxt",
        "-of",
        str(out_prefix),
        "-np",
        "-ps",
    ]
    proc = _run_with_env(cmd, os.environ.copy())
    txt = out_prefix.with_suffix(".txt")
    text = _clean_text(txt.read_text(encoding="utf-8", errors="replace")).strip() if txt.exists() else ""
    # Parakeet-cli has limited SRT; synthesize timestamps from printed segments if possible.
    srt_text = _segments_stdout_to_srt(proc.stdout or proc.stderr or "", text)
    return text, srt_text


def _word_timings_to_srt(words: list[dict]) -> str:
    if not words:
        return ""
    # Group into ~8s cue blocks
    blocks: list[tuple[float, float, list[str]]] = []
    cur_words: list[str] = []
    start = float(words[0].get("startTime", 0))
    end = start
    for w in words:
        w_start = float(w.get("startTime", end))
        w_end = float(w.get("endTime", w_start))
        token = str(w.get("word", "")).strip()
        if not token:
            continue
        if cur_words and (w_start - start) > 8.0:
            blocks.append((start, end, cur_words))
            cur_words = []
            start = w_start
        if not cur_words:
            start = w_start
        cur_words.append(token)
        end = w_end
    if cur_words:
        blocks.append((start, end, cur_words))
    lines = []
    for i, (s, e, toks) in enumerate(blocks, 1):
        lines.append(
            f"{i}\n{_seconds_to_srt(s)} --> {_seconds_to_srt(e)}\n{' '.join(toks)}\n"
        )
    return "\n".join(lines)


def _speaker_for_time(t: float, segments: list[dict]) -> str:
    for seg in segments:
        s = float(seg.get("startTimeSeconds", 0))
        e = float(seg.get("endTimeSeconds", 0))
        if s <= t <= e:
            return str(seg.get("speakerId") or "S?")
    # nearest segment
    if not segments:
        return "S?"
    best = min(
        segments,
        key=lambda seg: min(
            abs(t - float(seg.get("startTimeSeconds", 0))),
            abs(t - float(seg.get("endTimeSeconds", 0))),
        ),
    )
    return str(best.get("speakerId") or "S?")


def _merge_words_with_speakers(words: list[dict], diar_segments: list[dict]) -> str:
    if not words:
        return ""
    lines: list[str] = []
    cur_speaker = None
    cur_start = None
    buf: list[str] = []
    last_end = 0.0

    def flush():
        nonlocal buf, cur_speaker, cur_start
        if not buf:
            return
        stamp = _fmt_seconds(cur_start or 0)
        spk = cur_speaker or "S?"
        lines.append(f"[{stamp}] {spk}: {' '.join(buf)}")
        buf = []

    for w in words:
        token = str(w.get("word", "")).strip()
        if not token:
            continue
        start = float(w.get("startTime", last_end))
        end = float(w.get("endTime", start))
        mid = (start + end) / 2
        spk = _speaker_for_time(mid, diar_segments)
        if cur_speaker is None:
            cur_speaker = spk
            cur_start = start
        if spk != cur_speaker or (start - last_end) > 2.5:
            flush()
            cur_speaker = spk
            cur_start = start
        buf.append(token)
        last_end = end
    flush()
    return "\n".join(lines)


def _transcribe_fluidaudio(
    wav_path: Path,
    out_prefix: Path,
    language: str,
    speakers: bool,
) -> tuple[str, str, str | None]:
    """Returns (display_text, srt_text, speaker_note)."""
    bin_path = _fluidaudio_bin()
    if not bin_path:
        raise RuntimeError("fluidaudiocli niet gevonden (~/.local/bin/fluidaudiocli)")

    json_path = out_prefix.with_suffix(".fa.json")
    cmd = [
        bin_path,
        "transcribe",
        str(wav_path),
        "--output-json",
        str(json_path),
        "--metadata",
        "--word-timestamps",
        "--model-version",
        "v3",
    ]
    if language and language != "auto":
        cmd.extend(["--language", language])

    env = os.environ.copy()
    env["PATH"] = f"{Path(bin_path).parent}:{env.get('PATH', '')}"
    _run_with_env(cmd, env)

    if not json_path.exists():
        raise RuntimeError("FluidAudio gaf geen JSON-output")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    text = _clean_text(data.get("text") or "").strip()
    words = data.get("wordTimings") or []
    srt_text = _word_timings_to_srt(words)
    note = None

    if speakers:
        diar_path = out_prefix.with_suffix(".diar.json")
        diar_cmd = [
            bin_path,
            "process",
            str(wav_path),
            "--mode",
            "offline",
            "--output",
            str(diar_path),
        ]
        _run_with_env(diar_cmd, env)
        if diar_path.exists():
            diar = json.loads(diar_path.read_text(encoding="utf-8"))
            segs = diar.get("segments") or []
            # Drop huge embeddings from memory path; merge only timing/speaker
            slim = [
                {
                    "startTimeSeconds": s.get("startTimeSeconds"),
                    "endTimeSeconds": s.get("endTimeSeconds"),
                    "speakerId": s.get("speakerId"),
                }
                for s in segs
            ]
            merged = _merge_words_with_speakers(words, slim)
            if merged:
                text = merged
            note = f"FluidAudio diarization · {diar.get('speakerCount', '?')} speaker(s)"
        else:
            note = "FluidAudio diarization gaf geen output; alleen transcriptie."

    return text, srt_text, note


def _segments_stdout_to_srt(stdout: str, fallback_text: str) -> str:
    """Best-effort: parse 't0 -> t1: text' style lines from -ps output."""
    lines = []
    idx = 1
    for raw in stdout.splitlines():
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:->|-|–)\s*(\d+(?:\.\d+)?)\s*[:|]\s*(.+)$",
            raw.strip(),
        )
        if not m:
            continue
        start, end, text = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        if not text:
            continue
        lines.append(
            f"{idx}\n{_seconds_to_srt(start)} --> {_seconds_to_srt(end)}\n{text}\n"
        )
        idx += 1
    if lines:
        return "\n".join(lines)
    if fallback_text.strip():
        # Single block without real timings
        return f"1\n00:00:00,000 --> 00:00:01,000\n{fallback_text.strip()}\n"
    return ""


def _process_job(
    job_id: str,
    src_path: Path,
    original_name: str,
    language: str,
    model_id: str,
    timestamps: bool,
    post_process: bool,
    speakers: bool,
) -> None:
    work = WORK_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        _update_job(
            job_id,
            status="converting",
            progress=5,
            message="Audio omzetten…",
            elapsed_seconds=0,
        )
        # Stereo only when whisper + speakers (channel diarize)
        want_stereo = model_id == "whisper-turbo" and speakers
        wav_path = work / "audio.wav"
        _convert_to_wav(src_path, wav_path, stereo=want_stereo)

        size_mb = wav_path.stat().st_size / (1024 * 1024)
        duration = _probe_duration_seconds(wav_path)
        # FluidAudio handles long-form natively; avoid artificial chunking.
        needs_split = (
            model_id != "fluidaudio"
            and (size_mb > CHUNK_TARGET_MB or duration > CHUNK_SECONDS)
        )

        if needs_split:
            _update_job(
                job_id,
                status="splitting",
                progress=15,
                message=f"Groot bestand ({size_mb:.0f} MB) — splitsen…",
                elapsed_seconds=round(time.time() - t0, 1),
            )
            chunks = _split_wav(wav_path, work / "chunks", CHUNK_SECONDS)
        else:
            chunks = [wav_path]

        texts: list[str] = []
        srt_parts: list[str] = []
        total = len(chunks)
        speaker_note = None
        fluidaudio_speaker_text = None

        for i, chunk in enumerate(chunks):
            pct = 20 + int(60 * (i / max(total, 1)))
            _update_job(
                job_id,
                status="transcribing",
                progress=pct,
                message=f"Transcriberen ({model_id}) {i + 1}/{total}…",
                elapsed_seconds=round(time.time() - t0, 1),
            )
            if speakers and model_id == "fluidaudio":
                _update_job(
                    job_id,
                    message=f"FluidAudio + speakers {i + 1}/{total}…",
                    elapsed_seconds=round(time.time() - t0, 1),
                )
            prefix = work / f"out_{i:03d}"
            if model_id == "fluidaudio":
                text, srt_text, note = _transcribe_fluidaudio(
                    chunk, prefix, language, speakers=speakers
                )
                if note:
                    speaker_note = note
                if speakers and text:
                    fluidaudio_speaker_text = text
            elif model_id == "parakeet":
                text, srt_text = _transcribe_parakeet(chunk, prefix)
                if speakers:
                    speaker_note = (
                        "Speaker labels: ggml-Parakeet heeft geen diarization. "
                        "Kies FluidAudio Parakeet (ANE) + Speakers."
                    )
            else:
                text, srt_text = _transcribe_whisper(
                    chunk, prefix, language, diarize=speakers
                )
                if speakers:
                    speaker_note = (
                        "Speaker mode: Whisper stereo-diarize (links/rechts kanaal), "
                        "geen echte multi-speaker clustering. "
                        "Voor echte speakers: FluidAudio Parakeet (ANE)."
                    )
            if text:
                texts.append(text)
            if srt_text:
                offset = i * CHUNK_SECONDS if needs_split else 0
                srt_parts.append(_shift_srt(srt_text, offset))

        raw_text = _clean_text("\n\n".join(texts).strip())
        full_srt = _clean_text("\n".join(srt_parts).strip())
        # FluidAudio speaker merge already includes timestamps + speaker labels.
        if fluidaudio_speaker_text:
            display_text = fluidaudio_speaker_text
        elif timestamps and full_srt:
            display_text = _srt_to_timestamped_text(full_srt)
        else:
            display_text = raw_text

        post_text = None
        if post_process:
            _update_job(
                job_id,
                status="postprocess",
                progress=88,
                message="Post-processen via Ollama…",
                elapsed_seconds=round(time.time() - t0, 1),
            )
            try:
                post_text = _post_process_text(raw_text)
            except Exception as exc:
                post_text = None
                speaker_note = (speaker_note or "") + f" Post-process mislukt: {exc}"

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = _now()
        stem = _safe_stem(original_name)
        base = OUTPUT_DIR / f"{stamp}_{stem}"
        txt_path = base.with_suffix(".txt")
        srt_path = base.with_suffix(".srt")
        meta_path = base.with_name(base.name + "_meta.json")

        final_body = post_text or display_text
        if speaker_note and speakers:
            final_body = final_body + "\n\n—\n" + speaker_note
        txt_path.write_text(_clean_text(final_body.strip() + "\n"), encoding="utf-8")
        if full_srt:
            srt_path.write_text(full_srt + "\n", encoding="utf-8")

        elapsed = round(time.time() - t0, 2)
        model_path = {
            "fluidaudio": _fluidaudio_bin() or "fluidaudiocli",
            "parakeet": str(PARAKEET_MODEL),
            "whisper-turbo": str(WHISPER_MODEL),
        }.get(model_id, model_id)
        meta = {
            "source": original_name,
            "created_at": stamp,
            "model": model_id,
            "model_path": model_path,
            "language": language or "auto",
            "timestamps": timestamps,
            "post_process": bool(post_text),
            "speakers": speakers,
            "speaker_note": speaker_note,
            "chunks": total,
            "audio_duration_seconds": duration,
            "audio_duration_label": _fmt_seconds(duration),
            "elapsed_seconds": elapsed,
            "elapsed_label": _fmt_seconds(elapsed),
            "realtime_factor": round(duration / elapsed, 1) if elapsed > 0 else None,
            "wav_size_mb": round(size_mb, 1),
            "txt": str(txt_path),
            "srt": str(srt_path) if srt_path.exists() else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        _update_job(
            job_id,
            status="done",
            progress=100,
            message=f"Klaar in {_fmt_seconds(elapsed)} · audio {_fmt_seconds(duration)}",
            text=final_body + "\n",
            raw_text=raw_text,
            post_text=post_text,
            output_txt=str(txt_path),
            output_srt=str(srt_path) if srt_path.exists() else None,
            output_dir=str(OUTPUT_DIR),
            elapsed_seconds=elapsed,
            elapsed_label=_fmt_seconds(elapsed),
            audio_duration_seconds=duration,
            audio_duration_label=_fmt_seconds(duration),
            realtime_factor=meta["realtime_factor"],
            speaker_note=speaker_note,
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="error",
            progress=100,
            message=str(exc),
            elapsed_seconds=round(time.time() - t0, 1),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _list_transcripts(limit: int = 100) -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    for txt in OUTPUT_DIR.glob("*.txt"):
        try:
            stat = txt.stat()
            preview = _clean_text(txt.read_text(encoding="utf-8", errors="replace")).strip()
            preview = re.sub(r"\s+", " ", preview)
            if len(preview) > 160:
                preview = preview[:157] + "…"
            srt = txt.with_suffix(".srt")
            meta_file = txt.with_name(txt.stem + "_meta.json")
            meta = {}
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            items.append(
                {
                    "name": txt.name,
                    "stem": txt.stem,
                    "txt": txt.name,
                    "srt": srt.name if srt.exists() else None,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "mtime_label": datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "preview": preview,
                    "elapsed_label": meta.get("elapsed_label"),
                    "audio_duration_label": meta.get("audio_duration_label"),
                    "model": meta.get("model"),
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items[:limit]


@app.get("/")
def index():
    models = _available_models()
    tools_ok = Path(FFMPEG).exists() and Path(WHISPER).exists()
    return render_template(
        "index.html",
        model_ok=any(m["available"] for m in models),
        model_path=str(WHISPER_MODEL),
        tools_ok=tools_ok,
        output_dir=str(OUTPUT_DIR),
        voicememos_dir=str(VOICEMEMOS_DIR),
        voicememos_ok=VOICEMEMOS_DIR.is_dir(),
        handy_dir=str(HANDY_DIR),
        handy_ok=HANDY_DIR.is_dir(),
        models=models,
        parakeet_ok=PARAKEET_MODEL.exists(),
        ollama_model=_pick_ollama_model(),
    )


@app.get("/api/models")
def api_models():
    return jsonify(
        {
            "models": _available_models(),
            "ollama_model": _pick_ollama_model(),
            "speaker_support": {
                "fluidaudio": "offline diarization via FluidAudio process",
                "whisper-turbo": "stereo channel diarize only",
                "parakeet": "not available",
                "fluidaudio_bin": _fluidaudio_bin(),
            },
        }
    )


@app.post("/api/upload")
def upload():
    models = {m["id"]: m for m in _available_models()}
    model_id = (request.form.get("model") or "whisper-turbo").strip()
    if model_id not in models or not models[model_id]["available"]:
        return jsonify({"error": f"Model niet beschikbaar: {model_id}"}), 400
    if not Path(FFMPEG).exists():
        return jsonify({"error": "ffmpeg ontbreekt"}), 500

    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "Geen bestand gekozen"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Bestandstype niet ondersteund: {ext}"}), 400

    language = (request.form.get("language") or "auto").strip()
    timestamps = request.form.get("timestamps", "1") in {"1", "true", "on", "yes"}
    post_process = request.form.get("post_process", "0") in {"1", "true", "on", "yes"}
    speakers = request.form.get("speakers", "0") in {"1", "true", "on", "yes"}

    job_id = uuid.uuid4().hex[:12]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{job_id}{ext}"
    f.save(dest)

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "filename": f.filename,
            "status": "queued",
            "progress": 0,
            "message": "In wachtrij…",
            "text": "",
            "output_txt": None,
            "output_srt": None,
            "output_dir": str(OUTPUT_DIR),
            "created_at": time.time(),
            "elapsed_seconds": 0,
            "model": model_id,
        }

    threading.Thread(
        target=_process_job,
        args=(
            job_id,
            dest,
            f.filename,
            language,
            model_id,
            timestamps,
            post_process,
            speakers,
        ),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.get("/api/job/<job_id>")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job niet gevonden"}), 404
        return jsonify(job)


@app.get("/api/open-output")
def open_output():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(["open", str(OUTPUT_DIR)])
    return jsonify({"ok": True, "path": str(OUTPUT_DIR)})


@app.get("/api/open-folder")
def open_folder():
    which = (request.args.get("which") or "transcripts").strip().lower()
    if which in {"eg2", "eg2app", "embeddinggemma"}:
        if not EG2_APP.exists():
            return jsonify({"error": "EG2 Transcribe niet geïnstalleerd"}), 404
        subprocess.Popen(["open", str(EG2_APP)])
        return jsonify({"ok": True, "path": str(EG2_APP)})

    mapping = {
        "transcripts": OUTPUT_DIR,
        "voicememos": VOICEMEMOS_DIR,
        "dictafoon": VOICEMEMOS_DIR,
        "handy": HANDY_DIR,
    }
    path = mapping.get(which)
    if path is None:
        return jsonify({"error": "Onbekende map"}), 400
    if which == "transcripts":
        path.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return jsonify({"error": f"Map niet gevonden: {path}"}), 404
    subprocess.Popen(["open", str(path)])
    return jsonify({"ok": True, "path": str(path)})


@app.get("/api/transcripts")
def api_transcripts():
    return jsonify({"items": _list_transcripts(), "dir": str(OUTPUT_DIR)})


@app.get("/api/transcript/<path:filename>")
def api_transcript(filename: str):
    safe = Path(filename).name
    if not (safe.endswith(".txt") or safe.endswith(".srt")):
        return jsonify({"error": "Ongeldig bestand"}), 400
    path = OUTPUT_DIR / safe
    if not path.exists() or not path.is_file():
        return jsonify({"error": "Niet gevonden"}), 404
    text = _clean_text(path.read_text(encoding="utf-8", errors="replace"))
    meta = {}
    meta_path = path.with_name(path.stem + "_meta.json")
    if safe.endswith(".txt") and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return jsonify({"name": safe, "text": text, "meta": meta})


@app.get("/transcripts/<path:filename>")
def serve_transcript(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Whisper: {WHISPER_MODEL} ({'OK' if WHISPER_MODEL.exists() else 'MISSING'})")
    print(f"Parakeet: {PARAKEET_MODEL} ({'OK' if PARAKEET_MODEL.exists() else 'MISSING'})")
    print(f"Transcripts → {OUTPUT_DIR}")
    print("Open http://127.0.0.1:5055")
    app.run(host="127.0.0.1", port=5055, debug=False, threaded=True)
