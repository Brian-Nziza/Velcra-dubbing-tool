"""
Polish Video Auto-Dubber v3 - with Gemini translation
======================================================
Pipeline:
  1. Extract audio from video
  2. Whisper transcribes Polish → raw Polish text (no translation)
  3. Gemini translates full transcript → natural English
  4. edge-tts generates English voiceover
  5. ffmpeg merges everything back into the video

INSTALL:
    pip install openai-whisper edge-tts pydub torch google-genai

Usage:
    python polish_dubber.py myvideo.mp4
    Output: myvideo_dubbed.mp4
"""

import sys
import asyncio
import subprocess
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = "AIzaSyBvMNK3kywbBXCxD_BvOsS6nIgLs6ErQoA"

WHISPER_MODEL = "small"           # small=fast, medium=better quality
TTS_VOICE     = "en-US-GuyNeural" # Alternatives: en-US-JennyNeural, en-GB-RyanNeural
DUCK_ORIGINAL = True              # Keep quiet Polish audio in background
DUCK_VOLUME   = 0.08              # 8% volume for original (0 = fully mute)

# ──────────────────────────────────────────────────────────────────────────────

def run(cmd):
    print(f"  >> {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result

def get_video_duration(video_path: Path) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def extract_audio(video_path: Path, out_wav: Path):
    print("\n[1/6] Extracting audio from video...")
    run(["ffmpeg", "-y", "-i", str(video_path),
         "-ac", "1", "-ar", "16000", "-vn", str(out_wav)])
    print(f"      Done: {out_wav}")

def transcribe_polish(wav_path: Path) -> list:
    """Transcribe Polish audio to Polish text with timestamps. No translation."""
    print(f"\n[2/6] Transcribing Polish audio with Whisper ({WHISPER_MODEL})...")
    print("      Downloading model if first time (~460MB for small)...")
    import whisper
    model = whisper.load_model(WHISPER_MODEL)
    print("      Model loaded. Transcribing... (watch progress below)")
    result = model.transcribe(
        str(wav_path),
        language="pl",
        task="transcribe",  # Polish text only, no translation
        verbose=True        # Shows each segment as it's processed
    )
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result["segments"]
        if s["text"].strip()
    ]
    if not segments:
        print("ERROR: Whisper returned no segments. Is there speech in the video?")
        sys.exit(1)

    print(f"\n      Got {len(segments)} segments of Polish text.")

    # Save raw Polish transcript
    polish_transcript_path = wav_path.parent / "transcript_polish.txt"
    with open(polish_transcript_path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"[{s['start']:.2f}:{s['end']:.2f}] {s['text']}\n")
    print(f"      Polish transcript saved: {polish_transcript_path}")
    return segments

def translate_with_gemini(segments: list, tmp_dir: Path) -> list:
    """Send full Polish transcript to Gemini for natural English translation."""
    print(f"\n[3/6] Translating with Gemini (full context, natural English)...")

    if GEMINI_API_KEY == "PASTE YOUR GEMINI API KEY HERE":
        print("ERROR: You need to set your Gemini API key in the script!")
        sys.exit(1)

    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build the transcript with timestamps so Gemini can preserve them
    transcript_text = "\n".join(
        f"[{s['start']:.2f}:{s['end']:.2f}] {s['text']}"
        for s in segments
    )

    prompt = f"""You are translating a Polish educational video transcript to English.

Your job:
- Translate each line from Polish to natural, fluent English
- Keep the EXACT timestamp tags like [12.34:15.67] unchanged at the start of each line
- Make the English sound natural and coherent - not word-for-word robotic
- Keep the same number of lines as the input
- Educational tone, clear language

Transcript to translate:
{transcript_text}

Return ONLY the translated lines with their timestamps, nothing else."""

    print("      Sending transcript to Gemini...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    translated_text = response.text.strip()

    # Save English transcript
    english_transcript_path = tmp_dir / "transcript_english.txt"
    with open(english_transcript_path, "w", encoding="utf-8") as f:
        f.write(translated_text)
    print(f"      English transcript saved: {english_transcript_path}")

    # Parse back into segments
    translated_segments = []
    for line in translated_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            # Parse [start:end] timestamp
            bracket_end = line.index("]")
            timestamps = line[1:bracket_end].split(":")
            start = float(timestamps[0])
            end   = float(timestamps[1])
            text  = line[bracket_end+1:].strip()
            if text:
                translated_segments.append({"start": start, "end": end, "text": text})
        except (ValueError, IndexError):
            # If a line doesn't parse cleanly, skip it
            print(f"      Skipping unparseable line: {line[:60]}")
            continue

    if not translated_segments:
        print("ERROR: Could not parse Gemini's response. Check transcript_english.txt")
        sys.exit(1)

    print(f"      Parsed {len(translated_segments)} translated segments.")
    return translated_segments

async def generate_all_tts(segments: list, tmp_dir: Path):
    import edge_tts

    async def make_one(i, text, out_path):
        if out_path.exists():
            return  # resume support
        try:
            communicate = edge_tts.Communicate(text, TTS_VOICE)
            await communicate.save(str(out_path))
        except Exception as e:
            print(f"      Warning: TTS failed for segment {i}: {e}")

    batch_size = 20
    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start : batch_start + batch_size]
        tasks = [
            make_one(
                batch_start + i,
                seg["text"],
                tmp_dir / f"seg_{batch_start + i:04d}.mp3"
            )
            for i, seg in enumerate(batch)
        ]
        await asyncio.gather(*tasks)
        done = min(batch_start + batch_size, len(segments))
        print(f"      TTS progress: {done}/{len(segments)} segments")

def generate_tts_audio(segments: list, tmp_dir: Path, video_duration: float) -> Path:
    print(f"\n[4/6] Generating English TTS ({TTS_VOICE})...")
    print("      (Requires internet)")

    asyncio.run(generate_all_tts(segments, tmp_dir))

    print("      Assembling audio timeline...")
    from pydub import AudioSegment

    full_audio = AudioSegment.silent(duration=int(video_duration * 1000))
    loaded = 0

    for i, seg in enumerate(segments):
        mp3_path = tmp_dir / f"seg_{i:04d}.mp3"
        if not mp3_path.exists():
            continue
        try:
            tts_audio = AudioSegment.from_mp3(str(mp3_path))
        except Exception as e:
            print(f"      Warning: could not load segment {i}: {e}")
            continue

        start_ms = int(seg["start"] * 1000)
        end_ms   = int(seg["end"]   * 1000)
        slot_ms  = end_ms - start_ms

        # Speed up TTS to fit the slot if needed (max 1.5x)
        if len(tts_audio) > slot_ms > 0:
            speed_factor = min(len(tts_audio) / slot_ms, 1.5)
            new_frame_rate = int(tts_audio.frame_rate * speed_factor)
            tts_audio = tts_audio._spawn(
                tts_audio.raw_data,
                overrides={"frame_rate": new_frame_rate}
            ).set_frame_rate(44100)

        full_audio = full_audio.overlay(tts_audio, position=start_ms)
        loaded += 1

    print(f"      Assembled {loaded}/{len(segments)} segments into audio track.")
    out_wav = tmp_dir / "dubbed_audio.wav"
    full_audio.export(str(out_wav), format="wav")
    print(f"      Dubbed audio saved: {out_wav}")
    return out_wav

def merge_into_video(video_path: Path, dubbed_wav: Path, original_wav: Path,
                     video_duration: float, out_path: Path):
    print("\n[5/6] Merging dubbed audio into video...")
    duration_str = str(video_duration)

    if DUCK_ORIGINAL and DUCK_VOLUME > 0:
        filter_complex = (
            f"[1:a]volume={DUCK_VOLUME}[orig];"
            f"[2:a]volume=1.0[dub];"
            f"[orig][dub]amix=inputs=2:duration=longest[aout]"
        )
        run([
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(original_wav),
            "-i", str(dubbed_wav),
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-t", duration_str,
            str(out_path)
        ])
    else:
        run([
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(dubbed_wav),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-t", duration_str,
            str(out_path)
        ])

    print(f"\n[6/6] Done!")
    print(f"      Output: {out_path}")

def check_dependencies():
    missing = []
    try:
        import whisper
    except ImportError:
        missing.append("openai-whisper")
    try:
        import edge_tts
    except ImportError:
        missing.append("edge-tts")
    try:
        from pydub import AudioSegment
    except ImportError:
        missing.append("pydub")
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        from google import genai
    except ImportError:
        missing.append("google-genai")

    if missing:
        print("\nMissing packages. Run:")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python polish_dubber.py <input_video.mp4>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: File not found: {video_path}")
        sys.exit(1)

    check_dependencies()

    out_path = video_path.parent / (video_path.stem + "_dubbed.mp4")
    tmp_dir  = video_path.parent / (video_path.stem + "_tmp")
    tmp_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(" Polish to English Auto-Dubber v3 (Gemini translation)")
    print("=" * 60)
    print(f" Input:  {video_path}")
    print(f" Output: {out_path}")
    print(f" Tmp:    {tmp_dir}")
    print("=" * 60)

    video_duration = get_video_duration(video_path)
    print(f" Video duration: {video_duration:.1f}s ({video_duration/60:.1f} min)")

    original_wav = tmp_dir / "original_audio.wav"
    extract_audio(video_path, original_wav)

    polish_segments  = transcribe_polish(original_wav)
    english_segments = translate_with_gemini(polish_segments, tmp_dir)
    dubbed_wav       = generate_tts_audio(english_segments, tmp_dir, video_duration)
    merge_into_video(video_path, dubbed_wav, original_wav, video_duration, out_path)

    print("\n Transcripts saved in the _tmp folder.")
    print(" All done!")

if __name__ == "__main__":
    main()