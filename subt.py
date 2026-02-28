"""
Transcript to SRT Converter
============================
Converts transcript_english_test.txt to a proper .srt subtitle file.

Usage:
    python to_srt.py
"""

from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

TRANSCRIPT_FILE = r"C:\Users\user\OneDrive\Desktop\velcra\lesson1_tmp\transcript_english.txt"
# Example: r"C:\Users\user\OneDrive\Desktop\velcra\test1_tmp\transcript_english_test.txt"

# ─────────────────────────────────────────────────────────────────────────────

def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = int(seconds % 60)
    millis  = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def main():
    if TRANSCRIPT_FILE == "PASTE THE FULL PATH TO YOUR transcript_english_test.txt HERE":
        print("ERROR: Set the path to your transcript file at the top of this script.")
        return

    path = Path(TRANSCRIPT_FILE)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    srt_blocks = []
    index = 1

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            bracket_end = line.index("]")
            timestamps  = line[1:bracket_end].split(":")
            start       = float(timestamps[0])
            end         = float(timestamps[1])
            text        = line[bracket_end+1:].strip()
            if not text:
                continue

            srt_blocks.append(
                f"{index}\n"
                f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n"
                f"{text}\n"
            )
            index += 1
        except (ValueError, IndexError):
            print(f"Skipping unparseable line: {line[:60]}")
            continue

    if not srt_blocks:
        print("ERROR: No valid lines found in transcript.")
        return

    out_path = path.parent / (path.stem + ".srt")
    out_path.write_text("\n".join(srt_blocks), encoding="utf-8")
    print(f"Done! {index-1} subtitles written to:")
    print(f"  {out_path}")
    print(f"\nYou can now:")
    print(f"  - Load it in VLC (Subtitles > Add Subtitle File)")
    print(f"  - Upload it to YouTube as a subtitle track")
    print(f"  - Burn it into the video with ffmpeg")

if __name__ == "__main__":
    main()