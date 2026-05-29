# Velcra Dubbing Tool (Demo only — not production)

This tool lets you dub foreign-language videos into English automatically using AI. Built for translating Polish educational videos, but works with any language Whisper supports.

## Demo

| | |
|---|---|
| **Input** | Polish lecture video |
| **Input (test snippet)** | [Polish university lecture — Computer Architecture](https://youtu.be/wSV10PlkAH8) |
| **Output** | [Dubbed English version](https://youtu.be/TpXL_c7IeUw) |

## Core pipeline

1. Whisper transcribes the original audio → raw Polish text with timestamps
2. Gemini translates the full transcript → natural English
3. Microsoft Edge TTS generates the English voiceover
4. ffmpeg merges everything back into the video

## Why this architecture?

**Whisper for transcription, Gemini for translation — not Whisper alone**

Whisper has a built-in translation mode but after testing it the output was too literal — word for word, no real context. Switched to Gemini because it actually understands what's being said, not just what words are there. The difference in natural-sounding output was immediately obvious.

**Why the transcript gets sent in chunks**

First version just dumped the entire Polish transcript into one Gemini API call. Worked fine on short videos. On longer ones the dubbed video would just stop halfway — took a while to figure out why. Turns out Gemini has an output token limit and was silently cutting off the response. The fix was splitting the transcript into chunks of 100 segments and sending each one separately. The tradeoff is that Gemini loses context at chunk boundaries — it doesn't know what was said in the previous chunk, so terminology and pronouns can drift slightly between chunks. For this demo version that's acceptable since most sentences in educational content are self-contained anyway.

In the production version we swapped Gemini out for a locally hosted LLaMA model and moved to a two-pass translation approach. First pass translates all chunks in parallel, second pass audits the chunk boundaries specifically for consistency of terminology and pronouns. This was only feasible with LLaMA running locally since it requires multiple passes over the same content without API cost concerns. Local hosting also means no data leaving the machine, which matters for university content.

**Full transcript sent with timestamps**

Each line gets a [12.34:15.67] tag before being sent to Gemini. This means whatever comes back can always be parsed back into timed segments, even if a chunk is partially malformed. The parser just skips unparseable lines and moves on rather than crashing the whole pipeline.

**Audio ducking instead of muting the original**

Keeping the original at 8% volume underneath the English dub sounds more natural than a hard swap. Gives it a feel closer to a real dub rather than just replacing the audio track completely.

**Edge TTS over other options**

Free, no API key required, and sounds decent enough for educational content. Individual clips get sped up by up to 1.5x if the English translation runs longer than the original time slot.

## Requirements

- Python 3.10+
- ffmpeg installed and on PATH
- A free Gemini API key from [aistudio.google.com](https://aistudio.google.com)

## Install

```bash
pip install -r requirements.txt
```

## Setup

Copy `.env.example` to `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

## Usage

**Dub a video**

```bash
python Velcra.py myvideo.mp4
```

Output: `myvideo_dubbed.mp4`

**Convert transcript to subtitles (.srt)**

```bash
python subtitles.py
```

Set the path to your `transcript_english.txt` at the top of the file. Outputs a `.srt` file you can load in VLC, upload to YouTube, or burn into the video.

## Config

At the top of `Velcra.py` you can tweak:

| Setting | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `small` | `small` is fast, `medium` is more accurate |
| `TTS_VOICE` | `en-GB-RyanNeural` | Any Microsoft Edge TTS voice |
| `DUCK_ORIGINAL` | `True` | Keep original audio quietly in background |
| `DUCK_VOLUME` | `0.08` | Volume of original audio (0 = mute it) |
| `GEMINI_CHUNK_SIZE` | `100` | Segments per Gemini API call |

## Files

| File | Description |
|---|---|
| `Velcra.py` | Main dubbing pipeline |
| `subtitles.py` | Convert transcript to `.srt` subtitles |

## Notes

- First run downloads the Whisper model (~460MB for small, ~1.5GB for medium)
- Whisper transcription is the slowest step — expect 5–10 min per 10 min of video on CPU
- Gemini translation is nearly instant even with chunking — each chunk is a separate API call but they're small
- A `_tmp` folder is created alongside your video with transcripts and audio segments — safe to delete after you're happy with the output
- Gemini's free tier is more than enough — you're making one small API call per chunk, not one giant one