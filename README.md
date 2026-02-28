Velcra Dubbing Tool

Automatically dub foreign language videos to English using AI. Built for translating Polish educational videos but works with any language Whisper supports.
Pipeline:

Whisper extracts and transcribes the original audio
Gemini translates the full transcript to natural English (with full context)
Microsoft Edge TTS generates the English voiceover
ffmpeg merges everything back into the video


Requirements

Python 3.10+
ffmpeg installed and on PATH
A free Gemini API key from aistudio.google.com


Install
bashpip install openai-whisper edge-tts pydub torch google-genai

Usage
1. Dub a video
Open polish_dubber.py and set your Gemini API key at the top:
pythonGEMINI_API_KEY = "your_key_here"
Then run:
python pd.py myvideo.mp4
Output: myvideo_dubbed.mp4

3. Test Gemini translation on an existing transcript
If you already have a transcript_polish.txt and just want to test the translation:
python tg.py
Set the path and API key at the top of the file first.

4. Convert transcript to subtitles (.srt)
python subt.py
Set the path to your transcript_english_test.txt at the top. Outputs a .srt file you can load in VLC, upload to YouTube, or burn into the video.

Config
At the top of polish_dubber.py you can tweak:
SettingDefaultDescriptionWHISPER_MODELsmallsmall is fast, medium is more accurateTTS_VOICEen-US-GuyNeuralAny Microsoft Edge TTS voiceDUCK_ORIGINALTrueKeep original audio quietly in backgroundDUCK_VOLUME0.08Volume of original audio (0 = mute it)

Files
FileDescription
pd.py  - Main dubbing pipeline
test_gemini.py - Standalone Gemini translation tester
subt.pyConvert transcript to .srt subtitles

Notes

First run downloads the Whisper model (~460MB for small, ~1.5GB for medium)
Whisper transcription is the slowest step — expect 5-10 min per 10 min of video on CPU
Gemini translation is nearly instant (one API call for the whole transcript)
A _tmp folder is created alongside your video with transcripts and audio segments — safe to delete after you're happy with the output
Gemini free tier is more than enough for this — you're making one API call per video

