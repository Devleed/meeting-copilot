# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the tool

```bash
# Without context (context is optional)
python3 src/copilot.py

# With one or more context files
python3 src/copilot.py context.txt
python3 src/copilot.py context.pdf
python3 src/copilot.py context.docx

# Required env var
export OPENAI_API_KEY=sk-...
```

## Dependencies

Install with:
```bash
pip install faster-whisper sounddevice numpy silero-vad openai pypdf2 python-docx torch
```

Target platform: Apple Silicon macOS. Whisper uses `device="auto"` and `compute_type="int8"` (switched from `float16` due to compatibility). VAD model is loaded from `snakers4/silero-vad` via `torch.hub`.

BlackHole 2ch virtual audio device must be installed and set as system audio output via macOS Audio MIDI Setup Multi-Output Device.

## Architecture and data flow

```
BlackHole (system audio) → sounddevice InputStream → audio_callback → audio_queue
                                                                            ↓
                                                                    transcribe_loop
                                                                    (VAD per 512-sample chunk)
                                                                            ↓ (speech detected → silence)
                                                                    WhisperModel.transcribe()
                                                                            ↓ (text buffered)
                                                                    flush_to_openai() after FLUSH_WAIT_SECONDS
                                                                            ↓
                                                                    get_suggestion() → GPT-4o → terminal
```

The pipeline entry point is `src/copilot.py`. Audio pipeline lives in `src/audio/pipeline.py`. Assistant logic lives in `src/assistant/`. RAG/context logic lives in `src/rag/`.

## Key tuning parameters (in src/config.py, read from env vars)

| Variable | Default | Effect |
|---|---|---|
| `SILENCE_THRESHOLD` | 4.0s | Seconds of silence before utterance is committed |
| `FLUSH_WAIT_SECONDS` | 4.0s | Seconds after last buffered segment before OpenAI is called |
| `SPEECH_PROBABILITY_THRESHOLD` | 0.2 | VAD sensitivity (lower = catches more speech) |
| `MIN_UTTERANCE_SECONDS` | 1.0s | Minimum utterance length to transcribe |
| `VAD_CHUNK_SIZE` | 512 samples | Silero VAD processes audio in fixed-size chunks at 16kHz |

## Greeting/noise suppression

`src/assistant/greeting_filter.py` has an `is_greeting()` guard that skips OpenAI calls for short social phrases (hi, thanks, ok, etc.) and anything starting with a known greeting. Extend the `DEFAULT_GREETINGS` list to suppress more patterns.

Short transcriptions under 4 words should be filtered before calling `get_suggestion()` — this is currently handled implicitly by `MIN_UTTERANCE_SECONDS` and Whisper output quality, not by an explicit word-count check.

## OpenAI prompt contract

The system prompt expects GPT-4o to reply in exactly this format:
```
ANSWER: <suggested answer>
FOLLOW-UP: <suggested follow-up question>
```
The response is printed raw — there is no parser. If the format changes, the terminal output breaks silently.
