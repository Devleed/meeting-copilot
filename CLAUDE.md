# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the tool

```bash
# Without context (context is optional)
uv run copilot

# With one or more context files
uv run copilot context.txt
uv run copilot context.pdf
uv run copilot context.docx
uv run copilot context.txt context.pdf context.docx

# Required env vars (set in .env or shell)
export OPENAI_API_KEY=sk-...          # required for GPT-4o and embeddings
export ANTHROPIC_API_KEY=sk-ant-...  # required if LLM_PROVIDER=claude
export LLM_PROVIDER=openai           # "openai" (default) or "claude"
export COHERE_API_KEY=...            # optional — enables Cohere reranking in RAG mode
```

## Dependencies

Install with uv (preferred):

```bash
uv sync
```

Or with pip:

```bash
pip install faster-whisper sounddevice numpy openai anthropic torch torchaudio tiktoken qdrant-client pypdf2 python-docx python-dotenv rank-bm25 cohere
```

Target platform: Apple Silicon macOS. Whisper uses `device="auto"` and `compute_type="int8"`. VAD model is loaded from `snakers4/silero-vad` via `torch.hub`.

BlackHole 2ch virtual audio device must be installed and set as system audio output via macOS Audio MIDI Setup Multi-Output Device.

## Module structure

```
src/
├── copilot.py                   # Entry point — composition root, wires all components
├── config.py                    # AppConfig — reads all env vars with typed defaults
│
├── audio/
│   ├── pipeline.py              # AudioPipeline — orchestrates the full runtime loop
│   ├── audio_capture.py         # AudioCapture — opens the sounddevice InputStream
│   ├── audio_device.py          # BlackHoleDeviceLocator — finds BlackHole device index
│   ├── vad.py                   # VoiceActivityDetector — Silero VAD wrapper
│   ├── speech_transcriber.py    # SpeechTranscriber — faster-whisper wrapper
│   └── flush_buffer.py          # FlushBuffer — debounces text segments before AI call
│
├── assistant/
│   ├── suggestion_generator.py  # SuggestionGenerator — builds prompt, calls LLM, prints output
│   ├── greeting_filter.py       # GreetingFilter — skips filler phrases like "hi", "ok"
│   └── conversation_history.py  # ConversationHistory — rolling window of recent utterances
│
├── llm/
│   ├── base.py                  # BaseLLMService — abstract interface (get_suggestion, stream_suggestion)
│   ├── factory.py               # LLMServiceFactory — reads LLM_PROVIDER, returns right service
│   ├── openai_service.py        # OpenAIService — GPT-4o via OpenAI API
│   └── claude_service.py        # ClaudeService — Claude via Anthropic API (claude-sonnet-4-6)
│
├── rag/
│   ├── context_reader.py        # ContextLoader + ContextReaderFactory — loads .txt/.pdf/.docx
│   ├── retriever_builder.py     # RetrieverBuilder — decides full-context vs RAG, assembles retriever
│   ├── retriever.py             # BaseRetriever, FullContextRetriever, HybridRetriever
│   ├── embedding_service.py     # EmbeddingService — wraps OpenAI embeddings API
│   ├── vector_store.py          # VectorStore — in-memory Qdrant collection
│   ├── text_chunker.py          # TextChunker — splits docs into overlapping token-bounded chunks
│   └── embedder.py              # (embedding utilities)
│
└── chat_history/
    └── chat_history_service.py  # ChatHistoryService — records session exchanges, exports TXT
```

## Architecture and data flow

```
BlackHole (system audio)
    → sounddevice InputStream
    → _audio_callback()          (PortAudio thread — copies buffer onto audio_queue)
    → audio_queue                (thread-safe Queue decouples real-time from processing)
    → _vad_loop()                (background daemon thread)
         ↓ VAD per 512-sample chunk (Silero)
         ↓ speech detected → silence (SILENCE_THRESHOLD)
    → SpeechTranscriber.transcribe()    (faster-whisper)
    → FlushBuffer.add()          (debounce timer — FLUSH_WAIT_SECONDS)
         ↓ timer fires
    → SuggestionGenerator.generate()   (daemon thread — non-blocking)
         ↓ GreetingFilter check
         ↓ ConversationHistory.add()
         ↓ BaseRetriever.get_context()
         ↓ LLM system prompt assembled
    → BaseLLMService.stream_suggestion()
    → terminal output (streamed)
    → ChatHistoryService.add_entry()
```

### Retrieval modes

`RetrieverBuilder.build()` selects between two strategies based on total token count:

- **Full-context** (`FullContextRetriever`): total tokens < `EMBEDDER_TOKEN_THRESHOLD` (default 4000). The entire document is passed raw to the LLM — no embedding needed.
- **RAG / hybrid** (`HybridRetriever`): total tokens ≥ threshold. Documents are chunked, embedded via OpenAI, and stored in an in-memory Qdrant collection. At query time, results from vector search (semantic) and BM25 (keyword) are merged and optionally reranked by Cohere.

### Operating modes

The pipeline supports two modes, toggled at runtime by typing `m` + Enter:

- **auto** (default): VAD detects end-of-utterance and triggers transcription automatically.
- **manual**: audio accumulates in a buffer; pressing Enter triggers transcription. Greeting filter is bypassed in manual mode.

Type `q` + Enter to end the session, view chat history, and optionally export it as a TXT file.

## Key tuning parameters (in src/config.py, read from env vars)

| Variable                       | Default                | Effect                                                            |
| ------------------------------ | ---------------------- | ----------------------------------------------------------------- |
| `SILENCE_THRESHOLD`            | 4.0s                   | Seconds of silence after speech before utterance is committed     |
| `FLUSH_WAIT_SECONDS`           | 4.0s                   | Seconds after last buffered segment before the LLM is called      |
| `SPEECH_PROBABILITY_THRESHOLD` | 0.2                    | VAD sensitivity — lower catches more speech, raise in noisy rooms |
| `MIN_UTTERANCE_SECONDS`        | 1.0s                   | Utterances shorter than this are discarded                        |
| `VAD_CHUNK_SIZE`               | 512 samples            | Silero VAD chunk size at 16 kHz (32 ms); do not change            |
| `SAMPLE_RATE`                  | 16000                  | Audio sample rate in Hz; Whisper's native rate                    |
| `MODEL_SIZE`                   | base                   | faster-whisper model variant (tiny/base/small/medium/large-v2)    |
| `MODE`                         | auto                   | Default operating mode ("auto" or "manual")                       |
| `EMBEDDER_TOKEN_THRESHOLD`     | 4000                   | Token count above which RAG mode is activated                     |
| `EMBEDDER_TARGET_CHUNK_TOKENS` | 400                    | Target tokens per chunk in RAG mode                               |
| `EMBEDDER_OVERLAP_TOKENS`      | 50                     | Token overlap between consecutive chunks                          |
| `EMBEDDER_EMBED_MODEL`         | text-embedding-3-small | OpenAI embedding model                                            |
| `EMBEDDER_EMBED_DIM`           | 1536                   | Vector dimensionality (must match embed model)                    |
| `LLM_PROVIDER`                 | openai                 | LLM backend: "openai" (GPT-4o) or "claude" (claude-sonnet-4-6)    |

## LLM provider contract

All LLM providers implement `BaseLLMService` (`src/llm/base.py`), which has two methods:

- `get_suggestion(system_prompt, user_message, max_tokens)` → `str`
- `stream_suggestion(system_prompt, user_message, max_tokens)` → `Iterator[str]`

`SuggestionGenerator` always uses `stream_suggestion` so the response appears incrementally. The system prompt instructs the model to respond in exactly this format:

```
ANSWER: <suggested answer>
FOLLOW-UP: <suggested follow-up question>
```

The response is streamed raw to the terminal — there is no parser. If the format drifts, output breaks silently.

## Greeting/noise suppression

`src/assistant/greeting_filter.py` has an `is_greeting()` guard that skips LLM calls for short social phrases (hi, thanks, ok, etc.) and anything starting with a known greeting. Extend the `DEFAULT_GREETINGS` list to suppress more patterns. The filter is bypassed entirely when the user triggers transcription manually.
