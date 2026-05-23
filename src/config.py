# =============================================================================
# COMPONENT: config.py
#
# Single Responsibility: load and expose every tunable parameter from
# environment variables with typed defaults. No business logic lives here —
# only value resolution. Every other module imports from this class instead
# of calling os.getenv directly, which keeps env-var names in one place and
# makes testing easier (swap one AppConfig instance instead of monkey-patching
# dozens of os.getenv calls).
#
# SOLID notes:
#   S — sole job: parse env vars and expose them as typed attributes.
#   O — adding a new setting means adding one attribute; nothing else changes.
#   D — modules depend on this config object (abstraction), not on os.getenv
#       (the detail), satisfying the Dependency Inversion Principle.
# =============================================================================

import os                    # os — standard library; provides os.getenv for reading env vars
import platform              # platform — detects the OS at startup for per-platform defaults
from dotenv import load_dotenv  # load_dotenv — reads a .env file into os.environ at startup

# load_dotenv() — scans the current working directory for a file named ".env"
# and injects its KEY=VALUE lines into os.environ before any code reads them.
# Must be called before any os.getenv / os.environ.get usage so overrides take effect.
load_dotenv()


class AppConfig:
    """
    Central, single-source-of-truth configuration object for the entire pipeline.

    Responsibilities:
      - Read every environment variable the application cares about.
      - Convert raw strings to the correct Python type (int, float, str).
      - Fall back silently to sensible defaults when a variable is absent or empty.
      - Expose all settings as plain attributes so IDE auto-complete works and
        so tests can instantiate AppConfig and override individual values freely.

    Usage:
        config = AppConfig()
        print(config.sample_rate)   # 16000 (or whatever the env var says)
    """

    def __init__(self) -> None:
        # ── Audio capture settings ────────────────────────────────────────────

        # blackhole_device_index : int
        #   Fallback sounddevice index used only when BlackHole auto-detection fails.
        #   BlackHole is the virtual macOS audio device that mirrors system output.
        self.blackhole_device_index: int = self._int("BLACKHOLE_DEVICE_INDEX", 1)

        # sample_rate : int
        #   Audio sample rate in Hz. 16 000 is Whisper's native rate; any other
        #   value requires resampling and degrades transcription quality.
        self.sample_rate: int = self._int("SAMPLE_RATE", 16000)

        # capture_sample_rate : int
        #   Native sample rate of the audio capture device. On Linux/PipeWire the
        #   pulse device runs at 48 000 Hz; AudioCapture resamples to sample_rate.
        #   On Mac, BlackHole runs at 16 000 Hz so no resampling is needed.
        _default_capture_rate = 48000 if platform.system() == "Linux" else 16000
        self.capture_sample_rate: int = self._int("CAPTURE_SAMPLE_RATE", _default_capture_rate)

        # vad_chunk_size : int
        #   Silero VAD requires exactly 512 samples per inference call at 16 kHz
        #   (32 ms of audio per chunk). Changing this breaks the VAD model contract.
        self.vad_chunk_size: int = self._int("VAD_CHUNK_SIZE", 512)

        # ── Whisper transcription settings ───────────────────────────────────

        # model_size : str
        #   faster-whisper model variant. "base" is fast and accurate enough for
        #   meeting speech. Options (ascending quality / size):
        #   tiny → base → small → medium → large-v2.
        _VALID_MODEL_SIZES = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
        _model_size = self._str("MODEL_SIZE", "base")
        if _model_size not in _VALID_MODEL_SIZES:
            raise ValueError(
                f"Invalid MODEL_SIZE '{_model_size}'. "
                f"Must be one of: {', '.join(sorted(_VALID_MODEL_SIZES))}"
            )
        self.model_size: str = _model_size

        # ── Voice Activity Detection settings ────────────────────────────────

        # silence_threshold : float
        #   Seconds of continuous silence after speech that commits an utterance.
        #   Lower → faster response but may split mid-sentence.
        #   Higher → complete utterances but noticeable delay.
        self.silence_threshold: float = self._float("SILENCE_THRESHOLD", 4.0)

        # speech_probability_threshold : float
        #   Silero returns a float in [0, 1]. Chunks above this are classified
        #   as speech. 0.2 catches soft or distant voices. Raise to 0.5 in
        #   noisy environments to reduce false positives.
        self.speech_probability_threshold: float = self._float(
            "SPEECH_PROBABILITY_THRESHOLD", 0.2
        )

        # min_utterance_seconds : float
        #   Utterances shorter than this are discarded (clicks, breaths, fillers).
        self.min_utterance_seconds: float = self._float("MIN_UTTERANCE_SECONDS", 1.0)

        # ── Flush / debounce settings ─────────────────────────────────────────

        # flush_wait_seconds : float
        #   After the last transcribed segment lands, wait this long before
        #   calling the AI. Debounces multi-segment turns so "Can you" and
        #   "walk me through that?" arrive as one coherent prompt.
        self.flush_wait_seconds: float = self._float("FLUSH_WAIT_SECONDS", 4.0)

        # ── Operating mode ────────────────────────────────────────────────────

        # mode : str
        #   "auto"   → VAD end-of-utterance triggers transcription automatically.
        #   "manual" → audio accumulates; pressing Enter triggers transcription.
        self.mode: str = self._str("MODE", "auto")

        # ── Embedding / retrieval settings ────────────────────────────────────

        # embedding_encoding : str
        #   tiktoken encoding used for token counting.
        #   "cl100k_base" is the encoding used by GPT-4 and text-embedding-3-*.
        self.embedding_encoding: str = self._str("EMBEDDER_ENCODING", "cl100k_base")

        # token_threshold : int
        #   If the total context token count is below this threshold, skip
        #   chunking and embedding entirely — pass the raw text to GPT-4o as-is
        #   (full-context mode). Cheaper and simpler for small documents.
        self.token_threshold: int = self._int("EMBEDDER_TOKEN_THRESHOLD", 4000)

        # target_chunk_tokens : int
        #   Desired size in tokens for each chunk when chunking is enabled.
        #   Smaller chunks → more precise retrieval; larger chunks → more context.
        self.target_chunk_tokens: int = self._int("EMBEDDER_TARGET_CHUNK_TOKENS", 400)

        # overlap_tokens : int
        #   Token overlap between consecutive chunks to preserve cross-boundary
        #   context. Without overlap, sentences split at boundaries lose meaning.
        self.overlap_tokens: int = self._int("EMBEDDER_OVERLAP_TOKENS", 50)

        # embed_model : str
        #   OpenAI embedding model used to vectorise document chunks and queries.
        self.embed_model: str = self._str("EMBEDDER_EMBED_MODEL", "text-embedding-3-small")

        # embed_dim : int
        #   Dimensionality of vectors produced by embed_model.
        #   Must match the model spec: text-embedding-3-small → 1536 dimensions.
        self.embed_dim: int = self._int("EMBEDDER_EMBED_DIM", 1536)

        # collection_name : str
        #   Name of the Qdrant in-memory collection that stores chunk vectors.
        self.collection_name: str = self._str(
            "EMBEDDER_COLLECTION_NAME", "context_chunks"
        )

        # ── LLM provider selection ────────────────────────────────────────────

        # llm_provider : str
        #   Which LLM backend to use for suggestions. Accepted values:
        #   "openai"  → GPT-4o via OpenAI API (requires OPENAI_API_KEY)
        #   "claude"  → Claude via Anthropic API (requires ANTHROPIC_API_KEY)
        self.llm_provider: str = self._str("LLM_PROVIDER", "openai")

        # ── API keys ──────────────────────────────────────────────────────────

        # openai_api_key : str | None
        #   OpenAI API key. Required for GPT-4o suggestions and embeddings.
        #   None if the env var is missing; downstream code raises on first use.
        self.openai_api_key: str | None = os.environ.get("OPENAI_API_KEY")

        # anthropic_api_key : str | None
        #   Anthropic API key. Required when LLM_PROVIDER=claude.
        #   None if the env var is missing; downstream code raises on first use.
        self.anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")

        # cohere_api_key : str | None
        #   Cohere API key. Optional. If set, enables reranking of RAG candidates
        #   using Cohere's rerank-v3.5 model, improving retrieval relevance.
        self.cohere_api_key: str | None = os.environ.get("COHERE_API_KEY")

    # ── Private helper methods ─────────────────────────────────────────────────
    # Leading underscore _ signals "internal use only" by Python convention.
    # (Python has no true private members; the underscore is a social contract.)
    # @staticmethod — these helpers need no access to self or the class itself.

    @staticmethod
    def _int(name: str, default: int) -> int:
        # os.getenv(name) — returns the environment variable's value as a str,
        # or None if the variable is not set. In TS: process.env[name]
        value = os.getenv(name)

        # Treat both None (not set) and "" (set to empty string) as "use default".
        if value is None or value == "":
            return default

        try:
            # int(value) — convert the string "512" to integer 512.
            # Raises ValueError if the string is not a valid integer (e.g. "abc").
            return int(value)
        except ValueError:
            # Fall back silently rather than crashing the pipeline startup.
            return default

    @staticmethod
    def _float(name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None or value == "":
            return default
        try:
            # float(value) — convert the string "4.0" to float 4.0.
            return float(value)
        except ValueError:
            return default

    @staticmethod
    def _str(name: str, default: str) -> str:
        value = os.getenv(name)
        # Ternary / conditional expression: if value is falsy (None or ""), use default.
        # Syntax: <value_if_true> if <condition> else <value_if_false>
        # In TS: value ? value : default   (using truthiness)
        return default if not value else value
