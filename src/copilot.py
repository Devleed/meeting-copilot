# =============================================================================
# ENTRY POINT: copilot.py
#
# Thin composition root that wires all classes together and starts the pipeline.
# No business logic lives here — each concern is owned by a dedicated class.
#
# Startup sequence:
#   1. Parse CLI arguments — collect one or more context file paths.
#   2. Load config         — AppConfig reads all environment variables.
#   3. Load context files  — ContextLoader reads .txt / .pdf / .docx files.
#   4. Build retriever     — RetrieverBuilder decides full-context vs RAG mode.
#   5. Build collaborators — GreetingFilter, ConversationHistory.
#   6. Build generator     — SuggestionGenerator wired with all collaborators.
#   7. Build pipeline      — AudioPipeline wired with config + generator.
#   8. Run                 — blocks forever (audio capture + VAD + Whisper + flush).
#
# Dependency graph (→ means "depends on"):
#   copilot.py
#     → AppConfig
#     → ContextLoader → ContextReaderFactory → TxtContextReader / PdfContextReader / DocxContextReader
#     → RetrieverBuilder → TextChunker, EmbeddingService, VectorStore
#                        → FullContextRetriever | HybridRetriever
#     → GreetingFilter
#     → ConversationHistory
#     → SuggestionGenerator → GreetingFilter, ConversationHistory, BaseRetriever
#     → AudioPipeline → BlackHoleDeviceLocator, AudioCapture,
#                        VoiceActivityDetector, SpeechTranscriber, FlushBuffer
# =============================================================================

import sys   # sys — sys.argv for CLI arguments, sys.exit() for error exits

# ── Configuration ──────────────────────────────────────────────────────────────
# AppConfig reads all environment variables and .env overrides at construction.
from config import AppConfig

# ── Context loading ────────────────────────────────────────────────────────────
# ContextLoader reads .txt / .pdf / .docx and returns [{filename, content}].
from rag.context_reader import ContextLoader

# ── Retriever construction ─────────────────────────────────────────────────────
# RetrieverBuilder decides full-context vs RAG and returns a BaseRetriever.
from rag.retriever_builder import RetrieverBuilder

# ── Greeting filter ────────────────────────────────────────────────────────────
# GreetingFilter and the default phrase list used to initialise it.
from assistant.greeting_filter import GreetingFilter, DEFAULT_GREETINGS

# ── Conversation history ───────────────────────────────────────────────────────
# ConversationHistory maintains the last N utterances for the GPT-4o prompt.
from assistant.conversation_history import ConversationHistory

# ── LLM service ───────────────────────────────────────────────────────────────
# LLMServiceFactory reads LLM_PROVIDER and returns the right BaseLLMService.
from llm.factory import LLMServiceFactory

# ── AI suggestion generator ────────────────────────────────────────────────────
# SuggestionGenerator calls the injected LLM service and prints ANSWER + FOLLOW-UP.
from assistant.suggestion_generator import SuggestionGenerator

# ── Audio pipeline ─────────────────────────────────────────────────────────────
# AudioPipeline runs the audio capture → VAD → Whisper → flush loop.
from audio.pipeline import AudioPipeline

# ── Chat history ───────────────────────────────────────────────────────────────
# ChatHistoryService records every exchange for the current session.
from chat_history.chat_history_service import ChatHistoryService


def main() -> None:
    config = AppConfig()

    loader = ContextLoader()
    all_docs: list = []
    for path in sys.argv[1:]:
        docs: list = loader.load(path)
        all_docs.extend(docs)
        print(f"Context loaded from: {path}")
        print(f"Preview: {docs[0]['content'][:100]}...\n")

    if not all_docs:
        print("No context files provided — running without context.\n")

    retriever = RetrieverBuilder(config).build(all_docs)
    greeting_filter = GreetingFilter(DEFAULT_GREETINGS)
    conversation_history = ConversationHistory(max_size=5)
    llm_service = LLMServiceFactory.create(config)
    chat_history = ChatHistoryService()

    print(f"Session ID: {chat_history.session_id}\n")

    generator = SuggestionGenerator(
        llm_service=llm_service,
        retriever=retriever,
        greeting_filter=greeting_filter,
        conversation_history=conversation_history,
        chat_history_service=chat_history,
    )

    pipeline = AudioPipeline(
        config=config,
        suggestion_generator=generator,
        chat_history_service=chat_history,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
