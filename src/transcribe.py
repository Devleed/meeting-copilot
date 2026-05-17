# =============================================================================
# LEGACY STUB: transcribe.py
#
# The audio pipeline logic that previously lived in this file has been
# refactored into SOLID-compliant classes distributed across multiple modules:
#
#   Audio capture    → audio/audio_capture.py   (AudioCapture)
#   Device detection → audio/audio_device.py   (BlackHoleDeviceLocator)
#   VAD              → audio/vad.py             (VoiceActivityDetector)
#   Transcription    → audio/speech_transcriber.py (SpeechTranscriber)
#   Flush/debounce   → audio/flush_buffer.py   (FlushBuffer)
#   Pipeline wiring  → audio/pipeline.py       (AudioPipeline)
#   Configuration    → config.py         (AppConfig)
#
# The entry point is copilot.py, which constructs and runs AudioPipeline.
# =============================================================================

# Re-export AudioPipeline under the old name for any external callers that
# still import `start_listening` from this module.
from audio.pipeline import AudioPipeline as _AudioPipeline
from config import AppConfig as _AppConfig
from assistant.suggestion_generator import SuggestionGenerator as _SuggestionGenerator


def start_listening(retriever):
    """
    Backwards-compatibility shim.

    Prefer instantiating AudioPipeline directly via copilot.py.
    This shim builds a minimal SuggestionGenerator from environment variables
    and delegates to AudioPipeline.run().
    """
    from assistant.greeting_filter import GreetingFilter, DEFAULT_GREETINGS
    from assistant.conversation_history import ConversationHistory

    config = _AppConfig()
    generator = _SuggestionGenerator(
        config=config,
        retriever=retriever,
        greeting_filter=GreetingFilter(DEFAULT_GREETINGS),
        conversation_history=ConversationHistory(max_size=5),
    )
    pipeline = _AudioPipeline(config=config, suggestion_generator=generator)
    pipeline.run()
