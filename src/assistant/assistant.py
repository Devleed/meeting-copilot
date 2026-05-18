# =============================================================================
# LEGACY STUB: assistant.py
#
# The assistant logic that previously lived in this file has been refactored
# into SOLID-compliant classes distributed across multiple modules:
#
#   Greeting detection    → greeting_filter.py     (GreetingFilter)
#   Conversation history  → conversation_history.py (ConversationHistory)
#   OpenAI suggestion     → suggestion_generator.py (SuggestionGenerator)
#
# The entry point is copilot.py, which constructs all of these and wires
# them into AudioPipeline.
# =============================================================================

# Re-export get_suggestion for any external callers that still import it from here.
from assistant.suggestion_generator import SuggestionGenerator as _SuggestionGenerator
from assistant.greeting_filter import GreetingFilter, DEFAULT_GREETINGS
from assistant.conversation_history import ConversationHistory
from config import AppConfig as _AppConfig
from llm.factory import LLMServiceFactory as _LLMServiceFactory

def get_suggestion(they_said: str, retriever, manual: bool = False) -> None:
    """
    Backwards-compatibility shim.

    Prefer instantiating SuggestionGenerator directly via copilot.py.
    """
    config = _AppConfig()
    generator = _SuggestionGenerator(
        llm_service=_LLMServiceFactory.create(config),
        retriever=retriever,
        greeting_filter=GreetingFilter(DEFAULT_GREETINGS),
        conversation_history=ConversationHistory(max_size=5),
    )
    generator.generate(they_said, manual=manual)
