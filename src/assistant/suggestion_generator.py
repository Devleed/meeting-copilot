from typing import Iterator

from llm.base import BaseLLMService


class SuggestionGenerator:
    """
    Generates real-time meeting suggestions via an injected LLM service.

    Two usage modes:
      generate()       — prints ANSWER + FOLLOW-UP to the terminal (direct use).
      stream_response() — yields raw LLM chunks without printing (REST server use).

    Collaborators injected at construction (Dependency Inversion Principle):
      llm_service          — BaseLLMService; handles the actual API call.
      retriever            — BaseRetriever; supplies meeting context for each query.
      greeting_filter      — GreetingFilter; skips API calls for filler phrases.
      conversation_history — ConversationHistory; provides conversational context.
    """

    def __init__(
        self,
        llm_service,
        retriever,
        greeting_filter,
        conversation_history,
        chat_history_service=None,
    ) -> None:
        self._llm = llm_service
        self._retriever = retriever
        self._greeting_filter = greeting_filter
        self._history = conversation_history
        self._chat_history = chat_history_service
        self._max_tokens: int = 300

    # ── Shared prompt assembly ─────────────────────────────────────────────────

    def _build_prompt(
        self, they_said: str, manual: bool
    ) -> tuple[str, str] | None:
        """
        Build (system_prompt, user_message) for the LLM call.

        Returns None if the utterance should be skipped (greeting in auto mode).
        Side-effect: adds the utterance to conversation history.
        """
        if not manual and self._greeting_filter.is_greeting(they_said):
            return None

        self._history.add(they_said)

        meeting_context: str = self._retriever.get_context(they_said)
        history_block: str = self._history.format_block()

        context_block: str = (
            f"\n\nMeeting context:\n{meeting_context}" if meeting_context else ""
        )
        system_prompt: str = f"""You are a real-time meeting assistant.
A person in the meeting has just finished speaking. Your job is to:
1. Suggest a concise, confident answer the user can say
2. Suggest one smart follow-up question the user can ask

Keep suggestions short and natural — like something a human would actually say.
Format your response exactly like this:

ANSWER: <suggested answer>
FOLLOW-UP: <suggested follow-up question>

<meeting-context>{context_block}</meeting-context>
<chat-history>{history_block}</chat-history>"""

        return system_prompt, f"They just said: {they_said}"

    # ── Direct terminal use ────────────────────────────────────────────────────

    def generate(self, they_said: str, manual: bool = False) -> None:
        """
        Generate and print an AI suggestion for the given utterance.

        Skips the API call if the utterance is a greeting (unless manual=True).
        """
        result = self._build_prompt(they_said, manual)
        if result is None:
            print(f"\n[ greeting ignored: '{they_said}' ]\n")
            return

        system_prompt, user_message = result

        try:
            print("\n" + "─" * 50)
            print(f"THEY SAID: {they_said}")
            print()

            chunks: list[str] = []
            for chunk in self._llm.stream_suggestion(
                system_prompt=system_prompt,
                user_message=user_message,
                max_tokens=self._max_tokens,
            ):
                print(chunk, end="", flush=True)
                chunks.append(chunk)

            print("\n" + "─" * 50 + "\n")

            if self._chat_history is not None:
                self._chat_history.add_entry(they_said, "".join(chunks))

        except Exception as e:
            print(f"[LLM error]: {e}")

    # ── REST server use ────────────────────────────────────────────────────────

    def stream_response(
        self, they_said: str, manual: bool = False
    ) -> Iterator[str]:
        """
        Yield raw LLM chunks without printing.

        Used by the REST server to stream the response via SSE. Returns an
        empty iterator if the utterance is a greeting in auto mode.
        """
        result = self._build_prompt(they_said, manual)
        if result is None:
            return

        system_prompt, user_message = result

        yield from self._llm.stream_suggestion(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=self._max_tokens,
        )
