# =============================================================================
# COMPONENT: suggestion_generator.py
#
# Sends a transcribed utterance to the configured LLM and prints a suggested
# answer plus a follow-up question the user can use in the meeting.
#
# Single Responsibility: build the prompt from retrieved context and
# conversation history, call the LLM service, and print the result. That's all.
# Greeting filtering and history management are handled by collaborating classes
# injected at construction time (Dependency Inversion Principle).
#
# SOLID notes:
#   S — sole job: generate and display an AI suggestion for one utterance.
#   O — system prompt format and token limit are configurable without changing
#       the class structure; switching LLM providers requires no change here.
#   D — depends on injected GreetingFilter, ConversationHistory, retriever, and
#       BaseLLMService abstractions, not on any concrete provider client.
# =============================================================================

from llm.base import BaseLLMService


class SuggestionGenerator:
    """
    Generates real-time meeting suggestions via an injected LLM service.

    Responsibility: given a transcribed utterance, retrieve relevant meeting
    context, build a prompt, call the LLM service, and print the response
    (ANSWER + FOLLOW-UP) to the terminal.

    Collaborators injected at construction (Dependency Inversion Principle):
      llm_service       — BaseLLMService; handles the actual API call.
      retriever         — BaseRetriever; supplies meeting context for each query.
      greeting_filter   — GreetingFilter; skips API calls for filler phrases.
      conversation_history — ConversationHistory; provides conversational context.

    Usage:
        gen = SuggestionGenerator(llm_service, retriever, greeting_filter, history)
        gen.generate("Can you walk me through the architecture?")
    """

    def __init__(
        self,
        llm_service,          # llm_service         — BaseLLMService instance
        retriever,            # retriever           — BaseRetriever instance
        greeting_filter,      # greeting_filter     — GreetingFilter instance
        conversation_history, # conversation_history — ConversationHistory instance
    ) -> None:
        self._llm = llm_service
        self._retriever = retriever
        self._greeting_filter = greeting_filter
        self._history = conversation_history
        self._max_tokens: int = 300

    def generate(self, they_said: str, manual: bool = False) -> None:
        """
        Generate and print an AI suggestion for the given utterance.

        Skips the API call if the utterance is a greeting (unless manual=True,
        which bypasses the filter for user-initiated transcriptions).

        Parameters
        ----------
        they_said : str
            The transcribed text of the speaker's utterance.
        manual : bool
            If True, bypass greeting filtering. Used when the user manually
            triggers transcription by pressing Enter in manual mode.
        """
        # ── Step 1: Greeting guard ────────────────────────────────────────────
        # Skip social fillers like "hi", "thanks", "ok" to avoid wasting API calls.
        # `not manual` — only apply the filter in auto mode; manual mode always fires.
        if not manual and self._greeting_filter.is_greeting(they_said):
            print(f"\n[ greeting ignored: '{they_said}' ]\n")
            return   # return — exit the method early; no API call is made

        # ── Step 2: Update conversation history ───────────────────────────────
        # Add the current utterance to the rolling window before building the prompt
        # so the history block includes utterances up to (but not including) this one.
        self._history.add(they_said)

        # ── Step 3: Retrieve meeting context ──────────────────────────────────
        # Ask the retriever for the most relevant portion of the meeting context.
        # In full-context mode this returns all text; in RAG mode it returns
        # the top-K chunks most semantically similar to the utterance.
        meeting_context: str = self._retriever.get_context(they_said)

        # ── Step 4: Build the conversation history block ──────────────────────
        # format_block() returns a formatted "Previous utterances: ..." string,
        # or "" if there is no prior history yet.
        history_block: str = self._history.format_block()

        # ── Step 5: Compose the system prompt ────────────────────────────────
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

        # ── Step 6: Call the LLM service ─────────────────────────────────────
        # Wrap in try/except to handle network errors, rate limits, and auth failures
        # gracefully — the audio pipeline continues even if one API call fails.
        try:
            reply: str = self._llm.get_suggestion(
                system_prompt=system_prompt,
                user_message=f"They just said: {they_said}",
                max_tokens=self._max_tokens,
            )

            # ── Step 7: Print the formatted response ──────────────────────────
            print("\n" + "─" * 50)
            print(f"THEY SAID: {they_said}")
            print()
            print(reply)
            print("─" * 50 + "\n")

        except Exception as e:
            print(f"[LLM error]: {e}")
