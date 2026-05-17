# =============================================================================
# COMPONENT: conversation_history.py
#
# Maintains a rolling window of recent utterances so the AI assistant has
# context about the flow of conversation, not just the most recent sentence.
#
# Single Responsibility: store utterances, enforce a maximum size (evicting
# the oldest entry when full), and format the history as a readable block for
# insertion into a prompt. No I/O, no threading, no external dependencies.
#
# SOLID notes:
#   S — sole job: manage a capped list of utterance strings.
#   O — max window size is injected; no code change needed to resize it.
#   D — depends only on Python built-ins; nothing to inject beyond max_size.
# =============================================================================


class ConversationHistory:
    """
    Rolling window of the most recent conversation utterances.

    Responsibility: maintain a list of transcribed utterances, cap it at
    a configurable size by evicting the oldest entry when full, and expose
    the history as a formatted block suitable for GPT-4o's system prompt.

    The sliding window gives GPT-4o conversational context (e.g. prior
    questions that were already asked) so its suggestions are coherent with
    the flow of the meeting.

    Usage:
        history = ConversationHistory(max_size=5)
        history.add("What's the timeline for the project?")
        history.add("Can you walk me through the architecture?")
        print(history.format_block())
        # Previous utterances:
        # - What's the timeline for the project?
    """

    def __init__(self, max_size: int = 5) -> None:
        # max_size : int
        #   Maximum number of utterances to retain. Older entries are evicted
        #   when the list reaches this size. Default of 5 keeps recent context
        #   without sending too many tokens to GPT-4o.
        self._max_size: int = max_size

        # _utterances : list[str]
        #   The ordered list of utterances, oldest first.
        #   Plain list — Python's GIL protects simple list operations from
        #   data races in CPython; no explicit lock needed here.
        self._utterances: list = []

    def add(self, utterance: str) -> None:
        """
        Append a new utterance to the history, evicting the oldest if full.

        Parameters
        ----------
        utterance : str
            The transcribed text of one speaker turn.
        """
        # list.append(item) — add `utterance` to the end of the list.
        # The most recent utterance is always at index -1 (the last element).
        self._utterances.append(utterance)

        # Enforce the sliding window: if we now have more than max_size entries,
        # remove the oldest one (at index 0).
        # len(self._utterances) — current number of stored utterances.
        # > self._max_size — only evict if we've exceeded the cap.
        if len(self._utterances) > self._max_size:
            # list.pop(0) — remove and discard the element at index 0 (the oldest).
            # This is O(n) because all remaining elements shift left by one position.
            # For max_size=5 this is negligible; for large lists a deque would be better.
            # In TS: this._utterances.shift()
            self._utterances.pop(0)

    def format_block(self) -> str:
        """
        Return the conversation history formatted as a prompt-ready string.

        The most recent utterance is excluded from the block because it is
        always shown separately as "They just said: <utterance>" in the user
        message of the GPT-4o request. Including it here would duplicate it.

        Returns
        -------
        str
            A "Previous utterances:" block listing all stored utterances except
            the last, or an empty string if there is only one utterance (nothing
            to show as "previous").
        """
        # self._utterances[:-1] — all elements except the last (the current utterance).
        # Slice notation: list[start:stop]; omitting start means 0; -1 means up-to-last.
        # In TS: this._utterances.slice(0, -1)
        prior: list = self._utterances[:-1]

        # If there are no prior utterances, return an empty string.
        # Nothing is prepended to the GPT-4o prompt in that case.
        if not prior:
            return ""   # empty string — falsy in Python; caller can check `if block:`

        # Build a Markdown-style bullet list of prior utterances.
        # f"- {u}" — format each utterance as a "- " prefixed bullet point.
        # "\n".join(...) — join all bullet lines with newlines.
        history_lines: str = "\n".join(f"- {u}" for u in prior)

        # Return the formatted block, starting with a blank line for visual separation.
        # The \n at the start ensures this block is separated from the text before it.
        return f"\nPrevious utterances:\n{history_lines}"
