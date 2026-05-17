# =============================================================================
# COMPONENT: greeting_filter.py
#
# Detects whether a transcribed utterance is a social filler phrase that does
# not warrant an AI suggestion. Prevents wasted OpenAI API calls on "hi",
# "thanks", "ok", and similar short conversational tokens.
#
# Single Responsibility: classify a string as a greeting (True) or not (False).
# No side effects, no I/O, no external dependencies — pure text logic.
#
# SOLID notes:
#   S — sole job: determine if text is a greeting.
#   O — extend supported phrases by passing a longer list at construction;
#       the detection logic never changes.
#   D — the list of phrases is injected, so tests can supply a minimal list
#       without depending on the production constant.
# =============================================================================


# Default phrases to suppress. Social fillers that don't need an AI response.
# Defined at module level so it can be imported as a constant by callers
# that don't want to hard-code it themselves.
DEFAULT_GREETINGS: list = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "how are you doing", "nice to meet you", "great to meet you",
    "thanks", "thank you", "bye", "goodbye", "see you", "take care", "sure",
    "okay", "ok", "yes", "no", "alright", "got it", "sounds good", "perfect",
]


class GreetingFilter:
    """
    Determines whether a transcribed utterance is a greeting or social filler.

    Responsibility: given a text string, decide if it matches any configured
    greeting phrase — either exactly (full match) or as a prefix (starts-with).
    Matching is case-insensitive and strips leading/trailing punctuation.

    Usage:
        fltr = GreetingFilter(DEFAULT_GREETINGS)
        fltr.is_greeting("Hi there!")   # → True
        fltr.is_greeting("What is the timeline?")  # → False
    """

    def __init__(self, greetings: list) -> None:
        # Store the list of greeting phrases as a private attribute.
        # Each phrase is expected to be already lowercase (the detection logic
        # lowercases the input before comparing, not the phrases themselves).
        self._greetings: list = greetings

    def is_greeting(self, text: str) -> bool:
        """
        Return True if `text` is (or begins with) a known greeting phrase.

        Matching steps:
          1. Strip leading/trailing whitespace.
          2. Lowercase the entire string.
          3. Strip trailing punctuation (.,!?) so "Hi!" matches "hi".
          4. Check for an exact match against the greetings list.
          5. Check if the text starts with any greeting (catches "Thanks for that").

        Parameters
        ----------
        text : str
            The transcribed utterance to classify.

        Returns
        -------
        bool
            True if the utterance is a greeting / filler; False otherwise.
        """
        # .strip()      — remove leading and trailing whitespace characters.
        #                 "  hi  " → "hi"
        # .lower()      — convert to lowercase so "Hello" matches "hello".
        #                 In TS: .toLowerCase()
        # .rstrip(".,!?") — strip trailing punctuation characters from the right.
        #                   "thanks!" → "thanks". rstrip removes any character in the
        #                   given string from the right end repeatedly.
        #                   In TS: .replace(/[.,!?]+$/, '')
        cleaned: str = text.strip().lower().rstrip(".,!?")

        # Exact match: is the cleaned text one of the known phrases?
        # `in self._greetings` — checks list membership (O(n) linear scan).
        # In TS: this._greetings.includes(cleaned)
        if cleaned in self._greetings:
            return True  # e.g. "okay" exactly matches "okay"

        # Prefix match: does the cleaned text begin with any greeting phrase?
        # Catches utterances like "Thanks for sharing that" (starts with "thanks").
        # `for phrase in self._greetings:` — iterate over every phrase in the list.
        # In TS: for (const phrase of this._greetings)
        for phrase in self._greetings:
            # .startswith(phrase) — returns True if cleaned begins with the phrase string.
            # In TS: cleaned.startsWith(phrase)
            if cleaned.startswith(phrase):
                return True   # e.g. "thanks for that" starts with "thanks"

        # No match found — this is a substantive utterance, not a greeting.
        return False  # False in Python — capital F; true/false in JS are lowercase
