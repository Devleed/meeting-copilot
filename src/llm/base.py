from abc import ABC, abstractmethod


class BaseLLMService(ABC):
    """
    Abstract base for LLM provider integrations.

    All providers must implement get_suggestion(), which takes a system prompt,
    a user message, and a token cap, and returns the model's raw text reply.
    """

    @abstractmethod
    def get_suggestion(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 300,
    ) -> str:
        """
        Call the underlying LLM and return its text response.

        Parameters
        ----------
        system_prompt : str
            Instructions and meeting context passed as the system role.
        user_message : str
            The transcribed utterance from the speaker.
        max_tokens : int
            Upper bound on response length in tokens.

        Returns
        -------
        str
            The model's raw text reply (ANSWER + FOLLOW-UP format).
        """
