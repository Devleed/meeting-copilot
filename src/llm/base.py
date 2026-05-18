from abc import ABC, abstractmethod
from collections.abc import Iterator


class BaseLLMService(ABC):
    """
    Abstract base for LLM provider integrations.

    All providers must implement get_suggestion() and stream_suggestion().
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

    @abstractmethod
    def stream_suggestion(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 300,
    ) -> Iterator[str]:
        """
        Stream the LLM response, yielding text chunks as they arrive.

        Parameters
        ----------
        system_prompt : str
            Instructions and meeting context passed as the system role.
        user_message : str
            The transcribed utterance from the speaker.
        max_tokens : int
            Upper bound on response length in tokens.

        Yields
        ------
        str
            Successive text chunks of the model's reply.
        """
