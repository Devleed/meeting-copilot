import anthropic

from llm.base import BaseLLMService


class ClaudeService(BaseLLMService):
    """Claude (Anthropic) implementation of BaseLLMService."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def get_suggestion(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 300,
    ) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )
        return response.content[0].text.strip()
