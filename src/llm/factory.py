from llm.base import BaseLLMService


class LLMServiceFactory:
    """
    Reads LLM_PROVIDER from config and constructs the matching BaseLLMService.

    Supported providers (LLM_PROVIDER env var):
      "openai"  → OpenAIService  (requires OPENAI_API_KEY)
      "claude"  → ClaudeService  (requires ANTHROPIC_API_KEY)
    """

    @staticmethod
    def create(config) -> BaseLLMService:
        provider = config.llm_provider.lower()

        if provider == "openai":
            if not config.openai_api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is required for LLM_PROVIDER=openai but is not set."
                )
            from llm.openai_service import OpenAIService
            return OpenAIService(api_key=config.openai_api_key)

        if provider == "claude":
            if not config.anthropic_api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY is required for LLM_PROVIDER=claude but is not set."
                )
            from llm.claude_service import ClaudeService
            return ClaudeService(api_key=config.anthropic_api_key)

        raise ValueError(
            f"Unknown LLM_PROVIDER '{config.llm_provider}'. "
            "Valid options: 'openai', 'claude'."
        )
