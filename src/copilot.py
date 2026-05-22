import argparse

from config import AppConfig
from audio.pipeline import AudioPipeline
from chat_history.chat_history_service import ChatHistoryService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time meeting copilot — audio capture + AI suggestions."
    )
    parser.add_argument(
        "context_files",
        nargs="*",
        metavar="FILE",
        help="Optional context files (.txt, .pdf, .docx) passed to the suggestion service.",
    )
    parser.add_argument(
        "--remote",
        metavar="URL",
        help="URL of a running suggestion service (e.g. http://localhost:8000). "
             "When set, the local LLM stack is not loaded.",
    )
    args = parser.parse_args()

    config = AppConfig()
    chat_history = ChatHistoryService()
    print(f"Session ID: {chat_history.session_id}\n")

    if args.remote:
        from api.client import RemoteSuggestionClient
        generator = RemoteSuggestionClient(args.remote)
        print(f"Using remote suggestion service: {args.remote}\n")
    else:
        from rag.context_reader import ContextLoader
        from rag.retriever_builder import RetrieverBuilder
        from assistant.greeting_filter import GreetingFilter, DEFAULT_GREETINGS
        from assistant.conversation_history import ConversationHistory
        from llm.factory import LLMServiceFactory
        from assistant.suggestion_generator import SuggestionGenerator

        loader = ContextLoader()
        all_docs: list = []
        for path in args.context_files:
            docs: list = loader.load(path)
            all_docs.extend(docs)
            print(f"Context loaded from: {path}")

        if not all_docs:
            print("No context files provided — running without context.\n")

        retriever = RetrieverBuilder(config).build(all_docs)
        llm_service = LLMServiceFactory.create(config)

        generator = SuggestionGenerator(
            llm_service=llm_service,
            retriever=retriever,
            greeting_filter=GreetingFilter(DEFAULT_GREETINGS),
            conversation_history=ConversationHistory(max_size=5),
            chat_history_service=chat_history,
        )

    pipeline = AudioPipeline(
        config=config,
        suggestion_generator=generator,
        chat_history_service=chat_history,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
