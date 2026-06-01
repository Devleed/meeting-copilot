import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from api.models import SuggestRequest, SuggestResponse
from assistant.conversation_history import ConversationHistory
from assistant.greeting_filter import DEFAULT_GREETINGS, GreetingFilter
from assistant.suggestion_generator import SuggestionGenerator
from config import AppConfig
from llm.factory import LLMServiceFactory
from rag.context_reader import ContextLoader
from rag.retriever_builder import RetrieverBuilder


def _build_generator() -> SuggestionGenerator:
    config = AppConfig()

    env_files = os.environ.get("CONTEXT_FILES", "")
    context_paths = [p.strip() for p in env_files.split(",") if p.strip()]

    all_docs: list = []
    if context_paths:
        loader = ContextLoader()
        for path in context_paths:
            all_docs.extend(loader.load(path))
            print(f"Context loaded: {path}")
    else:
        print("No context files — running without context.")

    retriever = RetrieverBuilder(config).build(all_docs)
    llm_service = LLMServiceFactory.create(config)

    return SuggestionGenerator(
        llm_service=llm_service,
        retriever=retriever,
        greeting_filter=GreetingFilter(DEFAULT_GREETINGS),
        conversation_history=ConversationHistory(max_size=5),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.generator = _build_generator()
    yield


app = FastAPI(title="Suggestion Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    generator: SuggestionGenerator = app.state.generator
    print("\n" + "─" * 50)
    print(f"THEY SAID: {req.speech}")
    print()
    chunks: list[str] = []
    for chunk in generator.stream_response(req.speech, req.manual):
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print("\n" + "─" * 50 + "\n")
    return SuggestResponse(response="".join(chunks))


@app.post("/suggest/stream")
def suggest_stream(req: SuggestRequest):
    print(f"Speech received: '{req.speech}' (manual={req.manual})")

    generator: SuggestionGenerator = app.state.generator

    generator.generate(req.speech, req.manual)

    return None


def start() -> None:
    import uvicorn

    # Accept context file paths as CLI args: uv run suggest-server ctx.txt ctx.pdf
    context_files = [a for a in sys.argv[1:] if not a.startswith("--")]
    if context_files:
        os.environ["CONTEXT_FILES"] = ",".join(context_files)

    host = os.environ.get("SUGGEST_HOST", "0.0.0.0")
    port = int(os.environ.get("SUGGEST_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
