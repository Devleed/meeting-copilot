# =============================================================================
# LEGACY STUB: embedder.py
#
# The embedding and retriever-building logic that previously lived in this
# file has been refactored into SOLID-compliant classes distributed across
# multiple modules:
#
#   Token counting     → config.py + retriever_builder.py (AppConfig._count_tokens)
#   Text chunking      → text_chunker.py   (TextChunker)
#   Embedding API      → embedding_service.py (EmbeddingService)
#   Vector storage     → vector_store.py   (VectorStore)
#   Retriever assembly → retriever_builder.py (RetrieverBuilder)
#   Retriever classes  → retriever.py      (FullContextRetriever, HybridRetriever)
#
# The entry point is copilot.py, which constructs RetrieverBuilder directly.
# =============================================================================

# Re-export build_retriever for any external callers that still import it from here.
from rag.retriever_builder import RetrieverBuilder as _RetrieverBuilder
from config import AppConfig as _AppConfig


def build_retriever(context_docs: list):
    """
    Backwards-compatibility shim.

    Prefer instantiating RetrieverBuilder directly via copilot.py.
    """
    return _RetrieverBuilder(_AppConfig()).build(context_docs)
