# =============================================================================
# COMPONENT: retriever_builder.py
#
# Decides which retrieval strategy to use based on total token count, then
# constructs and returns the appropriate BaseRetriever implementation.
#
# Single Responsibility: orchestrate the one-time setup work needed before
# retrieval can begin — count tokens, chunk documents, call the embedding
# API in a batch, populate the vector store, optionally set up Cohere.
# Return a ready-to-use BaseRetriever. Nothing else.
#
# This is the Builder pattern: the caller (copilot.py) calls build(docs) and
# receives a fully initialised retriever without needing to know whether
# full-context or RAG mode was selected, or how chunks / vectors were created.
#
# SOLID notes:
#   S — sole job: assemble a retriever from context documents.
#   O — adding a new retrieval strategy means a new BaseRetriever subclass
#       and possibly a new branch here; existing strategies are untouched.
#   D — depends on injected AppConfig and OpenAI client, not on module-level
#       globals or os.getenv calls scattered across functions.
# =============================================================================

import tiktoken  # tiktoken — counts tokens so we can decide full-context vs RAG mode

from openai import OpenAI  # OpenAI — client class; used to create embeddings

# Local classes used to assemble the retriever.
from rag.embedding_service import EmbeddingService
from rag.vector_store import VectorStore
from rag.text_chunker import TextChunker
from rag.retriever import BaseRetriever, FullContextRetriever, HybridRetriever


class RetrieverBuilder:
    """
    Constructs the appropriate BaseRetriever from a list of context documents.

    Responsibility: examine the total token count of the loaded documents and
    decide whether to use FullContextRetriever (cheap, for small docs) or
    HybridRetriever (RAG, for large docs). For RAG, chunk the text, embed
    all chunks in a single API call, populate an in-memory Qdrant store, and
    optionally configure Cohere reranking.

    Usage:
        builder = RetrieverBuilder(config)
        retriever = builder.build(docs)   # docs = [{"filename": ..., "content": ...}]
    """

    def __init__(self, config) -> None:
        # config — AppConfig instance; provides token threshold, embedding model,
        # chunk sizes, API keys, and all other tunable parameters.
        self._config = config

    def build(self, context_docs: list) -> BaseRetriever:
        """
        Build and return the right retriever for the given documents.

        Parameters
        ----------
        context_docs : list[dict]
            List of loaded document dicts: [{"filename": str, "content": str}]

        Returns
        -------
        BaseRetriever
            Either a FullContextRetriever or a HybridRetriever, depending
            on the total token count of the provided documents.
        """
        # ── Step 0: No documents — return an empty-context retriever ─────────
        if not context_docs:
            return FullContextRetriever(raw_text="")

        # ── Step 1: Count total tokens ────────────────────────────────────────
        # Concatenate all document texts with blank lines between them.
        # "\n\n".join(...) — join strings with two newlines (paragraph separator).
        all_text: str = "\n\n".join(doc["content"] for doc in context_docs)

        # Count tokens using the same tiktoken encoding that the embedding model uses.
        total_tokens: int = self._count_tokens(all_text)
        print(f"Total context tokens: {total_tokens}")

        # ── Step 2: Branch on token count ─────────────────────────────────────
        if total_tokens < self._config.token_threshold:
            # Documents are small enough to fit directly in the GPT-4o system
            # prompt without chunking or embedding — fast and cheap.
            print(
                f"Under {self._config.token_threshold} tokens — "
                "using full context (no embedding)\n"
            )
            # FullContextRetriever — just returns the raw text every time.
            return FullContextRetriever(raw_text=all_text)

        # ── Step 3: Chunk all documents ───────────────────────────────────────
        print(
            f"{total_tokens} tokens exceed threshold — "
            "switching to RAG mode (chunking + embedding)...\n"
        )

        # TextChunker — splits each document into overlapping token-bounded chunks.
        chunker = TextChunker(
            encoding_name=self._config.embedding_encoding,  # tiktoken encoding name
            target_tokens=self._config.target_chunk_tokens, # desired tokens per chunk
            overlap_tokens=self._config.overlap_tokens,     # overlap between chunks
        )

        # Collect chunks from all documents into one flat list.
        all_chunks: list = []
        for doc in context_docs:
            # chunker.chunk() returns a list of chunk dicts; extend adds them all.
            all_chunks.extend(chunker.chunk(doc["filename"], doc["content"]))

        print(f"Created {len(all_chunks)} chunks — embedding...")

        # ── Step 4: Embed all chunks in one API call ──────────────────────────
        # Create the OpenAI client using the API key from config.
        openai_client = OpenAI(api_key=self._config.openai_api_key)

        # EmbeddingService — wraps the embeddings API call.
        embedding_service = EmbeddingService(
            client=openai_client,
            model=self._config.embed_model,
        )

        # Extract just the text content from each chunk dict for the API call.
        # List comprehension: [chunk["content"] for chunk in all_chunks]
        texts: list = [chunk["content"] for chunk in all_chunks]

        # Embed all chunk texts in one batched API call (more efficient than one call per chunk).
        vectors: list = embedding_service.embed(texts)

        # ── Step 5: Populate the vector store ─────────────────────────────────
        # VectorStore — in-memory Qdrant collection; stores vectors + metadata.
        vector_store = VectorStore(
            dim=self._config.embed_dim,              # vector dimensionality
            collection_name=self._config.collection_name,  # Qdrant collection name
        )
        # Upsert all chunk vectors with their metadata in one batch.
        vector_store.upsert(all_chunks, vectors)
        print(f"Embedded and indexed {len(all_chunks)} chunks\n")

        # ── Step 6: Optionally set up Cohere reranker ─────────────────────────
        cohere_client = None  # None — reranking disabled unless COHERE_API_KEY is set

        if self._config.cohere_api_key:
            # Lazy import — only import cohere if the key is configured.
            import cohere  # cohere — Cohere Python SDK for reranking

            # cohere.ClientV2(...) — create the v2 API client with the API key.
            cohere_client = cohere.ClientV2(api_key=self._config.cohere_api_key)
            print("Cohere reranking enabled\n")
        else:
            print("COHERE_API_KEY not set — reranking disabled\n")

        # ── Step 7: Return the assembled HybridRetriever ─────────────────────
        return HybridRetriever(
            chunks=all_chunks,                    # all document chunks (for BM25 index)
            embedding_service=embedding_service,  # used to embed queries at retrieval time
            vector_store=vector_store,            # pre-populated Qdrant collection
            cohere_client=cohere_client,          # None if reranking is disabled
        )

    def _count_tokens(self, text: str) -> int:
        # tiktoken.get_encoding(...) — retrieve the tokenizer for the configured encoding.
        enc = tiktoken.get_encoding(self._config.embedding_encoding)
        # enc.encode(text) — convert the text string to a list of integer token IDs.
        # len(...) — number of tokens.
        return len(enc.encode(text))
