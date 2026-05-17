# =============================================================================
# COMPONENT: retriever.py
#
# Context retrieval layer. Given a question from the transcription, retrieves
# the most relevant portion of the meeting context to include in the GPT-4o
# prompt.
#
# Class breakdown:
#
#   BaseRetriever      (abstract) — interface: get_context(question) → str.
#                        All retrieval strategies implement this single method.
#                        Callers (SuggestionGenerator) depend only on this
#                        abstraction, not on any specific retrieval strategy.
#
#   FullContextRetriever — trivial strategy: return the entire raw document text.
#                          Used when total tokens are below the threshold; no
#                          embedding or vector search needed.
#
#   HybridRetriever    — RAG strategy combining vector search (semantic) and
#                        BM25 keyword search, with optional Cohere reranking.
#                        Uses EmbeddingService and VectorStore injected at
#                        construction time. Satisfies the Dependency Inversion
#                        Principle: depends on those abstractions, not on the
#                        OpenAI or Qdrant clients directly.
#
# SOLID notes:
#   S — each class has one retrieval strategy and nothing else.
#   O — adding a new strategy (e.g. pure BM25) means a new subclass; no
#       existing class changes.
#   L — FullContextRetriever and HybridRetriever are fully substitutable for
#       BaseRetriever anywhere in the codebase.
#   I — BaseRetriever exposes only one method (get_context); no fat interface.
#   D — SuggestionGenerator depends on BaseRetriever, not on concrete classes.
# =============================================================================

# ABC, abstractmethod — machinery for declaring abstract base classes.
from abc import ABC, abstractmethod

# BM25Okapi — classic keyword-based ranking algorithm (Best Match 25).
# Scores documents based on term frequency and inverse document frequency.
# Complements vector search: vector search is good at semantic similarity,
# BM25 is good at exact keyword matches.
from rank_bm25 import BM25Okapi

# Type hint only — not used at runtime.
from typing import Optional


class BaseRetriever(ABC):
    """
    Abstract base class (interface) for all retrieval strategies.

    Responsibility: define the contract every retriever must fulfil.
    Given a natural-language question, return a string of context text
    to inject into the GPT-4o prompt.

    Callers (SuggestionGenerator) depend on this abstraction, so the
    retrieval strategy can be swapped (full-context ↔ RAG) without
    changing any calling code.
    """

    @abstractmethod  # @abstractmethod — every concrete subclass must implement this method
    def get_context(self, question: str) -> str:
        """
        Retrieve relevant context for the given question.

        Parameters
        ----------
        question : str
            The transcribed utterance / query to find context for.

        Returns
        -------
        str
            Context text to inject into the system prompt.
        """
        ...  # ellipsis — placeholder body; never executed on the abstract base


class FullContextRetriever(BaseRetriever):
    """
    Returns the entire raw document text as context on every query.

    Responsibility: trivial retrieval strategy used when the total token
    count of all context documents is below the embedding threshold (default
    4 000 tokens). No embedding or vector search needed — just return everything.

    Pros:  GPT-4o sees all context every time; no retrieval errors.
    Cons:  Uses more tokens per prompt; breaks down for large documents.
    """

    def __init__(self, raw_text: str) -> None:
        # raw_text — the entire concatenated text of all context documents.
        # Stored once at construction; returned verbatim on every get_context call.
        self._raw_text: str = raw_text

    def get_context(self, question: str) -> str:
        # question is intentionally unused — in full-context mode, the same text
        # is returned regardless of the specific question asked.
        # The underscore convention for unused params is also common (_, _question).
        return self._raw_text


class HybridRetriever(BaseRetriever):
    """
    Retrieves context using a hybrid of vector search and BM25 keyword search,
    with optional Cohere reranking.

    Responsibility: given a question, find the most relevant chunks from the
    pre-embedded document corpus using two complementary strategies:
      1. Vector search  — semantic similarity via OpenAI embeddings + Qdrant ANN.
      2. BM25 search    — keyword overlap via TF-IDF-style scoring.
    Results from both are merged (deduplicated) into a candidate set, then:
      - If a Cohere client is available: rerank candidates with Cohere rerank-v3.5.
      - Otherwise: take the top-K candidates as-is.

    The hybrid approach handles both semantic queries ("explain the main risk")
    and exact keyword queries ("what is the budget figure?") robustly.

    Injected dependencies (Dependency Inversion Principle):
      embedding_service — embeds the query vector at retrieval time.
      vector_store      — ANN lookup against pre-embedded chunk vectors.
      cohere_client     — optional reranker; None disables reranking.
    """

    # TOP_K : number of final chunks to return after reranking / slicing.
    TOP_K: int = 3

    # CANDIDATE_K : how many candidates to fetch from each retriever before fusion.
    # Fetching more candidates gives the reranker / deduplicator more to work with.
    CANDIDATE_K: int = 10

    def __init__(
        self,
        chunks: list,                       # chunks — list of {"filename", "content"} dicts
        embedding_service,                  # embedding_service — EmbeddingService instance
        vector_store,                       # vector_store      — VectorStore instance
        cohere_client: Optional[object],    # cohere_client     — cohere.ClientV2 or None
    ) -> None:
        # Store all injected dependencies as private attributes.
        self._chunks = chunks
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._cohere = cohere_client

        # Build the BM25 index from all chunk texts at construction time.
        # Indexing is O(n * avg_words) but done once; queries are much cheaper.
        #
        # tokenized — list of lists of lowercase word strings, one list per chunk.
        # chunk["content"].lower() — lowercase the chunk text for case-insensitive matching.
        # .split()                 — split on whitespace into a list of word tokens.
        # In TS: chunks.map(c => c.content.toLowerCase().split(/\s+/))
        tokenized: list = [chunk["content"].lower().split() for chunk in self._chunks]

        # BM25Okapi(tokenized) — build the BM25 index from the tokenized corpus.
        # "Okapi" refers to the Okapi BM25 variant (the standard one used in practice).
        self._bm25 = BM25Okapi(tokenized)

    def get_context(self, question: str) -> str:
        """
        Retrieve the most relevant chunks for the given question.

        Parameters
        ----------
        question : str
            The transcribed utterance / query to retrieve context for.

        Returns
        -------
        str
            Formatted context string: chunks separated by blank lines,
            each prefixed with "[Source: <filename>]".
        """
        # ── Step 1: Vector (semantic) search ──────────────────────────────────
        # Embed the question into a query vector using the same model as the chunks.
        query_vector: list = self._embedding_service.embed_one(question)

        # Search the Qdrant collection for the CANDIDATE_K most similar chunk vectors.
        # Returns a list of ScoredPoint objects; each has .payload with filename + content.
        vector_results: list = self._vector_store.query(query_vector, self.CANDIDATE_K)

        # ── Step 2: BM25 (keyword) search ─────────────────────────────────────
        # Tokenize the question the same way as the corpus (lowercase + split).
        tokenized_query: list = question.lower().split()

        # bm25.get_scores(tokenized_query) — returns a numpy array of BM25 scores,
        # one float per chunk in the corpus, in the same order as self._chunks.
        bm25_scores = self._bm25.get_scores(tokenized_query)

        # Sort chunk indices by score descending and take the top CANDIDATE_K.
        # range(len(bm25_scores)) — indices 0, 1, ..., len-1.
        # key=lambda i: bm25_scores[i] — sort by the BM25 score at each index.
        # reverse=True — descending order (highest score first).
        # [:self.CANDIDATE_K] — slice the top CANDIDATE_K indices.
        top_bm25_indices: list = sorted(
            range(len(bm25_scores)),          # all chunk indices as candidates
            key=lambda i: bm25_scores[i],     # score function for sorting
            reverse=True,                     # highest score first
        )[:self.CANDIDATE_K]                  # keep only the top CANDIDATE_K

        # ── Step 3: Merge and deduplicate ─────────────────────────────────────
        # Use a set to track which chunk contents have already been added,
        # so the same chunk from both vector and BM25 results is not duplicated.
        seen: set = set()        # set — O(1) membership test; tracks deduplicated content strings
        candidates: list = []    # ordered list of unique chunk dicts

        # Add vector search results first (higher-quality semantic matches take priority).
        for r in vector_results:
            # r.payload — the dict we stored when upserting: {filename, content}.
            content: str = r.payload["content"]
            # `content not in seen` — O(1) set membership check.
            if content not in seen:
                seen.add(content)   # mark this content as seen
                candidates.append({"filename": r.payload["filename"], "content": content})

        # Add BM25 results that are not already in candidates from vector search.
        for i in top_bm25_indices:
            content: str = self._chunks[i]["content"]
            if content not in seen:
                seen.add(content)
                candidates.append(self._chunks[i])

        # ── Step 4: Cohere reranking (optional) ───────────────────────────────
        if self._cohere and candidates:
            # self._cohere.rerank(...) — POST to Cohere's reranking endpoint.
            # model=     — Cohere reranking model to use.
            # query=     — the original question string.
            # documents= — list of content strings for each candidate.
            # top_n=     — how many results to return after reranking.
            rerank_response = self._cohere.rerank(
                model="rerank-v3.5",
                query=question,
                documents=[c["content"] for c in candidates],  # extract text from each candidate dict
                top_n=self.TOP_K,                              # return only the best TOP_K
            )
            # rerank_response.results — list of RerankResult objects; each has .index
            # pointing to the original position in `candidates`.
            # List comprehension: select candidates in reranked order.
            top_candidates: list = [candidates[r.index] for r in rerank_response.results]
        else:
            # No Cohere client — fall back to taking the first TOP_K candidates as-is.
            # (Vector results are already sorted by cosine similarity, so this is reasonable.)
            top_candidates = candidates[:self.TOP_K]

        # ── Step 5: Format as a readable string ───────────────────────────────
        # Build one string per chunk: "[Source: filename]\n<content>".
        # f-string with multi-line value: the [ and \n are literal characters.
        formatted_chunks: list = [
            f"[Source: {c['filename']}]\n{c['content']}"   # header + body for one chunk
            for c in top_candidates                          # iterate over the final top chunks
        ]

        # "\n\n".join(...) — separate chunks with a blank line for readability.
        return "\n\n".join(formatted_chunks)
