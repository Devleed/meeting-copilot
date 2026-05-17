# =============================================================================
# COMPONENT: embedding_service.py
#
# Wraps the OpenAI Embeddings API to convert text strings into dense float
# vectors (embeddings). These vectors encode semantic meaning — similar texts
# produce vectors that are close in cosine distance, which is what enables
# semantic (vector) search in the RAG pipeline.
#
# Single Responsibility: call the OpenAI embeddings endpoint and return
# vectors. Nothing else — no chunking, no storage, no retrieval logic.
#
# SOLID notes:
#   S — sole job: produce embedding vectors from text strings.
#   O — the model name is injected; switching models requires no code change.
#   D — depends on the OpenAI client abstraction injected at construction,
#       not on a globally imported client instance.
# =============================================================================


class EmbeddingService:
    """
    Converts text strings into embedding vectors using the OpenAI API.

    Responsibility: encapsulate the details of calling
    client.embeddings.create(...) and extracting the float vectors from the
    response, so callers never need to know the response structure.

    Usage:
        svc = EmbeddingService(openai_client, "text-embedding-3-small")
        vectors = svc.embed(["Hello world", "Meeting notes"])
        query_vec = svc.embed_one("What is the project status?")
    """

    def __init__(
        self,
        client,        # client — an OpenAI client instance (openai.OpenAI)
        model: str,    # model  — embedding model name, e.g. "text-embedding-3-small"
    ) -> None:
        # Store the injected dependencies as private attributes.
        # Private by convention (leading _) — not truly private in Python.
        self._client = client   # the OpenAI client used for all embedding calls
        self._model = model     # the model string sent in every API request

    def embed(self, texts: list) -> list:
        """
        Embed a batch of text strings in a single API call.

        Batching is more efficient than calling the API once per text because
        it reduces HTTP round-trips and is billed at the same rate.

        Parameters
        ----------
        texts : list[str]
            List of strings to embed. All strings are embedded in one request.

        Returns
        -------
        list[list[float]]
            List of embedding vectors, one per input string, in the same order.
        """
        # self._client.embeddings.create(...) — POST to the OpenAI embeddings endpoint.
        # model= — the embedding model to use (e.g. "text-embedding-3-small").
        # input= — list of strings; the API embeds all of them in one call.
        # Returns an EmbeddingResponse object with a .data attribute.
        response = self._client.embeddings.create(
            model=self._model,   # which embedding model to call
            input=texts,         # list of strings to embed
        )

        # response.data — list of Embedding objects, one per input string, same order.
        # d.embedding — the float vector for one input string (list of floats).
        # List comprehension: [expr for var in iterable] — builds a new list
        # by evaluating expr for every element var in iterable.
        # In TS: response.data.map(d => d.embedding)
        return [d.embedding for d in response.data]

    def embed_one(self, text: str) -> list:
        """
        Embed a single text string and return its vector.

        Convenience wrapper around embed() for the common query-time case
        where only one string needs to be embedded.

        Parameters
        ----------
        text : str
            The query or document string to embed.

        Returns
        -------
        list[float]
            A single embedding vector.
        """
        # self.embed([text]) — wrap text in a list (API always takes a list).
        # [0] — index 0 returns the first (and only) vector from the result list.
        return self.embed([text])[0]
