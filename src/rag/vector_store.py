# =============================================================================
# COMPONENT: vector_store.py
#
# Wraps the in-memory Qdrant vector database. Qdrant stores embedding vectors
# alongside metadata (the chunk text and source filename) and supports fast
# approximate nearest-neighbour (ANN) search by cosine similarity.
#
# "In-memory" means the store lives only for the duration of the process.
# No persistence across restarts, which is intentional — context is loaded
# fresh at every copilot session.
#
# Single Responsibility: own the lifecycle of the Qdrant collection (create,
# upsert, query). No chunking, no embedding, no retrieval fusion — just raw
# vector storage and lookup.
#
# SOLID notes:
#   S — sole job: store vectors and return nearest neighbours.
#   O — collection name and vector size are injected; swapping Qdrant for
#       another backend only requires changing this class.
#   D — depends on a QdrantClient instance injected at construction,
#       not on a globally imported client.
# =============================================================================

# QdrantClient — the main Qdrant Python client.
# Distance, VectorParams — enums/config classes for collection creation.
# PointStruct — data class representing one stored vector with metadata.
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class VectorStore:
    """
    In-memory Qdrant collection: store embedding vectors and query by similarity.

    Responsibility: manage the Qdrant collection lifecycle — create it once,
    accept upsert batches (vectors + payloads), and answer nearest-neighbour
    queries. Callers supply pre-computed float vectors; this class never calls
    the embedding API directly.

    Usage:
        store = VectorStore(dim=1536, collection_name="context_chunks")
        store.upsert(chunks, vectors)
        results = store.query(query_vector, limit=10)
    """

    def __init__(
        self,
        dim: int,                # dim             — dimensionality of each vector
        collection_name: str,    # collection_name — name for the Qdrant collection
    ) -> None:
        # QdrantClient(":memory:") — spin up an in-process, in-memory Qdrant instance.
        # No network call, no port, no persistence. ":memory:" is a special string
        # that tells Qdrant to skip disk I/O and store everything in RAM.
        self._client = QdrantClient(":memory:")

        # Store the collection name so all methods reference the same collection.
        self._collection_name: str = collection_name

        # Create the collection in Qdrant immediately at construction time.
        # After this call, the collection exists and can accept upserts.
        self._client.create_collection(
            # collection_name= — the string identifier for this collection.
            collection_name=self._collection_name,

            # vectors_config= — defines vector properties for the collection.
            # VectorParams(size=dim, distance=Distance.COSINE):
            #   size=dim        — each vector must have exactly `dim` floats.
            #   distance=COSINE — use cosine similarity for nearest-neighbour search.
            #                     Cosine measures the angle between vectors, which
            #                     is the right metric for normalised embeddings.
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(self, chunks: list, vectors: list) -> None:
        """
        Insert or update all chunk vectors in the collection.

        Parameters
        ----------
        chunks : list[dict]
            List of chunk dicts: [{"filename": str, "content": str}, ...]
            Provides the metadata stored alongside each vector.
        vectors : list[list[float]]
            Parallel list of embedding vectors, one per chunk.
            len(vectors) must equal len(chunks).
        """
        # Build a list of PointStruct objects — Qdrant's representation of one
        # stored vector with an integer ID and a metadata payload.
        #
        # range(len(all_chunks)) — produces integers 0, 1, 2, ... len-1.
        # In TS: Array.from({ length: chunks.length }, (_, i) => ...)
        points: list = [
            PointStruct(
                id=i,                           # unique integer ID within the collection
                vector=vectors[i],              # the float embedding vector for this chunk
                payload={                        # arbitrary metadata stored alongside the vector
                    "filename": chunks[i]["filename"],  # source document filename
                    "content":  chunks[i]["content"],   # raw text of the chunk
                },
            )
            # i iterates over 0 … len(chunks)-1; index-access both parallel lists.
            for i in range(len(chunks))
        ]

        # qdrant.upsert(...) — insert all points in one batch call.
        # "upsert" = insert if the ID is new, update if the ID already exists.
        self._client.upsert(
            collection_name=self._collection_name,   # which collection to write to
            points=points,                           # the list of PointStructs to store
        )

    def query(self, vector: list, limit: int) -> list:
        """
        Return the `limit` most similar chunks to the given query vector.

        Parameters
        ----------
        vector : list[float]
            Query embedding vector. Must have the same dimensionality as stored vectors.
        limit : int
            Maximum number of results to return.

        Returns
        -------
        list[ScoredPoint]
            Qdrant ScoredPoint objects. Each has:
              .payload["filename"] — source filename
              .payload["content"]  — chunk text
              .score               — cosine similarity (higher = more similar)
        """
        # query_points(...) — perform approximate nearest-neighbour search.
        # Returns a QueryResponse object; .points is the list of ScoredPoint results.
        return self._client.query_points(
            collection_name=self._collection_name,   # which collection to search
            query=vector,                            # the query vector to compare against
            limit=limit,                             # return at most this many results
        ).points  # .points — unwrap the list of ScoredPoint objects from the response wrapper
