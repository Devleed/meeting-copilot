# =============================================================================
# COMPONENT: text_chunker.py
#
# Splits a long text document into overlapping token-sized chunks suitable
# for embedding and vector-store retrieval.
#
# Single Responsibility: given a filename and its text content, produce a list
# of chunk dicts [{filename, content}]. Nothing else — no embedding, no storage,
# no config reading. Token counting and overlap logic live entirely here.
#
# Why chunk at all?
#   OpenAI's embedding models have a token limit per call (~8 192 for
#   text-embedding-3-small). More importantly, embedding a 20-page document
#   as one vector averages out all its meaning; retrieving sub-sections
#   (chunks) by similarity gives GPT-4o more relevant, targeted context.
#
# Why overlap?
#   A sentence split exactly at a chunk boundary loses half its meaning on
#   each side. Carrying the last OVERLAP_TOKENS from chunk N into chunk N+1
#   ensures no key phrase is silently cut in half.
#
# SOLID notes:
#   S — sole job: tokenise and split text into overlapping chunks.
#   O — chunk size and overlap are injected; no constants are hard-coded.
#   D — depends on tiktoken (a library), but that dependency is injected via
#       the encoding name string so tests can substitute a lightweight encoder.
# =============================================================================

import tiktoken  # tiktoken — OpenAI's fast tokenizer; converts text ↔ token-ID lists


class TextChunker:
    """
    Splits a document into overlapping token-bounded chunks for RAG retrieval.

    Responsibility: accept a block of text and a filename, partition the text
    into chunks that respect a maximum token budget, and carry a small overlap
    from each chunk into the next to preserve cross-boundary context.

    Attributes (set at construction time, never mutated):
        _encoding : tiktoken.Encoding
            The tokenizer used for token counting and decode.
        _target_tokens : int
            Approximate maximum tokens per chunk.
        _overlap_tokens : int
            Number of tokens from the tail of chunk N to prepend to chunk N+1.
    """

    def __init__(
        self,
        encoding_name: str,   # encoding_name — tiktoken encoding id, e.g. "cl100k_base"
        target_tokens: int,   # target_tokens  — desired max tokens per chunk
        overlap_tokens: int,  # overlap_tokens — tail-of-chunk overlap size in tokens
    ) -> None:
        # tiktoken.get_encoding(encoding_name) — loads (or retrieves from cache) the
        # tokenizer object for the named encoding. Encodings map strings to lists of
        # integer token IDs. "cl100k_base" is used by GPT-4 and text-embedding-3-*.
        self._encoding = tiktoken.get_encoding(encoding_name)

        # Store the sizing constants as instance attributes for use in chunk().
        self._target_tokens: int = target_tokens
        self._overlap_tokens: int = overlap_tokens

    def chunk(self, filename: str, text: str) -> list:
        """
        Split `text` into overlapping token-bounded chunks.

        Parameters
        ----------
        filename : str
            Source filename — stored in each chunk dict so the retriever can
            display "[Source: notes.pdf]" next to retrieved passages.
        text : str
            Full document text to be split.

        Returns
        -------
        list[dict]
            List of dicts: [{"filename": str, "content": str}, ...]
            Each dict represents one chunk of the document.
        """
        # Split the document on blank lines (paragraph boundaries).
        #
        # text.split("\n\n") — split on two consecutive newlines, the conventional
        #   paragraph separator in plain text and Markdown.
        #   In TS: text.split(/\n\n/)
        #
        # " ".join(p.split()) — for each paragraph p:
        #   p.split() with no argument splits on ANY whitespace (spaces, tabs, \n)
        #   and discards empty strings, collapsing internal whitespace runs.
        #   " ".join(...) rejoins the words with single spaces → clean flat string.
        #   Effect: normalise multi-line wrapped paragraphs into one line.
        #
        # if p.strip() — discard paragraphs that are entirely whitespace.
        #   p.strip() returns "" for blank paragraphs; empty strings are falsy in Python.
        paragraphs: list = [
            " ".join(p.split())        # normalise internal whitespace
            for p in text.split("\n\n")  # iterate over paragraphs
            if p.strip()               # skip blank paragraphs
        ]

        chunks: list = []        # final accumulator — list of completed chunk dicts
        chunk_paras: list = []   # paragraphs being built for the current chunk
        chunk_tokens: int = 0    # running token count for the current chunk

        for para in paragraphs:
            # Count how many tokens this paragraph alone would consume.
            # self._encoding.encode(para) — returns list of int token IDs.
            # len(...) — number of tokens.
            para_tokens: int = len(self._encoding.encode(para))

            # Decide whether adding this paragraph would overflow the current chunk.
            # chunk_paras — truthy (non-empty list) only after we've accumulated
            # at least one paragraph; prevents emitting an empty chunk when the
            # very first paragraph already exceeds target_tokens.
            if chunk_tokens + para_tokens > self._target_tokens and chunk_paras:
                # ── Seal the current chunk ─────────────────────────────────
                chunk_content: str = " ".join(chunk_paras)   # join paragraphs into one string
                # Append the completed chunk dict to the results list.
                chunks.append({"filename": filename, "content": chunk_content})

                # ── Compute overlap tail ───────────────────────────────────
                # Re-encode the just-flushed chunk to get its token-ID list.
                all_tokens: list = self._encoding.encode(chunk_content)

                # Take the last _overlap_tokens IDs.
                # Negative slice: all_tokens[-50:] = last 50 elements.
                # Safety guard: if the chunk is shorter than overlap_tokens, take all of it.
                overlap_slice: list = (
                    all_tokens[-self._overlap_tokens:]   # last N token IDs
                    if len(all_tokens) >= self._overlap_tokens   # guard: chunk is long enough
                    else all_tokens                      # chunk is shorter than overlap; take all
                )

                # self._encoding.decode(token_ids) — convert token IDs back to a string.
                overlap_text: str = self._encoding.decode(overlap_slice)

                # Start the next chunk with [overlap tail, current paragraph].
                # The overlap text anchors the new chunk to the previous one semantically.
                chunk_paras = [overlap_text, para]
                # Recount tokens for the new starting state.
                chunk_tokens = len(self._encoding.encode(overlap_text)) + para_tokens
            else:
                # Paragraph fits within the budget — add it to the current chunk.
                chunk_paras.append(para)   # list.append — add element to end of list
                chunk_tokens += para_tokens  # += shorthand for chunk_tokens = chunk_tokens + para_tokens

        # After the loop: chunk_paras holds the final partial chunk (paragraphs
        # that never triggered a flush because they never exceeded target_tokens).
        # "if chunk_paras" — truthy when the list is non-empty; emit the last chunk.
        if chunk_paras:
            chunks.append({"filename": filename, "content": " ".join(chunk_paras)})

        return chunks
