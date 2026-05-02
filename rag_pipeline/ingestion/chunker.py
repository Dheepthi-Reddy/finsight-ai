"""
Splits SEC filing documents into overlapping text chunks for embedding.

Why overlap matters
-------------------
A sentence or regulatory clause can span the boundary between two adjacent
chunks. Without overlap, the embedding model sees an incomplete thought at the
tail of one chunk and a dangling continuation at the head of the next — neither
chunk encodes the full meaning, so neither will be retrieved when a user asks
about that clause.

Overlap of 64 characters (≈1–2 sentences) ensures that boundary content is
fully represented in at least one of the two neighbouring chunks. The tradeoff
is a ~12% increase in total chunk count (64/512), which is acceptable given
the retrieval quality improvement for compliance Q&A where precision matters.

Why RecursiveCharacterTextSplitter
-----------------------------------
It tries to split on paragraph breaks (\n\n) first, then line breaks (\n),
then spaces, and only falls back to hard character cuts as a last resort.
This keeps regulatory paragraphs and risk-factor items intact wherever
possible, instead of cutting mid-sentence as a fixed-stride splitter would.

Usage
-----
    from rag_pipeline.ingestion.chunker import chunk_documents

    from rag_pipeline.ingestion.sec_loader import load_filings
    docs   = load_filings()
    chunks = chunk_documents(docs)
    # [{"text": "...", "metadata": {"ticker": "JPM", ...}}, ...]
"""

from __future__ import annotations

from typing import Any

# LangChain reorganised its package structure in v0.2:
#   - Before v0.2: langchain.text_splitter
#   - v0.2+:       langchain_text_splitters (separate package)
# We try the new location first and fall back so this works with any version
# that has langchain in requirements.txt.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]

# ── Chunking parameters ───────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # characters — fits comfortably within most embedding model
                      # context windows (e.g. all-MiniLM-L6-v2 has a 256-token limit,
                      # ~384 chars; 512 chars gives ~1-token headroom with short words)
CHUNK_OVERLAP = 64    # characters — ~12% of chunk size, covering 1–2 boundary sentences

MIN_CHUNK_LEN = 50    # discard chunks shorter than this — they are typically
                      # page headers, exhibit labels, or lone numbers that carry
                      # no semantic content and would pollute retrieval results


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_length: int = MIN_CHUNK_LEN,
) -> list[dict[str, Any]]:
    """
    Split a list of filing documents into overlapping text chunks.

    Parameters
    ----------
    documents : list[dict]
        Each dict must have keys: text, ticker, filing_type, path.
        Produced by rag_pipeline.ingestion.sec_loader.load_filings().
    chunk_size : int
        Maximum characters per chunk (default 512).
    chunk_overlap : int
        Characters shared between consecutive chunks (default 64).
    min_length : int
        Chunks shorter than this are dropped (default 50).

    Returns
    -------
    list[dict]
        Each dict: {"text": str, "metadata": dict}
        metadata keys: ticker, filing_type, source, chunk_index, chunk_count
        chunk_index is 0-based within each source document.
        chunk_count is the total number of chunks kept for that document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Separator priority: try paragraph breaks before line breaks before
        # word boundaries before hard cuts. Keeps regulatory items together.
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    all_chunks: list[dict[str, Any]] = []
    total_dropped = 0

    for doc in documents:
        raw_text    = doc.get("text", "")
        ticker      = doc.get("ticker", "UNKNOWN")
        filing_type = doc.get("filing_type", "UNKNOWN")
        source_path = doc.get("path", "")

        if not raw_text.strip():
            continue

        # Split the full document text into overlapping windows
        splits = splitter.split_text(raw_text)

        # Filter noise before assigning indices so chunk_index stays gapless
        kept = [s for s in splits if len(s.strip()) >= min_length]
        total_dropped += len(splits) - len(kept)

        # chunk_count lets downstream code distinguish "chunk 3 of 47" vs
        # "chunk 3 of 4" — useful for weighting recency (later chunks in a
        # 10-K tend to be financial tables, earlier ones are narrative risk text)
        chunk_count = len(kept)

        for idx, chunk_text in enumerate(kept):
            all_chunks.append({
                "text": chunk_text.strip(),
                "metadata": {
                    "ticker":       ticker,
                    "filing_type":  filing_type,
                    "source":       source_path,
                    "chunk_index":  idx,
                    "chunk_count":  chunk_count,
                },
            })

    print(
        f"Chunked {len(documents)} documents → "
        f"{len(all_chunks)} chunks kept, {total_dropped} dropped (< {min_length} chars)"
    )
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rag_pipeline.ingestion.sec_loader import load_filings

    docs   = load_filings(force_stubs=True)
    chunks = chunk_documents(docs)

    # Distribution summary
    from collections import Counter
    by_type = Counter(c["metadata"]["filing_type"] for c in chunks)
    sizes   = [len(c["text"]) for c in chunks]

    print(f"\nChunk size stats:")
    print(f"  min={min(sizes)}  max={max(sizes)}  avg={sum(sizes)//len(sizes)}")
    print(f"By filing type: {dict(by_type)}")

    # Spot-check a boundary pair to verify overlap is working
    if len(chunks) >= 2:
        a, b = chunks[0]["text"], chunks[1]["text"]
        overlap_chars = 0
        # Walk backwards from end of a, check if tail matches head of b
        for length in range(min(CHUNK_OVERLAP * 2, len(a), len(b)), 0, -1):
            if a[-length:] == b[:length]:
                overlap_chars = length
                break
        print(f"\nOverlap check (chunk 0 → 1): {overlap_chars} shared chars")
        print(f"  End of chunk 0 : ...{repr(a[-40:])}")
        print(f"  Start of chunk 1: {repr(b[:40])}...")
