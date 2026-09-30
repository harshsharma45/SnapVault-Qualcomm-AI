"""
SnapVault — RAG Engine (rag_engine.py)
======================================
Multi-format document ingestion, semantic chunking, and dependency-free
vector search using all-MiniLM-L6-v2 embeddings + pure numpy cosine
similarity.

Supports:
  - PDF files (via pypdf) with page-level source metadata
  - CSV files (via pandas) with row-level source metadata
  - Plain text files (.txt)

All chunks carry strict source tags:
  [Source: <filename>, Page: <p>]   — for PDFs
  [Source: <filename>, Row: <r>]    — for CSVs
  [Source: <filename>]              — for plain text
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """A single chunk of text with source provenance."""
    text: str
    source_file: str
    page: Optional[int] = None
    row: Optional[int] = None
    chunk_index: int = 0

    @property
    def source_tag(self) -> str:
        if self.page is not None:
            return f"[Source: {self.source_file}, Page: {self.page}]"
        elif self.row is not None:
            return f"[Source: {self.source_file}, Row: {self.row}]"
        return f"[Source: {self.source_file}]"

    @property
    def tagged_text(self) -> str:
        return f"{self.text} {self.source_tag}"


@dataclass
class SearchResult:
    """A single vector search result with score."""
    chunk: DocumentChunk
    score: float


@dataclass
class RAGContext:
    """Aggregated context for prompt grounding."""
    results: list  # list[SearchResult]
    query: str
    search_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Embedding Model Wrapper
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Wraps sentence-transformers for embedding generation.
    Falls back to a lightweight deterministic hash-based embedding if
    sentence-transformers is unavailable (for zero-dependency smoke tests).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = None
        self._model_name = model_name
        self._dim = 384  # all-MiniLM-L6-v2 produces 384-d vectors
        self._fallback = False

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
            # Use new API name (v6+); fall back to legacy name for older versions
            if hasattr(self._model, "get_embedding_dimension"):
                self._dim = self._model.get_embedding_dimension()
            else:
                self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"✅ Embedding model loaded — dimension: {self._dim}"
            )
        except Exception as exc:
            logger.warning(
                f"sentence-transformers unavailable ({exc}). "
                "Using deterministic hash embeddings for development."
            )
            self._fallback = True

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def is_fallback(self) -> bool:
        return self._fallback

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings to a (N, dim) float32 matrix.
        """
        if self._model is not None:
            return self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        # Deterministic fallback: reproducible hash-based vectors
        return np.array([self._hash_embed(t) for t in texts], dtype=np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        """Generate a deterministic unit-norm vector from text content."""
        import hashlib
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------------------
# Document Parsers
# ---------------------------------------------------------------------------

def parse_pdf(file_path: str, filename: str) -> list[DocumentChunk]:
    """Parse a PDF file, returning one chunk per page."""
    from pypdf import PdfReader

    chunks = []
    try:
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                chunks.append(DocumentChunk(
                    text=text.strip(),
                    source_file=filename,
                    page=page_num,
                ))
    except Exception as exc:
        logger.error(f"Failed to parse PDF '{filename}': {exc}")

    logger.info(f"PDF '{filename}': extracted {len(chunks)} pages")
    return chunks


def parse_pdf_bytes(data: bytes, filename: str) -> list[DocumentChunk]:
    """Parse a PDF from in-memory bytes (Streamlit upload)."""
    import io
    from pypdf import PdfReader

    chunks = []
    try:
        reader = PdfReader(io.BytesIO(data))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                chunks.append(DocumentChunk(
                    text=text.strip(),
                    source_file=filename,
                    page=page_num,
                ))
    except Exception as exc:
        logger.error(f"Failed to parse PDF bytes '{filename}': {exc}")

    return chunks


def parse_csv(file_path: str, filename: str) -> list[DocumentChunk]:
    """Parse a CSV file, converting each row to structured text."""
    import pandas as pd

    chunks = []
    try:
        df = pd.read_csv(file_path)
        for row_idx, row in df.iterrows():
            row_num = int(row_idx) + 1
            parts = [f"{col}={val}" for col, val in row.items() if pd.notna(val)]
            text = f"Row {row_num}: " + ", ".join(parts)
            chunks.append(DocumentChunk(
                text=text,
                source_file=filename,
                row=row_num,
            ))
    except Exception as exc:
        logger.error(f"Failed to parse CSV '{filename}': {exc}")

    logger.info(f"CSV '{filename}': extracted {len(chunks)} rows")
    return chunks


def parse_csv_bytes(data: bytes, filename: str) -> list[DocumentChunk]:
    """Parse a CSV from in-memory bytes."""
    import io
    import pandas as pd

    chunks = []
    try:
        df = pd.read_csv(io.BytesIO(data))
        for row_idx, row in df.iterrows():
            row_num = int(row_idx) + 1
            parts = [f"{col}={val}" for col, val in row.items() if pd.notna(val)]
            text = f"Row {row_num}: " + ", ".join(parts)
            chunks.append(DocumentChunk(
                text=text,
                source_file=filename,
                row=row_num,
            ))
    except Exception as exc:
        logger.error(f"Failed to parse CSV bytes '{filename}': {exc}")

    return chunks


def parse_text(file_path: str, filename: str) -> list[DocumentChunk]:
    """Parse a plain text file."""
    chunks = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        if text:
            chunks.append(DocumentChunk(
                text=text,
                source_file=filename,
            ))
    except Exception as exc:
        logger.error(f"Failed to parse text '{filename}': {exc}")

    logger.info(f"TXT '{filename}': extracted {len(chunks)} blocks")
    return chunks


def parse_text_bytes(data: bytes, filename: str) -> list[DocumentChunk]:
    """Parse plain text from in-memory bytes."""
    text = data.decode("utf-8", errors="replace").strip()
    if text:
        return [DocumentChunk(text=text, source_file=filename)]
    return []


# ---------------------------------------------------------------------------
# Semantic Chunker
# ---------------------------------------------------------------------------

def semantic_chunk(
    raw_chunks: list[DocumentChunk],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """
    Split raw document chunks into ~chunk_size character segments
    with overlap, preserving source metadata.
    """
    result = []
    idx = 0
    for raw in raw_chunks:
        text = raw.text
        start = 0
        while start < len(text):
            end = start + chunk_size
            segment = text[start:end]

            # Try to break at a sentence boundary
            if end < len(text):
                last_period = segment.rfind(". ")
                last_newline = segment.rfind("\n")
                break_point = max(last_period, last_newline)
                if break_point > chunk_size * 0.3:
                    segment = text[start:start + break_point + 1]
                    end = start + break_point + 1

            if segment.strip():
                result.append(DocumentChunk(
                    text=segment.strip(),
                    source_file=raw.source_file,
                    page=raw.page,
                    row=raw.row,
                    chunk_index=idx,
                ))
                idx += 1

            start = max(start + 1, end - overlap)

    return result


# ---------------------------------------------------------------------------
# Main RAG Engine
# ---------------------------------------------------------------------------

class SnapVaultRAG:
    """
    End-to-end RAG pipeline: ingest → chunk → embed → search → ground.

    Uses pure numpy cosine similarity for vector search — no heavy
    C++ vector stores, guaranteeing zero wheel build failures on ARM.
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self._embedder = EmbeddingModel(embedding_model_name)
        self._chunks: list[DocumentChunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self._document_registry: dict[str, int] = {}  # filename → chunk count

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def document_count(self) -> int:
        return len(self._document_registry)

    @property
    def document_registry(self) -> dict[str, int]:
        return dict(self._document_registry)

    @property
    def is_embedding_fallback(self) -> bool:
        return self._embedder.is_fallback

    # -----------------------------------------------------------------
    # Ingestion — File path based
    # -----------------------------------------------------------------

    def ingest_file(self, file_path: str) -> int:
        """Ingest a single file by path. Returns number of chunks created."""
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            raw = parse_pdf(file_path, filename)
        elif ext == ".csv":
            raw = parse_csv(file_path, filename)
        elif ext == ".txt":
            raw = parse_text(file_path, filename)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return 0

        return self._process_raw_chunks(raw, filename)

    # -----------------------------------------------------------------
    # Ingestion — Bytes based (for Streamlit uploads)
    # -----------------------------------------------------------------

    def ingest_bytes(self, data: bytes, filename: str) -> int:
        """Ingest file from in-memory bytes. Returns number of chunks."""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            raw = parse_pdf_bytes(data, filename)
        elif ext == ".csv":
            raw = parse_csv_bytes(data, filename)
        elif ext == ".txt":
            raw = parse_text_bytes(data, filename)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return 0

        return self._process_raw_chunks(raw, filename)

    def _process_raw_chunks(self, raw: list[DocumentChunk], filename: str) -> int:
        """Chunk, embed, and add to the index."""
        chunks = semantic_chunk(raw)
        if not chunks:
            return 0

        # Embed
        texts = [c.tagged_text for c in chunks]
        new_embeddings = self._embedder.encode(texts)

        # Append to index
        self._chunks.extend(chunks)
        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        self._document_registry[filename] = (
            self._document_registry.get(filename, 0) + len(chunks)
        )

        logger.info(
            f"Ingested '{filename}': {len(chunks)} chunks, "
            f"total index: {len(self._chunks)} chunks"
        )
        return len(chunks)

    # -----------------------------------------------------------------
    # Clear
    # -----------------------------------------------------------------

    def clear(self) -> None:
        """Clear all indexed documents and embeddings."""
        self._chunks.clear()
        self._embeddings = None
        self._document_registry.clear()
        logger.info("Knowledge base cleared.")

    # -----------------------------------------------------------------
    # Vector Search
    # -----------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> RAGContext:
        """
        Perform cosine similarity search over indexed chunks.

        Returns a RAGContext with top-k results and timing.
        """
        if not self._chunks or self._embeddings is None:
            return RAGContext(results=[], query=query, search_time_ms=0.0)

        t_start = time.perf_counter()

        # Encode the query
        query_vec = self._embedder.encode([query])  # shape: (1, dim)

        # Cosine similarity (embeddings are already L2-normalized)
        similarities = np.dot(self._embeddings, query_vec.T).flatten()

        # Top-k indices
        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = [
            SearchResult(
                chunk=self._chunks[i],
                score=float(similarities[i]),
            )
            for i in top_indices
            if similarities[i] > 0.0
        ]

        t_end = time.perf_counter()
        search_time = (t_end - t_start) * 1000

        return RAGContext(
            results=results,
            query=query,
            search_time_ms=round(search_time, 2),
        )

    # -----------------------------------------------------------------
    # Prompt Grounding
    # -----------------------------------------------------------------

    def build_grounded_prompt(
        self,
        query: str,
        context: Optional[RAGContext] = None,
        top_k: int = 5,
    ) -> tuple[str, RAGContext]:
        """
        Build a grounded prompt that constrains the LLM to answer
        strictly from retrieved context with mandatory citations.

        Returns (prompt_text, rag_context).
        """
        if context is None:
            context = self.search(query, top_k=top_k)

        if not context.results:
            prompt = (
                "You are SnapVault, a confidential document copilot running "
                "100% on-device with zero cloud connectivity.\n\n"
                "The user has asked a question but no relevant documents have "
                "been uploaded or no matching context was found.\n\n"
                "Respond politely that you need relevant documents to be "
                "uploaded before you can answer questions. Do NOT fabricate "
                "information.\n\n"
                f"Question: {query}"
            )
            return prompt, context

        # Build context block
        context_block = []
        for i, r in enumerate(context.results, 1):
            context_block.append(
                f"[Chunk {i} | Score: {r.score:.3f}] "
                f"{r.chunk.tagged_text}"
            )

        context_text = "\n\n".join(context_block)

        prompt = (
            "You are SnapVault, a confidential document copilot running "
            "100% on-device with zero cloud connectivity. You must:\n"
            "1. Answer ONLY using the provided context below.\n"
            "2. Cite every claim with the source tag "
            "(e.g., [Source: file.pdf, Page: 3]).\n"
            "3. If the context does not contain enough information to answer, "
            "say so explicitly. Do NOT fabricate information.\n"
            "4. For numeric questions, extract exact numbers from context.\n"
            "5. Be concise and professional.\n\n"
            "--- Context ---\n"
            f"{context_text}\n"
            "--- End Context ---\n\n"
            f"Question: {query}\n\n"
            "Answer (with citations):"
        )

        return prompt, context

    # -----------------------------------------------------------------
    # Convenience: One-click audit prompts
    # -----------------------------------------------------------------

    def build_metrics_prompt(self) -> tuple[str, RAGContext]:
        """Build a prompt for extracting key metrics and numeric totals."""
        query = (
            "Extract all key metrics, numeric totals, financial figures, "
            "percentages, and quantitative data from the documents. "
            "Present them in a structured list with exact values."
        )
        return self.build_grounded_prompt(query, top_k=8)

    def build_risk_prompt(self) -> tuple[str, RAGContext]:
        """Build a prompt for identifying liabilities and risk factors."""
        query = (
            "Identify all liabilities, risk factors, warnings, concerns, "
            "declining trends, regulatory issues, and potential problems "
            "mentioned in the documents. Categorize each risk by severity."
        )
        return self.build_grounded_prompt(query, top_k=8)

    def build_brief_prompt(self) -> tuple[str, RAGContext]:
        """Build a prompt for generating an executive brief."""
        query = (
            "Generate a concise executive brief summarizing the key points, "
            "findings, decisions, and action items from the documents. "
            "Structure it with clear headings and bullet points."
        )
        return self.build_grounded_prompt(query, top_k=10)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("SnapVault — RAG Engine Smoke Test")
    print("=" * 60)

    # Create a temporary test document
    test_text = """Quarterly Financial Report — Q3 2025

Revenue reached $12.5 million, representing an 18% increase year-over-year.
Operating expenses were $8.2 million, primarily driven by R&D expansion.
Net income was $2.1 million, with a profit margin of 16.8%.

The company faces potential risks including:
- Supply chain disruptions in the semiconductor sector
- Increased competition from cloud-based alternatives
- Regulatory changes in data privacy (GDPR, CCPA compliance costs)

Employee headcount grew to 245, up from 198 in Q2.
Customer retention rate remains strong at 94.5%.
"""

    # Write temp file
    tmp_dir = tempfile.mkdtemp()
    tmp_file = os.path.join(tmp_dir, "q3_report.txt")
    with open(tmp_file, "w") as f:
        f.write(test_text)

    # Test RAG pipeline
    rag = SnapVaultRAG()
    n_chunks = rag.ingest_file(tmp_file)
    print(f"\n✅ Ingested {n_chunks} chunks from test document")
    print(f"   Total chunks in index: {rag.chunk_count}")
    print(f"   Documents registered: {rag.document_count}")
    print(f"   Embedding fallback: {rag.is_embedding_fallback}")

    # Test search
    query = "What was the revenue in Q3?"
    context = rag.search(query, top_k=3)
    print(f"\n🔍 Search query: '{query}'")
    print(f"   Results: {len(context.results)}")
    print(f"   Search time: {context.search_time_ms:.1f} ms")
    for r in context.results:
        print(f"   [{r.score:.3f}] {r.chunk.tagged_text[:80]}...")

    # Test grounded prompt
    prompt, ctx = rag.build_grounded_prompt(query)
    print(f"\n📝 Grounded prompt length: {len(prompt)} chars")
    print(f"   Context chunks used: {len(ctx.results)}")

    # Test audit prompts
    for name, fn in [("Metrics", rag.build_metrics_prompt),
                     ("Risks", rag.build_risk_prompt),
                     ("Brief", rag.build_brief_prompt)]:
        p, c = fn()
        print(f"   {name} prompt: {len(p)} chars, {len(c.results)} chunks")

    # Test clear
    rag.clear()
    assert rag.chunk_count == 0
    print("\n🗑️  Knowledge base cleared successfully")

    # Cleanup
    os.unlink(tmp_file)
    os.rmdir(tmp_dir)

    print("\n✅ RAG engine smoke test PASSED")
