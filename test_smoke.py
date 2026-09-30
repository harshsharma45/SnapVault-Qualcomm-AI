"""
SnapVault - Smoke Test Suite (test_smoke.py)
=============================================
Automated verification of all core modules:
  1. Import validation (zero import errors)
  2. RAG engine pipeline (ingest, chunk, search, prompt)
  3. Inference engine (emulation mode generation + telemetry)
  4. End-to-end RAG + inference integration
"""

import os
import sys
import time
import traceback

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure the project directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []


def test(name, fn):
    """Run a test function and record the result."""
    print(f"\n{'-' * 60}")
    print(f"  TEST: {name}")
    print(f"{'-' * 60}")
    try:
        fn()
        results.append((name, PASS, ""))
        print(f"\n  {PASS}")
    except Exception as exc:
        tb = traceback.format_exc()
        results.append((name, FAIL, str(exc)))
        print(f"\n  {FAIL}: {exc}")
        print(f"  Traceback:\n{tb}")


# ===================================================================
# Test 1: Import Validation
# ===================================================================

def test_imports():
    """Verify all core modules import without errors."""
    import numpy as np
    print(f"  numpy {np.__version__}")

    import pandas as pd
    print(f"  pandas {pd.__version__}")

    try:
        import onnxruntime as ort
        print(f"  onnxruntime {ort.__version__}")
        print(f"  Available providers: {ort.get_available_providers()}")
    except ImportError:
        print(f"  {WARN} onnxruntime not installed (emulation mode)")

    from qnn_engine import SnapdragonInferenceEngine, InferenceTelemetry
    print(f"  qnn_engine imported successfully")

    from rag_engine import SnapVaultRAG, DocumentChunk, SearchResult
    print(f"  rag_engine imported successfully")


# ===================================================================
# Test 2: RAG Engine — Text Ingestion
# ===================================================================

def test_rag_text_ingestion():
    """Test ingesting a text file and verifying chunk creation."""
    from rag_engine import SnapVaultRAG

    rag = SnapVaultRAG()

    # Ingest sample document
    sample_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "sample_data", "sample_report.txt"
    )

    if os.path.exists(sample_path):
        n = rag.ingest_file(sample_path)
        print(f"  Ingested sample_report.txt: {n} chunks")
    else:
        # Create inline test data
        test_text = (
            "Revenue for Q3 was $12.5M, up 18% year-over-year.\n"
            "Operating expenses rose to $8.2M due to R&D expansion.\n"
            "Net income was $2.1M with a profit margin of 16.8%.\n"
            "The company faces supply chain risks in semiconductors.\n"
            "Employee headcount grew to 245 from 198 in Q2.\n"
            "Customer retention rate remains strong at 94.5%."
        )
        n = rag.ingest_bytes(test_text.encode("utf-8"), "test_report.txt")
        print(f"  Ingested inline test data: {n} chunks")

    assert rag.chunk_count > 0, "No chunks created"
    assert rag.document_count > 0, "No documents registered"
    print(f"  Total chunks: {rag.chunk_count}")
    print(f"  Documents: {rag.document_count}")
    print(f"  Registry: {rag.document_registry}")


# ===================================================================
# Test 3: RAG Engine — CSV Ingestion
# ===================================================================

def test_rag_csv_ingestion():
    """Test ingesting CSV data with row-level metadata."""
    from rag_engine import SnapVaultRAG

    rag = SnapVaultRAG()

    csv_data = (
        "Product,Revenue,Growth\n"
        "Widget A,$5.2M,12%\n"
        "Widget B,$3.1M,8%\n"
        "Widget C,$4.2M,22%\n"
    )

    n = rag.ingest_bytes(csv_data.encode("utf-8"), "products.csv")
    print(f"  Ingested CSV: {n} chunks")
    assert n > 0, "No chunks from CSV"
    print(f"  Total chunks: {rag.chunk_count}")


# ===================================================================
# Test 4: RAG Engine — Vector Search
# ===================================================================

def test_rag_search():
    """Test semantic search returns ranked results."""
    from rag_engine import SnapVaultRAG

    rag = SnapVaultRAG()

    # Ingest test data
    texts = [
        "Revenue for Q3 was $12.5 million, up 18% YoY.",
        "The company hired 47 new engineers in Q3.",
        "Customer churn rate decreased to 5.5%.",
        "R&D spending increased by $1.2M for AI initiatives.",
        "Supply chain disruptions remain a key risk factor.",
    ]

    for i, t in enumerate(texts):
        rag.ingest_bytes(t.encode("utf-8"), f"doc_{i}.txt")

    # Search
    ctx = rag.search("What was the revenue?", top_k=3)
    print(f"  Search results: {len(ctx.results)}")
    print(f"  Search time: {ctx.search_time_ms:.1f} ms")

    assert len(ctx.results) > 0, "No search results"
    for r in ctx.results:
        print(f"    [{r.score:.4f}] {r.chunk.tagged_text[:60]}...")


# ===================================================================
# Test 5: RAG Engine — Grounded Prompt
# ===================================================================

def test_rag_grounded_prompt():
    """Test grounded prompt construction with citations."""
    from rag_engine import SnapVaultRAG

    rag = SnapVaultRAG()
    rag.ingest_bytes(
        b"Net income was $2.1M with a 16.8% profit margin.",
        "financials.txt",
    )

    prompt, ctx = rag.build_grounded_prompt("What is the profit margin?")
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Context chunks: {len(ctx.results)}")

    assert "SnapVault" in prompt, "System identity missing"
    assert "citation" in prompt.lower() or "cite" in prompt.lower(), \
        "Citation enforcement missing"
    assert len(prompt) > 100, "Prompt too short"


# ===================================================================
# Test 6: Inference Engine — Emulation Mode
# ===================================================================

def test_inference_emulation():
    """Test the inference engine in emulation mode."""
    from qnn_engine import SnapdragonInferenceEngine

    engine = SnapdragonInferenceEngine(model_path=None)

    assert engine.is_emulation, "Should be in emulation mode"
    assert engine.active_provider == "EmulationMode"

    status = engine.device_status()
    print(f"  Provider: {status['active_provider']}")
    print(f"  Label: {status['provider_label']}")
    print(f"  Emulation: {status['emulation']}")

    # Test generation
    prompt = (
        "--- Context ---\n"
        "Revenue was $12.5M. [Source: report.pdf, Page: 1]\n"
        "--- End Context ---\n\n"
        "Question: What was the revenue?\n"
        "Answer (with citations):"
    )

    response, telemetry = engine.generate(prompt, max_tokens=128)
    print(f"  Response length: {len(response)} chars")
    print(f"  TTFT: {telemetry.ttft_ms:.1f} ms")
    print(f"  Tokens: {telemetry.tokens_generated}")
    print(f"  Throughput: {telemetry.tokens_per_sec:.1f} tok/s")

    assert len(response) > 0, "Empty response"
    assert telemetry.ttft_ms > 0, "TTFT should be positive"
    assert telemetry.tokens_generated > 0, "Should generate tokens"


# ===================================================================
# Test 7: Inference Engine — Device Status
# ===================================================================

def test_device_status():
    """Test device status reporting."""
    from qnn_engine import SnapdragonInferenceEngine

    engine = SnapdragonInferenceEngine()
    status = engine.device_status()

    required_keys = [
        "active_provider", "provider_label", "is_npu",
        "emulation", "model_loaded", "model_path",
        "system_memory_used_mb", "system_memory_total_mb",
    ]

    for key in required_keys:
        assert key in status, f"Missing key: {key}"
        print(f"  {key}: {status[key]}")


# ===================================================================
# Test 8: End-to-End Integration
# ===================================================================

def test_end_to_end():
    """Test the full RAG + inference pipeline."""
    from rag_engine import SnapVaultRAG
    from qnn_engine import SnapdragonInferenceEngine

    rag = SnapVaultRAG()
    engine = SnapdragonInferenceEngine()

    # Ingest a document
    doc = (
        "Acme Corp Q3 2025 Financial Summary\n"
        "Total Revenue: $45.2 million (up 22% YoY)\n"
        "Operating Expenses: $31.8 million\n"
        "Net Income: $8.7 million\n"
        "Profit Margin: 19.2%\n"
        "Headcount: 1,247 employees\n"
        "Risk: Pending litigation may result in $3M liability.\n"
        "Risk: Foreign exchange volatility affecting EMEA revenue.\n"
    )
    rag.ingest_bytes(doc.encode("utf-8"), "acme_q3_2025.txt")
    print(f"  Ingested: {rag.chunk_count} chunks")

    # Build grounded prompt and generate
    prompt, ctx = rag.build_grounded_prompt("What are the key financial metrics?")
    response, telemetry = engine.generate(prompt)

    print(f"  Search results: {len(ctx.results)}")
    print(f"  Search time: {ctx.search_time_ms:.1f} ms")
    print(f"  Response preview: {response[:120]}...")
    print(f"  TTFT: {telemetry.ttft_ms:.1f} ms")
    print(f"  Throughput: {telemetry.tokens_per_sec:.1f} tok/s")

    assert len(response) > 0, "Empty response"

    # Test audit actions
    for name, fn in [("Metrics", rag.build_metrics_prompt),
                     ("Risks", rag.build_risk_prompt),
                     ("Brief", rag.build_brief_prompt)]:
        p, c = fn()
        r, t = engine.generate(p)
        print(f"  {name}: {len(r)} chars, {t.tokens_per_sec:.1f} tok/s")
        assert len(r) > 0, f"Empty {name} response"


# ===================================================================
# Run All Tests
# ===================================================================

def main():
    print("=" * 60)
    print("  SnapVault — Smoke Test Suite")
    print("=" * 60)
    print(f"  Python: {sys.version}")
    print(f"  CWD: {os.getcwd()}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    test("1. Import Validation", test_imports)
    test("2. RAG Text Ingestion", test_rag_text_ingestion)
    test("3. RAG CSV Ingestion", test_rag_csv_ingestion)
    test("4. RAG Vector Search", test_rag_search)
    test("5. RAG Grounded Prompt", test_rag_grounded_prompt)
    test("6. Inference Emulation", test_inference_emulation)
    test("7. Device Status", test_device_status)
    test("8. End-to-End Integration", test_end_to_end)

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)

    for name, status, error in results:
        line = f"  {status}  {name}"
        if error:
            line += f"  ({error})"
        print(line)

    print(f"\n  Total: {len(results)} | Passed: {passed} | Failed: {failed}")

    if failed == 0:
        print("\n  🎉 ALL TESTS PASSED — SnapVault is ready!")
    else:
        print(f"\n  ⚠️  {failed} test(s) failed — review errors above.")

    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
