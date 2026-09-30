"""
SnapVault — Streamlit Dashboard & Control Center (app.py)
=========================================================
Modern dark-themed application with:
  - Hardware acceleration status pill (QNN / DirectML / CPU)
  - Zero-cloud telemetry badges
  - Document upload sidebar with registry
  - One-click audit actions
  - Chat interface with streaming + context accordion
"""

import streamlit as st
import time
import os

from qnn_engine import SnapdragonInferenceEngine
from rag_engine import SnapVaultRAG

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SnapVault — On-Device Confidential Copilot",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark Theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Root variables */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a2e;
        --bg-card-hover: #222240;
        --accent-primary: #6c5ce7;
        --accent-secondary: #a29bfe;
        --accent-green: #00d2a0;
        --accent-amber: #ffc107;
        --accent-red: #ff6b6b;
        --text-primary: #e8e8f0;
        --text-secondary: #9999b0;
        --border-color: #2a2a40;
        --glass-bg: rgba(26, 26, 46, 0.8);
        --glass-border: rgba(108, 92, 231, 0.2);
    }

    /* Global styling */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0d0d1a 50%, var(--bg-primary) 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .snap-header {
        background: linear-gradient(135deg, rgba(108,92,231,0.15), rgba(162,155,254,0.08));
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(20px);
    }

    .snap-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.2rem;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe, #00d2a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .snap-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-top: 4px;
        font-weight: 400;
    }

    /* Status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 24px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 4px;
        backdrop-filter: blur(10px);
    }

    .status-npu {
        background: rgba(0, 210, 160, 0.12);
        border: 1px solid rgba(0, 210, 160, 0.3);
        color: #00d2a0;
    }

    .status-fallback {
        background: rgba(255, 193, 7, 0.12);
        border: 1px solid rgba(255, 193, 7, 0.3);
        color: #ffc107;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 3px;
    }

    .badge-secure {
        background: rgba(108, 92, 231, 0.12);
        border: 1px solid rgba(108, 92, 231, 0.3);
        color: #a29bfe;
    }

    .badge-airgap {
        background: rgba(0, 210, 160, 0.10);
        border: 1px solid rgba(0, 210, 160, 0.25);
        color: #00d2a0;
    }

    .badge-hp {
        background: rgba(255, 193, 7, 0.10);
        border: 1px solid rgba(255, 193, 7, 0.25);
        color: #ffc107;
    }

    /* Metric card */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: var(--glass-border);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108, 92, 231, 0.15);
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent-secondary);
        margin: 8px 0 4px 0;
    }

    .metric-label {
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Audit action button */
    .audit-btn {
        display: block;
        width: 100%;
        padding: 14px 18px;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        color: var(--text-primary);
        font-size: 0.9rem;
        font-weight: 500;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-bottom: 8px;
    }

    .audit-btn:hover {
        background: var(--bg-card-hover);
        border-color: var(--accent-primary);
        transform: translateX(4px);
    }

    /* Chat messages */
    .chat-user {
        background: linear-gradient(135deg, rgba(108,92,231,0.15), rgba(108,92,231,0.05));
        border: 1px solid rgba(108, 92, 231, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        color: var(--text-primary);
    }

    .chat-assistant {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        color: var(--text-primary);
    }

    .chat-timestamp {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-secondary);
        margin-top: 8px;
    }

    /* Context accordion */
    .context-chunk {
        background: rgba(108, 92, 231, 0.06);
        border-left: 3px solid var(--accent-primary);
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    .context-score {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent-green);
        font-weight: 600;
        font-size: 0.75rem;
    }

    /* Sidebar styling */
    .sidebar-title {
        font-weight: 700;
        color: var(--text-primary);
        font-size: 1.1rem;
        margin-bottom: 12px;
    }

    .doc-item {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    /* Streamlit element overrides */
    .stTextInput > div > div > input {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 1px var(--accent-primary) !important;
    }

    div[data-testid="stExpander"] {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
    }

    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialize all session state variables."""
    if "engine" not in st.session_state:
        st.session_state.engine = SnapdragonInferenceEngine(model_path=None)

    if "rag" not in st.session_state:
        st.session_state.rag = SnapVaultRAG()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "processing" not in st.session_state:
        st.session_state.processing = False


init_session_state()

engine: SnapdragonInferenceEngine = st.session_state.engine
rag: SnapVaultRAG = st.session_state.rag


# ---------------------------------------------------------------------------
# Header & Status Bar
# ---------------------------------------------------------------------------

def render_header():
    """Render the application header with status indicators."""
    status = engine.device_status()

    pill_class = "status-npu" if status["is_npu"] else "status-fallback"
    pill_text = status["provider_label"]

    st.markdown(f"""
    <div class="snap-header">
        <h1 class="snap-title">🔒 SnapVault</h1>
        <p class="snap-subtitle">On-Device Confidential Document & Tabular Copilot</p>
        <div style="margin-top: 12px;">
            <span class="status-pill {pill_class}">{pill_text}</span>
        </div>
        <div style="margin-top: 8px;">
            <span class="badge badge-secure">🛡️ Zero Cloud Telemetry</span>
            <span class="badge badge-airgap">🔐 100% Air-Gapped</span>
            <span class="badge badge-hp">💻 HP OmniBook Optimized</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


render_header()


# ---------------------------------------------------------------------------
# Telemetry Metrics Row
# ---------------------------------------------------------------------------

def render_metrics():
    """Render telemetry metric cards."""
    status = engine.device_status()
    telemetry = engine.last_telemetry

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mem_used = status.get("system_memory_used_mb", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Memory Usage</div>
            <div class="metric-value">{mem_used:,.0f}</div>
            <div class="metric-label">MB</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Indexed Chunks</div>
            <div class="metric-value">{rag.chunk_count}</div>
            <div class="metric-label">segments</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        throughput = telemetry.tokens_per_sec if telemetry else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Gen. Throughput</div>
            <div class="metric-value">{throughput:.1f}</div>
            <div class="metric-label">tokens/sec</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        ttft = telemetry.ttft_ms if telemetry else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Time to First Token</div>
            <div class="metric-value">{ttft:.0f}</div>
            <div class="metric-label">ms</div>
        </div>
        """, unsafe_allow_html=True)


render_metrics()
st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — File Uploader & Document Registry
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-title">📂 Document Upload</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            # Avoid re-ingesting the same file
            if uf.name not in rag.document_registry:
                with st.spinner(f"Indexing {uf.name}..."):
                    n_chunks = rag.ingest_bytes(uf.read(), uf.name)
                    if n_chunks > 0:
                        st.success(f"✅ {uf.name} — {n_chunks} chunks indexed")
                    else:
                        st.warning(f"⚠️ {uf.name} — no text extracted")

    st.markdown("---")
    st.markdown('<div class="sidebar-title">📋 Active Documents</div>', unsafe_allow_html=True)

    registry = rag.document_registry
    if registry:
        for doc_name, chunk_count in registry.items():
            st.markdown(f"""
            <div class="doc-item">
                📄 <strong>{doc_name}</strong><br>
                <span style="font-size: 0.75rem; color: #9999b0;">{chunk_count} chunks indexed</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No documents uploaded yet.")

    st.markdown("---")

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        rag.clear()
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")

    # Device info
    st.markdown('<div class="sidebar-title">⚙️ Device Info</div>', unsafe_allow_html=True)
    status = engine.device_status()
    st.caption(f"**Provider:** {status['active_provider']}")
    st.caption(f"**Model:** {'Loaded' if status['model_loaded'] else 'Emulation Mode'}")
    st.caption(f"**Emulation:** {status['emulation']}")
    st.caption(f"**RAM:** {status['system_memory_used_mb']:,.0f} / {status['system_memory_total_mb']:,.0f} MB")


# ---------------------------------------------------------------------------
# One-Click Audit Actions
# ---------------------------------------------------------------------------

st.markdown("### ⚡ Quick Audit Actions")

audit_col1, audit_col2, audit_col3 = st.columns(3)

with audit_col1:
    if st.button("🔍 Extract Key Metrics & Numeric Totals", use_container_width=True):
        if rag.chunk_count == 0:
            st.warning("Upload documents first.")
        else:
            prompt, context = rag.build_metrics_prompt()
            response, telemetry = engine.generate(prompt)
            st.session_state.chat_history.append({
                "role": "user",
                "content": "🔍 Extract Key Metrics & Numeric Totals",
                "timestamp": time.time(),
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "context": context,
                "telemetry": telemetry,
                "timestamp": time.time(),
            })
            st.rerun()

with audit_col2:
    if st.button("⚠️ Identify Liabilities & Risk Factors", use_container_width=True):
        if rag.chunk_count == 0:
            st.warning("Upload documents first.")
        else:
            prompt, context = rag.build_risk_prompt()
            response, telemetry = engine.generate(prompt)
            st.session_state.chat_history.append({
                "role": "user",
                "content": "⚠️ Identify Liabilities & Risk Factors",
                "timestamp": time.time(),
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "context": context,
                "telemetry": telemetry,
                "timestamp": time.time(),
            })
            st.rerun()

with audit_col3:
    if st.button("📋 Generate Executive Brief", use_container_width=True):
        if rag.chunk_count == 0:
            st.warning("Upload documents first.")
        else:
            prompt, context = rag.build_brief_prompt()
            response, telemetry = engine.generate(prompt)
            st.session_state.chat_history.append({
                "role": "user",
                "content": "📋 Generate Executive Brief",
                "timestamp": time.time(),
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "context": context,
                "telemetry": telemetry,
                "timestamp": time.time(),
            })
            st.rerun()

st.markdown("---")


# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------

st.markdown("### 💬 Document Chat")

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-user">
            <strong>You</strong><br>{msg['content']}
        </div>
        """, unsafe_allow_html=True)

    elif msg["role"] == "assistant":
        st.markdown(f"""
        <div class="chat-assistant">
            <strong>🔒 SnapVault</strong><br>{msg['content']}
            <div class="chat-timestamp">
                TTFT: {msg.get('telemetry', None) and f"{msg['telemetry'].ttft_ms:.0f}ms" or 'N/A'} •
                Tokens: {msg.get('telemetry', None) and msg['telemetry'].tokens_generated or 'N/A'} •
                Speed: {msg.get('telemetry', None) and f"{msg['telemetry'].tokens_per_sec:.1f} tok/s" or 'N/A'} •
                Provider: {msg.get('telemetry', None) and msg['telemetry'].provider or 'N/A'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Expandable context accordion
        context = msg.get("context")
        if context and context.results:
            with st.expander(f"📚 Retrieved Context — {len(context.results)} chunks ({context.search_time_ms:.1f}ms search)"):
                for i, r in enumerate(context.results, 1):
                    st.markdown(f"""
                    <div class="context-chunk">
                        <span class="context-score">Chunk {i} — Score: {r.score:.4f}</span><br>
                        {r.chunk.tagged_text[:300]}{'...' if len(r.chunk.tagged_text) > 300 else ''}
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
user_query = st.chat_input("Ask about your documents...")

if user_query:
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query,
        "timestamp": time.time(),
    })

    if rag.chunk_count == 0:
        response = (
            "I don't have any documents to reference yet. "
            "Please upload PDF, CSV, or TXT files using the sidebar, "
            "and I'll be able to answer questions about their content."
        )
        from qnn_engine import InferenceTelemetry
        telemetry = InferenceTelemetry(provider="EmulationMode")
        context = None
    else:
        # Build grounded prompt and generate
        prompt, context = rag.build_grounded_prompt(user_query)
        response, telemetry = engine.generate(prompt)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "context": context,
        "telemetry": telemetry,
        "timestamp": time.time(),
    })
    st.rerun()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 16px; color: #666680; font-size: 0.8rem;">
    <strong>SnapVault</strong> — 100% On-Device • Zero Cloud • Snapdragon X Optimized<br>
    Built for the Snapdragon® AI Lab Build & Present Challenge<br>
    <span style="color: #4a4a6a;">Powered by Qualcomm Hexagon NPU • ONNX Runtime QNN-EP • HP OmniBook</span>
</div>
""", unsafe_allow_html=True)
