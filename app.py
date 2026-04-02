import streamlit as st
import re
import time
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TubeChat — YouTube AI Chatbot",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:   #0D0D12;
    --bg-card:      #16161F;
    --bg-input:     #1E1E2A;
    --accent-red:   #FF2D55;
    --accent-dim:   #FF2D5530;
    --accent-glow:  #FF2D5515;
    --text-primary: #F0F0F5;
    --text-muted:   #7A7A9A;
    --border:       #2A2A38;
    --success:      #2ECC71;
    --step-done:    #FF2D5520;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary);
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Brand Header ── */
.brand-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 2rem;
}
.brand-icon {
    width: 44px; height: 44px;
    background: var(--accent-red);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 20px var(--accent-dim);
}
.brand-name {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #FF2D55, #FF6B6B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.brand-sub {
    font-size: 11px;
    color: var(--text-muted);
    font-family: 'Space Mono', monospace;
    margin-top: -4px;
}

/* ── Pipeline Steps ── */
.pipeline-wrapper {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 1.5rem 0 2rem;
    overflow-x: auto;
    padding-bottom: 4px;
}
.step-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 90px;
    padding: 10px 8px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    font-size: 11px;
    color: var(--text-muted);
    text-align: center;
    transition: all 0.3s ease;
    flex-shrink: 0;
}
.step-box.active {
    background: var(--step-done);
    border-color: var(--accent-red);
    color: var(--accent-red);
    box-shadow: 0 0 12px var(--accent-glow);
}
.step-icon { font-size: 18px; margin-bottom: 4px; }
.step-arrow {
    color: var(--border);
    font-size: 14px;
    padding: 0 4px;
    flex-shrink: 0;
}
.step-arrow.active { color: var(--accent-red); }

/* ── Cards ── */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.info-card h4 {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.info-card .value {
    font-size: 15px;
    color: var(--text-primary);
    font-weight: 500;
}
.stat-row {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
}
.stat-pill {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.stat-pill .num {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: var(--accent-red);
}
.stat-pill .label {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 2px;
}

/* ── Chat ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 14px;
    margin: 1rem 0;
}
.msg-user {
    align-self: flex-end;
    background: var(--accent-red);
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 78%;
    font-size: 14px;
    line-height: 1.5;
    box-shadow: 0 4px 14px var(--accent-dim);
}
.msg-bot {
    align-self: flex-start;
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    max-width: 85%;
    font-size: 14px;
    line-height: 1.6;
}
.msg-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    margin-bottom: 4px;
}
.msg-label.user { color: #FF6B6B; text-align: right; }
.msg-label.bot  { color: var(--text-muted); }

/* ── Streamlit Widgets Overrides ── */
.stTextInput input, .stTextArea textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent-red) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}
.stButton button {
    background: var(--accent-red) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px var(--accent-dim) !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px var(--accent-dim) !important;
}
.stSelectbox select, div[data-baseweb="select"] {
    background: var(--bg-input) !important;
    border-color: var(--border) !important;
}
div[data-baseweb="select"] * { background: var(--bg-input) !important; color: var(--text-primary) !important; }
.stSpinner > div { border-top-color: var(--accent-red) !important; }
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 10px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Welcome Screen ── */
.welcome-box {
    background: var(--bg-card);
    border: 1px dashed var(--border);
    border-radius: 18px;
    padding: 50px 30px;
    text-align: center;
    margin: 2rem 0;
}
.welcome-box h2 {
    font-family: 'Space Mono', monospace;
    font-size: 18px;
    color: var(--text-primary);
    margin-bottom: 10px;
}
.welcome-box p { color: var(--text-muted); font-size: 14px; line-height: 1.7; }
.welcome-icon { font-size: 48px; margin-bottom: 16px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    """Extracts the video ID just for embedding the iframe player."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


@st.cache_resource(show_spinner=False)
def load_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_transcript_with_langchain(video_url: str) -> str:
    """Uses LangChain's YoutubeLoader to fetch and auto-translate to English."""
    # A broad list of common languages to catch whatever language the video might be in.
    # The loader will look for these, and translate the result to English ('en').
    languages = [
        "en", "en-US", "en-GB", "es", "fr", "de", "hi", "zh", "ja", 
        "ko", "pt", "ru", "ar", "it", "nl", "tr", "pl", "id", "vi", "th"
    ]
    
    loader = YoutubeLoader.from_youtube_url(
        video_url, 
        add_video_info=False,
        language=languages, 
        translation="en"
    )
    
    docs = loader.load()
    if not docs:
        raise ValueError("Could not find a transcript for this video in any known language.")
        
    full_text = " ".join([doc.page_content for doc in docs])
    return full_text


def build_vectorstore(transcript: str, embeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.create_documents([transcript])
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, len(docs)


def answer_query(question: str, vectorstore, hf_token: str, model_id: str) -> str:
    # ── Retrieval ──
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # ── Prompt (Augmentation) ──
    prompt = f"""You are a helpful AI assistant that answers questions based strictly on the YouTube video transcript provided below.

TRANSCRIPT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the transcript context above.
- Be concise, clear, and accurate.
- If the answer is not in the transcript, say "I couldn't find this in the video transcript."
- Do not make up information.

ANSWER:"""

    # ── Generation ──
    client = InferenceClient(token=hf_token)
    response = client.text_generation(
        prompt,
        model=model_id,
        max_new_tokens=512,
        temperature=0.3,
        repetition_penalty=1.1,
        do_sample=True,
    )
    return response.strip()


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "vectorstore": None,
    "transcript": None,
    "num_chunks": 0,
    "video_id": None,
    "chat_history": [],
    "pipeline_step": 0,
    "video_loaded": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div class="brand-icon">▶</div>
        <div>
            <div class="brand-name">TubeChat</div>
            <div class="brand-sub">YouTube × GenAI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🔑 HuggingFace API Token")

    hf_token = st.secrets.get("huggingfacehub_api_token", "") if hasattr(st, "secrets") else ""
    if not hf_token:
        hf_token = st.text_input(
            "Paste your HF token",
            type="password",
            placeholder="hf_...",
            help="Get your token at huggingface.co/settings/tokens",
        )

    st.markdown("#### 🤖 Generation Model")
    model_options = {
        "Mistral 7B Instruct v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
        "Zephyr 7B Beta": "HuggingFaceH4/zephyr-7b-beta",
        "Phi-3 Mini 4K": "microsoft/Phi-3-mini-4k-instruct",
    }
    selected_model_label = st.selectbox("Choose model", list(model_options.keys()))
    model_id = model_options[selected_model_label]

    st.divider()

    st.markdown("#### 🎬 YouTube Video")
    video_url = st.text_input(
        "Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    load_btn = st.button("⚡ Load & Process Video", use_container_width=True)

    if st.session_state.video_loaded:
        st.divider()
        st.markdown(f"""
        <div style='background:#16161F;border:1px solid #2A2A38;border-radius:10px;padding:14px;'>
            <div style='font-size:11px;color:#7A7A9A;font-family:Space Mono,monospace;margin-bottom:8px;'>VIDEO STATUS</div>
            <div style='color:#2ECC71;font-size:13px;font-weight:600;'>✓ Ready to chat</div>
            <div style='font-size:12px;color:#7A7A9A;margin-top:6px;'>{st.session_state.num_chunks} chunks indexed</div>
            <div style='font-size:12px;color:#7A7A9A;'>{len(st.session_state.transcript.split()):,} words loaded</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑 Clear & Reset", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()

    st.divider()
    st.markdown("""
    <div style='font-size:11px;color:#7A7A9A;line-height:1.7;'>
        <b style='color:#F0F0F5;'>How it works</b><br>
        1. Loads YouTube transcript (Auto-translates to English via LangChain)<br>
        2. Splits into chunks<br>
        3. Embeds with MiniLM-L6-v2<br>
        4. Stores in FAISS index<br>
        5. Retrieves on your query<br>
        6. Augments & generates answer
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────

st.markdown("""
<h1 style='font-family:Space Mono,monospace;font-size:26px;font-weight:700;
    background:linear-gradient(135deg,#FF2D55,#FF9500);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:4px;'>
    TubeChat — Chat with any YouTube Video
</h1>
<p style='color:#7A7A9A;font-size:14px;margin-top:0;'>
    RAG pipeline: Transcript → Chunks → Embeddings → FAISS → Retrieval → Generation
</p>
""", unsafe_allow_html=True)


def pipeline_html(step: int):
    steps = [
        ("📄", "Load\nTranscript"),
        ("✂️",  "Split\nChunks"),
        ("🔢", "Embed"),
        ("🗄️",  "FAISS\nIndex"),
        ("🔍", "Retrieve"),
        ("🧩", "Augment"),
        ("✨", "Generate"),
    ]
    html = '<div class="pipeline-wrapper">'
    for i, (icon, label) in enumerate(steps):
        active = "active" if i < step else ""
        html += f'<div class="step-box {active}"><div class="step-icon">{icon}</div><div>{label}</div></div>'
        if i < len(steps) - 1:
            arrow_cls = "active" if i < step - 1 else ""
            html += f'<div class="step-arrow {arrow_cls}">→</div>'
    html += "</div>"
    return html

st.markdown(pipeline_html(st.session_state.pipeline_step), unsafe_allow_html=True)
st.divider()


# ─────────────────────────────────────────────
# LOAD VIDEO LOGIC
# ─────────────────────────────────────────────
if load_btn:
    if not hf_token:
        st.error("⚠️ Please enter your HuggingFace API token in the sidebar.")
    elif not video_url:
        st.error("⚠️ Please paste a YouTube video URL.")
    else:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check and try again.")
        else:
            st.session_state.chat_history = []
            st.session_state.video_loaded = False

            col_prog, _ = st.columns([3, 1])
            with col_prog:
                progress = st.progress(0, text="Starting pipeline…")

            try:
                # Step 1 — Load using Langchain's YoutubeLoader
                progress.progress(15, text="📄 Loading & translating transcript (LangChain)…")
                st.session_state.pipeline_step = 1
                transcript = get_transcript_with_langchain(video_url)
                time.sleep(0.3)

                # Step 2 — Split
                progress.progress(35, text="✂️ Splitting into chunks…")
                st.session_state.pipeline_step = 2
                embeddings = load_embeddings_model()
                time.sleep(0.2)

                # Step 3-4 — Embed + Index
                progress.progress(55, text="🔢 Embedding chunks & building FAISS index…")
                st.session_state.pipeline_step = 4
                vectorstore, num_chunks = build_vectorstore(transcript, embeddings)
                time.sleep(0.3)

                # Done
                progress.progress(100, text="✅ Pipeline complete — ready to chat!")
                st.session_state.pipeline_step = 7
                st.session_state.vectorstore = vectorstore
                st.session_state.transcript   = transcript
                st.session_state.num_chunks   = num_chunks
                st.session_state.video_id     = video_id
                st.session_state.video_loaded = True
                time.sleep(0.6)
                progress.empty()
                st.rerun()

            except Exception as e:
                progress.empty()
                st.error(f"❌ Could not load video: {str(e)}\n\n(Note: The video must have closed captions enabled to work.)")


# ─────────────────────────────────────────────
# CHAT UI
# ─────────────────────────────────────────────
if not st.session_state.video_loaded:
    st.markdown("""
    <div class="welcome-box">
        <div class="welcome-icon">▶️</div>
        <h2>Paste a YouTube URL to get started</h2>
        <p>
            TubeChat uses <b>LangChain's YoutubeLoader</b> to load and auto-translate transcripts,<br>
            breaks it into semantic chunks, embeds them with MiniLM-L6-v2, indexes in FAISS, <br>
            then answers your questions using HuggingFace's generation models.<br><br>
            <b>Works with any YouTube video that has captions enabled (even non-English!).</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Stats row
    word_count  = len(st.session_state.transcript.split())
    chunk_count = st.session_state.num_chunks
    char_count  = len(st.session_state.transcript)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill"><div class="num">{word_count:,}</div><div class="label">Words</div></div>
        <div class="stat-pill"><div class="num">{chunk_count}</div><div class="label">Chunks</div></div>
        <div class="stat-pill"><div class="num">{char_count:,}</div><div class="label">Characters</div></div>
        <div class="stat-pill"><div class="num">MiniLM</div><div class="label">Embeddings</div></div>
        <div class="stat-pill"><div class="num">FAISS</div><div class="label">Vector DB</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Embed player
    st.markdown(f"""
    <div style='border-radius:14px;overflow:hidden;margin-bottom:20px;border:1px solid #2A2A38;'>
        <iframe width="100%" height="300"
            src="https://www.youtube.com/embed/{st.session_state.video_id}"
            frameborder="0" allowfullscreen>
        </iframe>
    </div>
    """, unsafe_allow_html=True)

    # Chat history display
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div>
                    <div class="msg-label user">YOU</div>
                    <div style='display:flex;justify-content:flex-end;'>
                        <div class="msg-user">{msg["content"]}</div>
                    </div>
                </div>"""
            else:
                chat_html += f"""
                <div>
                    <div class="msg-label bot">TUBECHAT</div>
                    <div class="msg-bot">{msg["content"]}</div>
                </div>"""
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Example questions
    if not st.session_state.chat_history:
        st.markdown("""
        <div style='background:#16161F;border:1px solid #2A2A38;border-radius:12px;padding:16px 20px;margin-bottom:16px;'>
            <div style='font-size:12px;color:#7A7A9A;font-family:Space Mono,monospace;margin-bottom:10px;'>EXAMPLE QUESTIONS</div>
            <div style='font-size:13px;color:#F0F0F5;line-height:2;'>
                • What is the main topic of this video?<br>
                • Summarize the key points discussed.<br>
                • What conclusions did the speaker reach?<br>
                • What examples were given to explain the concept?
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Input
    user_q = st.text_input(
        "Ask anything about the video",
        placeholder="e.g. What are the key points discussed in this video?",
        label_visibility="collapsed",
        key="user_input",
    )

    col_send, col_clear = st.columns([1, 4])
    with col_send:
        send_btn = st.button("Send ➤", use_container_width=True)

    if send_btn and user_q.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_q.strip()})
        st.session_state.pipeline_step = 5  # retrieval active

        with st.spinner("🔍 Retrieving context → ✨ Generating answer…"):
            try:
                answer = answer_query(
                    user_q.strip(),
                    st.session_state.vectorstore,
                    hf_token,
                    model_id,
                )
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.pipeline_step = 7
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ Generation error: {str(e)}\n\nMake sure your HuggingFace token is valid and the model supports text generation."
                })
        st.rerun()
