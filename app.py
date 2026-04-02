import streamlit as st
import re
import time
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="TubeChat — YouTube AI Chatbot", page_icon="▶", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root { --bg-primary: #0D0D12; --bg-card: #16161F; --bg-input: #1E1E2A; --accent-red: #FF2D55; --accent-dim: #FF2D5530; --accent-glow: #FF2D5515; --text-primary: #F0F0F5; --text-muted: #7A7A9A; --border: #2A2A38; --success: #2ECC71; --step-done: #FF2D5520; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: var(--bg-primary) !important; color: var(--text-primary); }
#MainMenu, footer, header { visibility: hidden; } .block-container { padding-top: 1.5rem !important; max-width: 1100px; }
[data-testid="stSidebar"] { background: var(--bg-card) !important; border-right: 1px solid var(--border); } [data-testid="stSidebar"] * { color: var(--text-primary) !important; }
.brand-header { display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; } .brand-icon { width: 44px; height: 44px; background: var(--accent-red); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; box-shadow: 0 0 20px var(--accent-dim); } .brand-name { font-family: 'Space Mono', monospace; font-size: 22px; font-weight: 700; letter-spacing: -0.5px; background: linear-gradient(135deg, #FF2D55, #FF6B6B); -webkit-background-clip: text; -webkit-text-fill-color: transparent; } .brand-sub { font-size: 11px; color: var(--text-muted); font-family: 'Space Mono', monospace; margin-top: -4px; }
.pipeline-wrapper { display: flex; align-items: center; gap: 0; margin: 1.5rem 0 2rem; overflow-x: auto; padding-bottom: 4px; } .step-box { display: flex; flex-direction: column; align-items: center; min-width: 90px; padding: 10px 8px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; font-size: 11px; color: var(--text-muted); text-align: center; transition: all 0.3s ease; flex-shrink: 0; } .step-box.active { background: var(--step-done); border-color: var(--accent-red); color: var(--accent-red); box-shadow: 0 0 12px var(--accent-glow); } .step-icon { font-size: 18px; margin-bottom: 4px; } .step-arrow { color: var(--border); font-size: 14px; padding: 0 4px; flex-shrink: 0; } .step-arrow.active { color: var(--accent-red); }
.info-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; padding: 20px 22px; margin-bottom: 16px; } .info-card h4 { font-family: 'Space Mono', monospace; font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; } .info-card .value { font-size: 15px; color: var(--text-primary); font-weight: 500; } .stat-row { display: flex; gap: 12px; margin-bottom: 16px; } .stat-pill { flex: 1; background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; text-align: center; } .stat-pill .num { font-family: 'Space Mono', monospace; font-size: 22px; font-weight: 700; color: var(--accent-red); } .stat-pill .label { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.chat-container { display: flex; flex-direction: column; gap: 14px; margin: 1rem 0; } .msg-user { align-self: flex-end; background: var(--accent-red); color: white; padding: 12px 16px; border-radius: 18px 18px 4px 18px; max-width: 78%; font-size: 14px; line-height: 1.5; box-shadow: 0 4px 14px var(--accent-dim); } .msg-bot { align-self: flex-start; background: var(--bg-card); border: 1px solid var(--border); color: var(--text-primary); padding: 14px 18px; border-radius: 18px 18px 18px 4px; max-width: 85%; font-size: 14px; line-height: 1.6; } .msg-label { font-family: 'Space Mono', monospace; font-size: 10px; margin-bottom: 4px; } .msg-label.user { color: #FF6B6B; text-align: right; } .msg-label.bot { color: var(--text-muted); }
.stTextInput input, .stTextArea textarea { background: var(--bg-input) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--text-primary) !important; font-family: 'DM Sans', sans-serif !important; } .stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--accent-red) !important; box-shadow: 0 0 0 2px var(--accent-glow) !important; } .stButton button { background: var(--accent-red) !important; color: white !important; border: none !important; border-radius: 10px !important; font-family: 'Space Mono', monospace !important; font-size: 13px !important; font-weight: 700 !important; letter-spacing: 0.5px !important; padding: 0.55rem 1.2rem !important; transition: all 0.2s ease !important; box-shadow: 0 4px 14px var(--accent-dim) !important; } .stButton button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px var(--accent-dim) !important; } .stSelectbox select, div[data-baseweb="select"] { background: var(--bg-input) !important; border-color: var(--border) !important; } div[data-baseweb="select"] * { background: var(--bg-input) !important; color: var(--text-primary) !important; } .stSpinner > div { border-top-color: var(--accent-red) !important; } .stSuccess, .stInfo, .stWarning, .stError { border-radius: 10px !important; } hr { border-color: var(--border) !important; }
.welcome-box { background: var(--bg-card); border: 1px dashed var(--border); border-radius: 18px; padding: 50px 30px; text-align: center; margin: 2rem 0; } .welcome-box h2 { font-family: 'Space Mono', monospace; font-size: 18px; color: var(--text-primary); margin-bottom: 10px; } .welcome-box p { color: var(--text-muted); font-size: 14px; line-height: 1.7; } .welcome-icon { font-size: 48px; margin-bottom: 16px; } ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: var(--bg-primary); } ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def extract_video_id(url: str) -> str | None:
    patterns = [r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})"]
    for p in patterns:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

@st.cache_resource(show_spinner=False)
def load_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_transcript_via_rapidapi(video_id: str, api_key: str, api_host: str) -> str:
    """Fetches the transcript using a generic RapidAPI endpoint."""
    url = f"https://{api_host}/api/transcript"
    
    querystring = {"video_id": video_id}
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": api_host
    }

    response = requests.get(url, headers=headers, params=querystring)
    
    if response.status_code != 200:
        raise ValueError(f"RapidAPI Error ({response.status_code}): {response.text}")
    
    data = response.json()
    
    try:
        if isinstance(data, list):
            return " ".join([item.get('text', '') for item in data])
        elif isinstance(data, dict):
            if 'transcript' in data:
                return " ".join([item.get('text', '') for item in data['transcript']])
            elif 'data' in data:
                return str(data['data'])
            else:
                return str(data)
    except Exception as e:
        raise ValueError(f"Could not parse transcript data from API: {str(e)}")

def build_vectorstore(transcript: str, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])
    docs = splitter.create_documents([transcript])
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, len(docs)

def answer_query(question: str, vectorstore, hf_token: str, model_id: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"You are an AI assistant answering questions based on the transcript.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    client = InferenceClient(token=hf_token)
    response = client.text_generation(prompt, model=model_id, max_new_tokens=512, temperature=0.3, repetition_penalty=1.1, do_sample=True)
    return response.strip()

# ─────────────────────────────────────────────
# SESSION STATE & SIDEBAR
# ─────────────────────────────────────────────
defaults = {"vectorstore": None, "transcript": None, "num_chunks": 0, "video_id": None, "chat_history": [], "pipeline_step": 0, "video_loaded": False}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

with st.sidebar:
    st.markdown("<div class='brand-header'><div class='brand-icon'>▶</div><div><div class='brand-name'>TubeChat</div><div class='brand-sub'>YouTube × GenAI</div></div></div>", unsafe_allow_html=True)

    # 1. Try to load keys securely from Streamlit Secrets
    try:
        hf_token = st.secrets["HUGGINGFACE_TOKEN"]
        rapidapi_key = st.secrets["RAPIDAPI_KEY"]
        rapidapi_host = "youtube-transcript3.p.rapidapi.com"
    except (KeyError, FileNotFoundError):
        # 2. If secrets aren't found, show the manual input boxes
        st.markdown("#### 🔑 API Keys")
        hf_token = st.text_input("HuggingFace Token", type="password", placeholder="hf_...", help="For the LLM Generation")
        rapidapi_key = st.text_input("RapidAPI Key", type="password", placeholder="Enter your RapidAPI key", help="For bypassing YouTube blocks")
        rapidapi_host = st.text_input("RapidAPI Host", value="youtube-transcript3.p.rapidapi.com", help="Check your RapidAPI code snippet for the correct host URL")

    st.markdown("#### 🤖 Model")
    model_options = {"Mistral 7B Instruct": "mistralai/Mistral-7B-Instruct-v0.3", "Zephyr 7B Beta": "HuggingFaceH4/zephyr-7b-beta"}
    model_id = model_options[st.selectbox("Choose model", list(model_options.keys()))]

    st.divider()
    video_url = st.text_input("🎬 Video URL", placeholder="https://www.youtube.com/watch?v=...")
    load_btn = st.button("⚡ Load Video", use_container_width=True)

    if st.session_state.video_loaded:
        st.divider()
        st.markdown(f"<div style='background:#16161F;border:1px solid #2A2A38;border-radius:10px;padding:14px;'><div style='font-size:11px;color:#7A7A9A;font-family:Space Mono,monospace;margin-bottom:8px;'>STATUS</div><div style='color:#2ECC71;font-size:13px;font-weight:600;'>✓ Ready</div><div style='font-size:12px;color:#7A7A9A;margin-top:6px;'>{st.session_state.num_chunks} chunks</div></div>", unsafe_allow_html=True)
        if st.button("🗑 Clear", use_container_width=True):
            for k, v in defaults.items(): st.session_state[k] = v
            st.rerun()

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.markdown("<h1 style='font-family:Space Mono,monospace;font-size:26px;font-weight:700;background:linear-gradient(135deg,#FF2D55,#FF9500);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;'>TubeChat — Chat with any YouTube Video</h1>", unsafe_allow_html=True)

def pipeline_html(step: int):
    steps = [("📄", "Load"), ("✂️", "Split"), ("🔢", "Embed"), ("🗄️", "Index"), ("🔍", "Retrieve"), ("🧩", "Augment"), ("✨", "Generate")]
    html = '<div class="pipeline-wrapper">'
    for i, (icon, label) in enumerate(steps):
        html += f'<div class="step-box {"active" if i < step else ""}"><div class="step-icon">{icon}</div><div>{label}</div></div>'
        if i < len(steps) - 1: html += f'<div class="step-arrow {"active" if i < step - 1 else ""}">→</div>'
    return html + "</div>"

st.markdown(pipeline_html(st.session_state.pipeline_step), unsafe_allow_html=True)
st.divider()

if load_btn:
    if not hf_token or not rapidapi_key:
        st.error("⚠️ Please enter both HuggingFace and RapidAPI tokens in the sidebar (or add them to secrets.toml).")
    elif not video_url:
        st.error("⚠️ Please paste a YouTube URL.")
    else:
        video_id = extract_video_id(video_url)
        if not video_id: st.error("❌ Invalid URL.")
        else:
            st.session_state.chat_history = []
            st.session_state.video_loaded = False
            progress = st.progress(15, text="📄 Fetching transcript via RapidAPI…")
            try:
                st.session_state.pipeline_step = 1
                transcript = get_transcript_via_rapidapi(video_id, rapidapi_key, rapidapi_host)
                
                progress.progress(35, text="✂️ Splitting & Embedding…")
                st.session_state.pipeline_step = 2
                embeddings = load_embeddings_model()
                
                st.session_state.pipeline_step = 4
                vectorstore, num_chunks = build_vectorstore(transcript, embeddings)
                
                progress.progress(100, text="✅ Ready to chat!")
                st.session_state.pipeline_step = 7
                st.session_state.vectorstore = vectorstore
                st.session_state.transcript = transcript
                st.session_state.num_chunks = num_chunks
                st.session_state.video_id = video_id
                st.session_state.video_loaded = True
                time.sleep(0.6)
                progress.empty()
                st.rerun()
            except Exception as e:
                progress.empty()
                st.error(f"❌ API Error: {str(e)}")

if not st.session_state.video_loaded:
    st.markdown("<div class='welcome-box'><div class='welcome-icon'>▶️</div><h2>Paste a YouTube URL to get started</h2><p>Using RapidAPI to bypass YouTube blocks!</p></div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div style='border-radius:14px;overflow:hidden;margin-bottom:20px;border:1px solid #2A2A38;'><iframe width='100%' height='300' src='https://www.youtube.com/embed/{st.session_state.video_id}' frameborder='0' allowfullscreen></iframe></div>", unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user": chat_html += f"<div><div class='msg-label user'>YOU</div><div style='display:flex;justify-content:flex-end;'><div class='msg-user'>{msg['content']}</div></div></div>"
            else: chat_html += f"<div><div class='msg-label bot'>TUBECHAT</div><div class='msg-bot'>{msg['content']}</div></div>"
        st.markdown(chat_html + "</div><br>", unsafe_allow_html=True)

    user_q = st.text_input("Ask anything", placeholder="e.g. What are the key points?", label_visibility="collapsed")
    if st.button("Send ➤", use_container_width=True) and user_q.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_q.strip()})
        with st.spinner("🔍 Generating answer…"):
            try:
                ans = answer_query(user_q.strip(), st.session_state.vectorstore, hf_token, model_id)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
            except Exception as e: st.error(str(e))
        st.rerun()
