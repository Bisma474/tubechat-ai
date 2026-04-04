import streamlit as st
import re
import time
from supadata import Supadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from groq import Groq

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
.chat-container { display: flex; flex-direction: column; gap: 14px; margin: 1rem 0; } .msg-user { align-self: flex-end; background: var(--accent-red); color: white; padding: 12px 16px; border-radius: 18px 18px 4px 18px; max-width: 78%; font-size: 14px; line-height: 1.5; box-shadow: 0 4px 14px var(--accent-dim); } .msg-bot { align-self: flex-start; background: var(--bg-card); border: 1px solid var(--border); color: var(--text-primary); padding: 14px 18px; border-radius: 18px 18px 18px 4px; max-width: 85%; font-size: 14px; line-height: 1.6; } .msg-label { font-family: 'Space Mono', monospace; font-size: 10px; margin-bottom: 4px; } .msg-label.user { color: #FF6B6B; text-align: right; } .msg-label.bot { color: var(--text-muted); }
.stTextInput input, .stTextArea textarea { background: var(--bg-input) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--text-primary) !important; font-family: 'DM Sans', sans-serif !important; } .stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--accent-red) !important; box-shadow: 0 0 0 2px var(--accent-glow) !important; } .stButton button { background: var(--accent-red) !important; color: white !important; border: none !important; border-radius: 10px !important; font-family: 'Space Mono', monospace !important; font-size: 13px !important; font-weight: 700 !important; letter-spacing: 0.5px !important; padding: 0.55rem 1.2rem !important; transition: all 0.2s ease !important; box-shadow: 0 4px 14px var(--accent-dim) !important; } .stButton button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px var(--accent-dim) !important; } .stSelectbox select, div[data-baseweb="select"] { background: var(--bg-input) !important; border-color: var(--border) !important; } div[data-baseweb="select"] * { background: var(--bg-input) !important; color: var(--text-primary) !important; } .stSpinner > div { border-top-color: var(--accent-red) !important; } .stSuccess, .stInfo, .stWarning, .stError { border-radius: 10px !important; } hr { border-color: var(--border) !important; }
.welcome-box { background: var(--bg-card); border: 1px dashed var(--border); border-radius: 18px; padding: 50px 30px; text-align: center; margin: 2rem 0; } .welcome-box h2 { font-family: 'Space Mono', monospace; font-size: 18px; color: var(--text-primary); margin-bottom: 10px; } .welcome-box p { color: var(--text-muted); font-size: 14px; line-height: 1.7; } .welcome-icon { font-size: 48px; margin-bottom: 16px; } ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: var(--bg-primary); } ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def extract_video_id(url: str) -> str | None:
    """Used strictly to get the ID so we can embed the YouTube player in the UI."""
    patterns = [r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})"]
    for p in patterns:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

@st.cache_resource(show_spinner=False)
def load_embeddings_model():
    # Embeddings still run locally via sentence-transformers (100% free, no API key needed!)
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_transcript_via_supadata(url: str, api_key: str) -> str:
    """Fetches the transcript using the Supadata SDK, handling async jobs for large videos."""
    client = Supadata(api_key=api_key)
    
    response = client.transcript(
        url=url,
        lang="en",  
        text=True,  
        mode="auto" 
    )
    
    # 1. If the video is short and processes immediately
    if hasattr(response, 'content') and response.content:
        return response.content
        
    # 2. If the video is long and requires async background processing
    elif hasattr(response, 'job_id'):
        job_id = response.job_id
        status_box = st.empty()
        
        # Loop up to 40 times (waiting 3 seconds each time) = 2 minutes max
        for i in range(40): 
            status_box.info(f"⏳ Video is large! Processing asynchronously in the background... (Wait a moment)")
            time.sleep(3) 
            
            try:
                # Ask Supadata if the job is finished yet
                result = client.batch.get_batch_results(job_id)
                
                # Check safely depending on how Supadata formats the answer
                status = result.get('status') if isinstance(result, dict) else getattr(result, 'status', '')
                content = result.get('content') if isinstance(result, dict) else getattr(result, 'content', None)
                
                if status == "completed" or content:
                    status_box.empty() # Clear the waiting message
                    return content
                elif status in ["failed", "error"]:
                    status_box.empty()
                    raise ValueError("Supadata failed to extract the transcript.")
                    
            except Exception:
                # If it's not ready yet, just skip and check again in 3 seconds
                pass
                
        status_box.empty()
        raise ValueError("Timeout: The video took too long to process. Please try a shorter video.")
        
    else:
        raise ValueError("Could not retrieve transcript from Supadata.")

def build_vectorstore(transcript: str, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])
    docs = splitter.create_documents([transcript])
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, len(docs)

def answer_query(question: str, vectorstore, groq_api_key: str, model_id: str) -> str:
    """Generates the answer using Groq's lightning-fast API"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Initialize Groq Client
    client = Groq(api_key=groq_api_key)
    
        messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Answer the user's question based ONLY on the provided transcript context. YOU MUST ALWAYS RESPOND IN ENGLISH, NO MATTER WHAT LANGUAGE THE TRANSCRIPT OR QUESTION IS IN."},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
    ]
    
    # Call Groq API
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.3,
        max_tokens=512,
    )
    
    return completion.choices[0].message.content.strip()

# ─────────────────────────────────────────────
# SESSION STATE & SIDEBAR
# ─────────────────────────────────────────────
defaults = {"vectorstore": None, "transcript": None, "num_chunks": 0, "video_id": None, "chat_history": [], "pipeline_step": 0, "video_loaded": False}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# Keys are explicitly loaded from the backend secrets
groq_api_key = st.secrets["GROQ_API_KEY"]
supadata_key = st.secrets["SUPADATA_API_KEY"]

with st.sidebar:
    st.markdown("<div class='brand-header'><div class='brand-icon'>▶</div><div><div class='brand-name'>TubeChat</div><div class='brand-sub'>YouTube × Groq AI</div></div></div>", unsafe_allow_html=True)

    st.markdown("#### 🤖 Groq Model")
    model_options = {
    "Llama 3.1 8B (Fast)":        "llama-3.1-8b-instant",
    "Llama 3.3 70B (Best)":       "llama-3.3-70b-versatile",
    "Llama 4 Scout 17B":          "meta-llama/llama-4-scout-17b-16e-instruct",
    "Qwen3 32B":                  "qwen/qwen3-32b",
}
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
    steps = [("📄", "Load"), ("✂️", "Split"), ("🔢", "Embed"), ("🗄️", "Index"), ("🔍", "Retrieve"), ("🧩", "Augment"), ("⚡", "Groq")]
    html = '<div class="pipeline-wrapper">'
    for i, (icon, label) in enumerate(steps):
        html += f'<div class="step-box {"active" if i < step else ""}"><div class="step-icon">{icon}</div><div>{label}</div></div>'
        if i < len(steps) - 1: html += f'<div class="step-arrow {"active" if i < step - 1 else ""}">→</div>'
    return html + "</div>"

st.markdown(pipeline_html(st.session_state.pipeline_step), unsafe_allow_html=True)
st.divider()

if load_btn:
    if not video_url:
        st.error("⚠️ Please paste a YouTube URL.")
    else:
        video_id = extract_video_id(video_url)
        if not video_id: st.error("❌ Invalid URL.")
        else:
            st.session_state.chat_history = []
            st.session_state.video_loaded = False
            progress = st.progress(15, text="📄 Fetching transcript via Supadata…")
            try:
                st.session_state.pipeline_step = 1
                
                transcript = get_transcript_via_supadata(video_url, supadata_key)
                
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
    st.markdown("<div class='welcome-box'><div class='welcome-icon'>▶️</div><h2>Paste a YouTube URL to get started</h2><p>Using Supadata & lightning-fast Groq models!</p></div>", unsafe_allow_html=True)
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
        # Temporarily store the user's question in chat history
        st.session_state.chat_history.append({"role": "user", "content": user_q.strip()})
        
        with st.spinner("⚡ Generating answer with Groq…"):
            try:
                ans = answer_query(user_q.strip(), st.session_state.vectorstore, groq_api_key, model_id)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
                st.rerun() # Success! Reload the UI to show the answer.
            except Exception as e:
                st.session_state.chat_history.pop() # Remove the user's question so they can try again
                st.error(f"⚠️ Groq Error: {str(e)}")
