import uuid
import html
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from backend import workflow, retrieve_all

st.set_page_config(
    page_title="Hospital AI Copilot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

:root { --navy:#102a43; --teal:#167d8d; --muted:#627d98; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp {
    background:
      linear-gradient(115deg, rgba(8,25,43,.84), rgba(10,50,65,.69)),
      url("https://images.unsplash.com/photo-1586773860418-d37222d8fce3?auto=format&fit=crop&w=2000&q=85");
    background-size:cover;
    background-position:center;
    background-attachment:fixed;
}
[data-testid="stSidebar"] {
    background:rgba(9,24,39,.96);
    border-right:1px solid rgba(255,255,255,.10);
}
[data-testid="stSidebar"] * { color:#eef6fb; }
.brand {
    display:flex; align-items:center; gap:12px; padding:8px 0 18px;
}
.brand-icon {
    width:46px;height:46px;border-radius:14px;
    display:flex;align-items:center;justify-content:center;
    background:linear-gradient(135deg,#35b8b0,#167d8d);
    font-size:25px;box-shadow:0 8px 25px rgba(0,0,0,.18);
}
.brand-title { font-size:21px;font-weight:700;line-height:1.2;color:white; }
.brand-sub { font-size:12px;color:#b6c9d8;margin-top:4px; }
.main .block-container { max-width:1000px; padding-top:2rem; padding-bottom:8rem; }
.hero {
    color:white; text-align:center; padding:28px 18px 22px;
}
.hero h1 { color:white; font-size:clamp(30px,4vw,43px); margin:0 0 10px; font-weight:700; }
.hero p { color:#e3eef5; font-size:15px; margin:0 auto; max-width:680px; }
.pill {
    display:inline-block; padding:6px 12px; border-radius:999px;
    background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22);
    color:#f4fbff; font-size:12px; margin-bottom:15px;
}
.chat-card {
    background:rgba(255,255,255,.96); border:1px solid rgba(255,255,255,.45);
    border-radius:18px; padding:18px 20px; margin:12px 0;
    box-shadow:0 12px 34px rgba(0,0,0,.12); color:#183247;
}
.chat-card.user { background:#e4f6f5; border-color:#b9e5e1; }
.chat-label { color:#167d8d; font-size:11px; font-weight:700; letter-spacing:.08em; margin-bottom:7px; }
.chat-text { color:#183247; font-size:15px; line-height:1.65; white-space:pre-wrap; overflow-wrap:anywhere; }
.notice {
    max-width:780px;margin:12px auto;padding:12px 15px;border-radius:12px;
    color:#f0f7fb;background:rgba(255,255,255,.10);
    border:1px solid rgba(255,255,255,.18);font-size:12px;line-height:1.5;
}
div.stButton > button {
    border-radius:10px; min-height:42px; font-weight:600;
    border:1px solid rgba(255,255,255,.20);
}
div.stButton > button[kind="primary"] {
    background:#1b9aaa; border-color:#1b9aaa; color:white;
}
[data-testid="stChatInput"] {
    background:rgba(255,255,255,.97); border-radius:16px;
}
[data-testid="stChatInput"] textarea { color:#183247; }
section[data-testid="stSidebar"] .stButton button {
    text-align:left; width:100%; white-space:normal;
}
.small-muted { color:#d1e0e9; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def get_config(thread_id):
    return {"configurable": {"thread_id": thread_id}}

def get_thread_ids():
    try:
        return list(dict.fromkeys(retrieve_all()))
    except Exception:
        # Fallback for LangGraph versions where checkpointer listing differs.
        try:
            ids = []
            for item in workflow.checkpointer.list(None):
                cfg = item.config.get("configurable", {})
                tid = cfg.get("thread_id")
                if tid and tid not in ids:
                    ids.append(tid)
            return ids
        except Exception:
            return []

def get_messages(thread_id):
    try:
        snapshot = workflow.get_state(get_config(thread_id))
        return snapshot.values.get("messages", []) if snapshot and snapshot.values else []
    except Exception:
        return []

def get_title(thread_id):
    messages = get_messages(thread_id)
    for message in messages:
        if isinstance(message, HumanMessage):
            text = str(message.content).strip().replace("\\n", " ")
            return text[:34] + ("…" if len(text) > 34 else "")
    return "New conversation"

def visible_messages(messages):
    # Hide tool-call and tool-result messages; show user prompts and final assistant answers.
    result = []
    for message in messages:
        if isinstance(message, HumanMessage):
            result.append(("user", message.content))
        elif isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            if message.content:
                result.append(("assistant", message.content))
    return result

# ---------- Session state ----------
if "thread_id" not in st.session_state:
    existing = get_thread_ids()
    st.session_state.thread_id = existing[-1] if existing else str(uuid.uuid4())

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-icon">✚</div>
      <div>
        <div class="brand-title">Hospital Copilot</div>
        <div class="brand-sub">Your hospital information assistant</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("＋  New chat", type="primary", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown("---")
    st.markdown("#### Recent conversations")
    thread_ids = get_thread_ids()
    # Most recently updated threads tend to appear at the end; newest first.
    for tid in reversed(thread_ids):
        title = get_title(tid)
        is_current = tid == st.session_state.thread_id
        label = ("●  " if is_current else "   ") + title
        if st.button(label, key=f"thread_{tid}", use_container_width=True):
            st.session_state.thread_id = tid
            st.session_state.pending_prompt = None
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div class="small-muted">This assistant retrieves information from the configured hospital documents. '
        'For emergencies, contact hospital emergency services immediately.</div>',
        unsafe_allow_html=True
    )

# ---------- Main area ----------
st.markdown("""
<div class="hero">
  <div class="pill">● HOSPITAL INFORMATION • DOCTOR DIRECTORY • WAYFINDING</div>
  <h1>Your Hospital AI Copilot</h1>
  <p>Ask about hospital services, find a doctor, locate a department, or explore doctor profiles and synthetic patient-experience examples.</p>
</div>
""", unsafe_allow_html=True)

current_messages = get_messages(st.session_state.thread_id)
shown = visible_messages(current_messages)

if not shown:
    st.markdown("""
    <div class="notice">
      <b>Welcome.</b> Ask a question below to get started. This assistant uses the hospital documents connected to its retrieval system.
      Doctor ratings and patient experiences in the demo dataset are synthetic and should not be treated as real reviews.
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(2)
    suggestions = [
        "Where is the ICU and how do I reach it?",
        "Find a cardiologist and show their qualifications.",
        "Where is the hospital canteen?",
        "Summarize the fictional patient feedback for a doctor."
    ]
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button("✦  " + suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state.pending_prompt = suggestion
                st.rerun()
else:
    for role, content in shown:
        if role == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(str(content))
        else:
            # Render assistant output as Markdown for readable lists and headings.
            with st.chat_message("assistant", avatar="🏥"):
                st.markdown(content)

# ---------- Chat input ----------
prompt = st.chat_input("Message Hospital Copilot…")
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    config = get_config(st.session_state.thread_id)
    try:
        with st.spinner("Searching hospital information…"):
            workflow.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
        st.rerun()
    except Exception as exc:
        st.error(f"Could not generate a response: {exc}")
        st.caption("Check that backend.py is configured with a valid NVIDIA model name and API key.")
