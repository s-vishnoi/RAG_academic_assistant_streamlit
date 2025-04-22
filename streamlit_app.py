import os
import tempfile
import time
import streamlit as st
from streamlit_chat import message
from academic_assistant_deepseek import ChatPDF

st.set_page_config(page_title="RAG with Local DeepSeek R1")

# Cache the assistant so it's not reloaded on every rerun
@st.cache_resource
def get_chatpdf():
    return ChatPDF()

def display_messages():
    """Display the chat history."""
    st.subheader("Chat History")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    """Process the user input and generate an assistant response."""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
            try:
                agent_text = st.session_state["assistant"].ask(
                    user_text,
                    k=st.session_state["retrieval_k"],
                    score_threshold=st.session_state["retrieval_threshold"],
                )
            except ValueError as e:
                agent_text = str(e)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    """Handle file upload and ingestion."""
    for file in st.session_state["file_uploader"]:
        if file.name in st.session_state.get("ingested_files", []):
            continue  # Skip already ingested files

        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}..."):
            t0 = time.time()
            st.session_state["assistant"].ingest(file_path)
            t1 = time.time()

        st.session_state.setdefault("ingested_files", []).append(file.name)
        st.session_state["messages"].append((f"Ingested {file.name} in {t1 - t0:.2f} seconds", False))
        os.remove(file_path)

def page():
    """Main app page layout."""
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = get_chatpdf()
        st.session_state["messages"] = []
        st.session_state["ingested_files"] = []
        st.session_state["user_input"] = ""

    
    st.header("RAG with Local DeepSeek")

    if st.button("Reset Conversation"):
        st.session_state["assistant"].reset_chat()
        st.session_state["messages"] = []

    # File upload section
    st.subheader("Upload a Document")
    st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    # Retrieval settings
    st.subheader("Settings")
    st.session_state["retrieval_k"] = st.slider(
        "Number of Retrieved Results (k)", min_value=1, max_value=10, value=5
    )
    st.session_state["retrieval_threshold"] = st.slider(
        "Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )

    # Chat section
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

page()


