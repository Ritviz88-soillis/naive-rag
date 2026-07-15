"""Streamlit UI for the YouTube-transcript RAG app (the entry point / router)."""

import streamlit as st
from dotenv import load_dotenv

import config
from orchestrator import RAGOrchestrator
from utils.youtube import extract_video_id

load_dotenv()

st.set_page_config(page_title="YouTube RAG", page_icon="🎥")
st.title(" YouTube Video Q&A (RAG)")

#
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = RAGOrchestrator()
    st.session_state.indexed_video = None

#Sidebar: load & index a video
with st.sidebar:
    st.header("1. Load a video")
    video_url = st.text_input(
        "YouTube URL",
        value=f"https://www.youtube.com/watch?v={config.DEFAULT_VIDEO_ID}",
    )
    language = st.text_input("Language", value=config.DEFAULT_LANGUAGE)

    if st.button("Build index"):
        with st.spinner("Fetching transcript & building index..."):
            try:
                video_id = extract_video_id(video_url)          # URL -> 11-char id
                from_cache = st.session_state.orchestrator.index_video(video_id, language)
                st.session_state.indexed_video = video_id
                if from_cache:
                    st.success(f"Loaded {video_id} from cache (no re-embedding).")
                else:
                    st.success(f"Fetched & embedded {video_id} (saved for next time).")
            except ValueError as error:  # invalid / unparseable URL
                st.session_state.indexed_video = None
                st.error(f"Invalid YouTube URL: {error}")
            except Exception as error:  # ingestion / indexing failure
                st.session_state.indexed_video = None
                st.error(f"Failed to index: {error}")

# ask questions
st.header("2. Ask a question")

if st.session_state.indexed_video is None:
    st.info("Build the index for a video first (sidebar).")
else:
    st.caption(f"Answering about video: {st.session_state.indexed_video}")
    question = st.text_input("Your question")

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            answer = st.session_state.orchestrator.answer(question)
        st.markdown("### Answer")
        st.write(answer)
