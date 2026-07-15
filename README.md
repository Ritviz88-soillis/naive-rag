# naive-rag
# 🎥 YouTube Video Q&A (RAG)

A Retrieval-Augmented Generation app that answers questions about any YouTube
video from its transcript. Paste a URL, build the index, and ask questions —
every answer is grounded in the video's transcript.

## Features

- **Paste a full YouTube URL** — the video ID is extracted automatically
  (`watch?v=`, `youtu.be/`, `embed/`, `shorts/`, …).
- **Transcript → chunks → embeddings → FAISS** vector store.
- **Grounded answers** — the LLM is instructed to answer *only* from the
  retrieved transcript context.
- **On-disk caching** — a video embedded once is loaded from disk on later runs,
  so it's never re-fetched or re-embedded (survives restarts

## Architecture

```
app.py                    Streamlit UI (entry point)
orchestrator.py           thin delegate between UI and pipeline
config.py                 all settings (models, chunk size, k, paths)
prompts/qa_prompt.py      the QA prompt template
services/
  ingestion_service.py    fetch the transcript
  indexing_service.py     chunk + embed + FAISS (+ save/load cache)
  retrieval_service.py    build the retriever
  generation_service.py   prompt -> LLM -> answer
  rag_service.py          the brain: sequences the stages, cache-first
utils/
  youtube.py              extract video id from a URL
  formatting.py           format retrieved docs into context
```

## Setup

```bash
# 1. clone and enter
git clone <your-repo-url>
cd mini_project

# 2. create & activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt

# 4. add your HuggingFace token
cp .env.example .env        # then edit .env and paste your token

# 5. run
streamlit run app.py
```

## How to use

1. In the sidebar, paste a **YouTube URL** and click **Build index**
   (first time: fetches + embeds; later: loads from cache).
2. Ask questions in the main panel — answers come only from the transcript.

## Tech stack

- **UI:** Streamlit
- **Framework:** LangChain
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** `meta-llama/Llama-3.1-8B-Instruct` (via HuggingFace Inference)
- **Vector store:** FAISS (persisted to disk)

## Roadmap

This is a **naive RAG** baseline. Planned enhancements:

- **Evaluation:** Ragas, LangSmith
- **Pre-retrieval:** query rewriting, multi-query, domain-aware routing
- **Retrieval:** MMR, hybrid (dense + BM25), reranking
- **Post-retrieval:** contextual compression

## Notes

- `.env` (your token) and `vector_stores/` (the cache) are git-ignored — do not
  commit them.
