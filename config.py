"""Central configuration for the RAG mini-project.

All tunable settings live here so services stay free of hard-coded constants
(mirrors t1's ``app.config``).
"""

import os

# Models 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
LLM_TASK = "text-generation"

#  Chunking 
CHUNK_SIZE = 800
CHUNK_OVERLAP = 160

#  Retrieval 
SEARCH_TYPE = "similarity"   # later: "mmr"
RETRIEVAL_K = 4

# Defaults 
DEFAULT_LANGUAGE = "en"
DEFAULT_VIDEO_ID = "RNF0FvRjGZk"

#  Persistence
# Saved FAISS indexes live here, one folder per video id, so an already-embedded
# video is loaded from disk instead of re-fetching + re-embedding on every run.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_DIR = os.path.join(_PROJECT_DIR, "vector_stores")
