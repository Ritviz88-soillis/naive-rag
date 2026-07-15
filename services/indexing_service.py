"""Stage 2 — indexing: split the transcript, embed it, and build a vector store."""

import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


class IndexingService:
    """Turns raw transcript text into a searchable FAISS vector store."""

    def __init__(self) -> None:
        """Build the splitter and embedding model once (reused per request)."""

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        self._embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    def create_chunks(self, transcript: str) -> List[Document]:
        """Split a transcript into overlapping chunks.

        Args:
            transcript: The full transcript text.

        Returns:
            A list of chunk Documents.
        """

        return self._splitter.create_documents([transcript])

    def build_vector_store(self, transcript: str) -> FAISS:
        """Chunk, embed, and index the transcript into FAISS.

        Args:
            transcript: The full transcript text.

        Returns:
            A FAISS vector store over the transcript chunks.
        """

        chunks = self.create_chunks(transcript)
        return FAISS.from_documents(chunks, self._embeddings)

    def _store_path(self, video_id: str) -> str:
        """Return the on-disk folder for a video's saved index."""

        return os.path.join(config.VECTOR_STORE_DIR, video_id)

    def has_cached(self, video_id: str) -> bool:
        """Return True if a saved index already exists for this video.

        Args:
            video_id: The YouTube video id.

        Returns:
            Whether a persisted FAISS index is on disk.
        """

        return os.path.isdir(self._store_path(video_id))

    def save(self, vector_store: FAISS, video_id: str) -> None:
        """Persist a vector store to disk under the video's id.

        Args:
            vector_store: The FAISS store to save.
            video_id: The YouTube video id (used as the folder name).
        """

        vector_store.save_local(self._store_path(video_id))

    def load(self, video_id: str) -> FAISS:
        """Load a previously saved vector store from disk.

        Args:
            video_id: The YouTube video id.

        Returns:
            The persisted FAISS vector store, re-attached to the embeddings.
        """

        return FAISS.load_local(
            self._store_path(video_id),
            self._embeddings,
            allow_dangerous_deserialization=True,  # we created these files ourselves
        )
