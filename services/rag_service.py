"""The RAG brain: sequences ingestion -> indexing -> retrieval -> generation.

Holds no algorithm itself (mirrors t1's GaugeReadingService) -- it wires the
stage services together and keeps the per-video retriever as state.
"""

from typing import Optional

from langchain_core.vectorstores import VectorStoreRetriever

import config
from services.generation_service import GenerationService
from services.indexing_service import IndexingService
from services.ingestion_service import IngestionService
from services.retrieval_service import RetrievalService
from utils.formatting import format_docs


class RAGService:
    """Orchestrates the full RAG pipeline for one video at a time."""

    def __init__(self) -> None:
        """Instantiate the stage services once (reused across questions)."""

        self._ingestion = IngestionService()
        self._indexing = IndexingService()
        self._retrieval = RetrievalService()
        self._generation = GenerationService()
        self._retriever: Optional[VectorStoreRetriever] = None

    def index_video(self, video_id: str, language: str = config.DEFAULT_LANGUAGE) -> bool:
        """Prepare a video for question answering, using the on-disk cache.

        If this video was embedded before, its saved FAISS index is loaded from
        disk. Otherwise the transcript is fetched, embedded, and the new index is
        saved for next time. Either way the resulting retriever is stored as
        state so subsequent questions reuse it.

        Args:
            video_id: The YouTube video id.
            language: The transcript language code.

        Returns:
            True if the index was loaded from cache, False if freshly built.
        """

        from_cache = self._indexing.has_cached(video_id)
        if from_cache:
            vector_store = self._indexing.load(video_id)
        else:
            transcript = self._ingestion.fetch_transcript(video_id, language)
            vector_store = self._indexing.build_vector_store(transcript)
            self._indexing.save(vector_store, video_id)

        self._retriever = self._retrieval.build_retriever(vector_store)
        return from_cache

    def answer(self, question: str) -> str:
        """Answer a question against the currently indexed video.

        Args:
            question: The user's question.

        Returns:
            The grounded answer.

        Raises:
            RuntimeError: If no video has been indexed yet.
        """

        if self._retriever is None:
            raise RuntimeError("No video indexed. Call index_video() first.")

        retrieved_docs = self._retriever.invoke(question)
        context = format_docs(retrieved_docs)
        return self._generation.generate(context, question)
