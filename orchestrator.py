"""Delegation  the UI and the RAG service
"""

import config
from services.rag_service import RAGService


class RAGOrchestrator:
    def __init__(self) -> None:
        """Create the underlying RAG service."""

        self._service = RAGService()

    def index_video(self, video_id: str, language: str = config.DEFAULT_LANGUAGE) -> bool:
        """Index a video's transcript for question answering.

        Args:
            video_id: The YouTube video id.
            language: The transcript language code.

        Returns:
            True if the index was loaded from the on-disk cache, else False.
        """

        return self._service.index_video(video_id, language)

    def answer(self, question: str) -> str:
        """Answer a question against the indexed video.

        Args:
            question: The user's question.

        Returns:
            The grounded answer.
        """

        return self._service.answer(question)
