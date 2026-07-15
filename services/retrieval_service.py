"""Stage 3 — retrieval: build a retriever over the vector store.

This service is deliberately isolated: the advanced retrieval enhancements
(MMR, hybrid search, reranking, query rewriting) will all be added here without
touching the rest of the pipeline.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

import config


class RetrievalService:
    """Builds retrievers from a vector store."""

    def build_retriever(self, vector_store: FAISS) -> VectorStoreRetriever:
        """Create a similarity retriever over the given vector store.

        Args:
            vector_store: The FAISS store built by the indexing service.

        Returns:
            A configured retriever returning the top-k relevant chunks.
        """

        return vector_store.as_retriever(
            search_type=config.SEARCH_TYPE,
            search_kwargs={"k": config.RETRIEVAL_K},
        )
