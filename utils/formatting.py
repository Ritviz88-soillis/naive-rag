"""Shared, stateless helpers for the RAG pipeline (t1's ``utils/``)."""

from typing import List

from langchain_core.documents import Document


def format_docs(retrieved_docs: List[Document]) -> str:
    """Join retrieved document chunks into a single context string.

    Args:
        retrieved_docs: The documents returned by the retriever.

    Returns:
        The concatenated page contents, separated by blank lines.
    """

    return "\n\n".join(doc.page_content for doc in retrieved_docs)
