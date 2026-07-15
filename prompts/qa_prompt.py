"""
Prompt templates for the RAG pipeline.
"""

from langchain_core.prompts import PromptTemplate

QA_PROMPT = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    Context:
    {context}

    Question: {question}
    """,
    input_variables=["context", "question"],
)
