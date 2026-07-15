"""Stage 4 — generation: turn context + question into a grounded answer."""

from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import config
from prompts.qa_prompt import QA_PROMPT

parser = StrOutputParser()

class GenerationService:
    """Runs the LLM over the retrieved context to answer the question."""

    def __init__(self) -> None:
        """Build the LLM chain (prompt -> chat model -> string parser) once."""

        llm = HuggingFaceEndpoint(repo_id=config.LLM_REPO_ID, task=config.LLM_TASK)
        model = ChatHuggingFace(llm=llm)
        self._chain = QA_PROMPT | model | parser

    def generate(self, context: str, question: str) -> str:
        """Generate an answer grounded in the provided context.

        Args:
            context: The formatted retrieved-chunk text.
            question: The user's question.

        Returns:
            The generated answer string.
        """

        return self._chain.invoke({"context": context, "question": question})
