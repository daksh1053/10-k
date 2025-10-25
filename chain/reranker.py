"""
Reranking to improve retrieval precision.
"""

from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

def rerank_documents(config: Dict[str, Any], documents: List[Document], query: str) -> List[Document]:
    """Rerank documents using Cohere reranker."""
    if not documents:
        return documents

    compressor = CohereRerank(model=config["reranker_model"])
    return compressor.compress_documents(documents, query)
