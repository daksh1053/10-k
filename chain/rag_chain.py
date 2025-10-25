"""
RAG chain implementation
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from .reranker import rerank_documents
from .query_decomposer import decompose_query

class RAGChain:
    """Main RAG class with query decomposer, reranker, and table retrieval capabilities"""

    def __init__(
        self,
        config: Dict[str, Any],
        vector_store,  # This should be the VectorStore class instance
        table_retrieval_func: Optional[Callable[[str], Tuple[str, Dict[str, Any]]]] = None
    ):
        self.config = config
        self.vector_store = vector_store 
        self.table_retrieval_func = table_retrieval_func
        self.llm = ChatOpenAI(
            model=config["rag_answer_model"],
            temperature=config["rag_answer_model_temperature"]
        )
        self.prompt = PromptTemplate.from_template(config["rag_prompt_template"])
        self.num_selected_documents = config.get("num_selected_documents", 5)
        self.reranker = CohereRerank(model=config["reranker_model"])
        self.chain = self._make_chain()
        self._active_ids: Optional[List[int]] = None

    def _make_chain(self):
        return (
            RunnableParallel({
                "context": RunnableLambda(self._retrieve_context),
                "question": RunnablePassthrough()
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str, ids_to_retrieve: Optional[List[int]] = None) -> str:
        self._active_ids = ids_to_retrieve
        query_params = {
            "question": question
        }
        try:
            return self.chain.invoke(query_params)
        finally:
            self._active_ids = None


    def _retrieve_context(self, query_params: Dict[str, Any]) -> str:
        question = query_params["question"]
        chunk_k = int(self.config.get("chunk_k", 0))
        table_k = int(self.config.get("table_k", 0))
        ids_to_retrieve = self._active_ids if self._active_ids is not None else self.config.get("ids_to_retrieve")

        subqueries = decompose_query(self.config, question)
        if not subqueries:
            subqueries = [question]

        context_string = ''
        for subq in subqueries:
            chunk_docs, table_docs = self.vector_store.retriever(subq, chunk_k, table_k, ids_to_retrieve)

            if chunk_docs:
                chunk_docs = self._rerank_documents(chunk_docs, subq)[:chunk_k]
            if table_docs:
                table_docs = self._rerank_documents(table_docs, subq)[:table_k]

            context_string += self._format_docs(chunk_docs + table_docs)
            context_string += "\n\n"

            context_string += self._format_tables(table_docs)
            context_string += "\n\n"

        return context_string.strip()

    def _format_docs(self, docs: List[Document]) -> str:
        def _get_prefix(doc):
            company = doc.metadata.get('company', '')
            return company

        return "\n\n".join(
            f"{_get_prefix(doc)}\n{doc.page_content}"
            for doc in docs
        )

    def _format_tables(self, docs: List[Document]) -> str:
        processed_docs = []

        for doc in docs:
            table_id = doc.metadata.get("table_id")
            if table_id and self.table_retrieval_func:
                try:
                    table_data = self.table_retrieval_func(table_id)
                    if table_data:
                        content, metadata = table_data
                    else:
                        content = doc.page_content
                except Exception as e:
                    print(f"Error retrieving table {table_id}: {e}")
                    content = doc.page_content
            else:
                content = doc.page_content

            company = doc.metadata.get('company', '')
            processed_docs.append(f"{company}\n{content}")

        return "\n\n".join(processed_docs)

    def _rerank_documents(self, docs: List[Document], query: str) -> List[Document]:
        if not docs:
            return docs
        try:
            return self.reranker.compress_documents(docs, query)
        except Exception:
            return docs



