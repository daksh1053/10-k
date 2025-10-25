"""
Implement the vector stores and its embedding service
"""

import os
import time
import numpy as np
import requests
from typing import Dict, Any, List, Optional
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class VectorStore:
    """Vector store for text documents and tabel summaries"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_function = EmbeddingService(config, config["huggingface_token"])
        self.chunkStore, self.tableSummaryStore = self.initialize_store()

    def initialize_store(self):
        chunk_store_path = "chunk_store"
        table_summary_store_path = "table_summary_store"

        # Initialize chunk store
        if os.path.exists(chunk_store_path):
            self.chunkStore = FAISS.load_local(chunk_store_path, self.embedding_function, allow_dangerous_deserialization=True)
        else:
            # Get embedding dimension
            embedding_dim = len(self.embedding_function.embed_query(""))
            index = faiss.IndexFlatL2(embedding_dim)

            self.chunkStore = FAISS(
                embedding_function=self.embedding_function,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            self.chunkStore.save_local(chunk_store_path)

        # Initialize table summary store
        if os.path.exists(table_summary_store_path):
            self.tableSummaryStore = FAISS.load_local(table_summary_store_path, self.embedding_function, allow_dangerous_deserialization=True)
        else:
            # Get embedding dimension
            embedding_dim = len(self.embedding_function.embed_query(""))
            index = faiss.IndexFlatL2(embedding_dim)

            self.tableSummaryStore = FAISS(
                embedding_function=self.embedding_function,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            self.tableSummaryStore.save_local(table_summary_store_path)

        return self.chunkStore, self.tableSummaryStore

    def retriever(self, query: str, chunk_k: int, table_k: int, ids_to_retrieve: Optional[List[int]] = None):
        """Retrieve documents from both chunk and table stores."""
        chunks, tables = [], []

        if ids_to_retrieve is None:
            # No filtering - get top k from each store
            if chunk_k > 0:
                chunks = self.chunkStore.similarity_search(query, k=chunk_k)
            if table_k > 0:
                tables = self.tableSummaryStore.similarity_search(query, k=table_k)
        else:
            # Filter by filling_id
            def filter_func(metadata):
                return metadata.get("filling_id") in ids_to_retrieve

            if chunk_k > 0:
                chunks = self.chunkStore.similarity_search(
                    query, k=chunk_k, filter=filter_func
                )
            if table_k > 0:
                tables = self.tableSummaryStore.similarity_search(
                    query, k=table_k, filter=filter_func
                )

        return chunks, tables


    def add_chunk_documents(self, documents: List[Document]) -> None:
        self.chunkStore.add_documents(documents)
        self.chunkStore.save_local("chunk_store")

    def add_table_documents(self, documents: List[Document]) -> None:
        self.tableSummaryStore.add_documents(documents)
        self.tableSummaryStore.save_local("table_summary_store")

class EmbeddingService(Embeddings):
    """Embedding generator for VectorStore using HuggingFace Inference API."""
    
    def __init__(self, config: Dict[str, Any], hf_token: str):
        self.api_url = config["embedding_api_url"]
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        self.query_instruction = "Represent this question for searching relevant passages: "
        self.embedding_dim = config["embedding_dim"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts_to_embed = [text.replace("\n", " ") for text in texts]
        return self._generate_embeddings(texts_to_embed)

    def embed_query(self, text: str) -> List[float]:
        text_to_embed = self.query_instruction + text.replace("\n", " ")
        return self._generate_embeddings([text_to_embed])[0]

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        results = [np.zeros(self.embedding_dim).tolist() for _ in range(len(texts))]
        batch_size = 30
        long_retries = 0

        batches = [(i, texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]

        for batch_id, batch in batches:
            start_time = time.time()

            # Retry logic for each batch
            while time.time() - start_time < 30:
                try:
                    response = self._send_batch_request(batch)

                    if not isinstance(response, list):
                        continue

                    break
                except Exception as e:
                    time.sleep(1)
                    continue

            # Handle long retries
            if time.time() - start_time > 10:
                long_retries += 1
                if long_retries > 3:
                    print("Too many long retries, stopping embedding generation")
                    return results

            for j, embedding in enumerate(response):
                if isinstance(embedding, list) and len(embedding) == self.embedding_dim:
                    results[batch_id + j] = embedding
                else:
                    print(f"Invalid embedding format at index {batch_id + j}: {str(embedding)[:100]}...")

        if len(texts) > 1:
            successful = sum(1 for emb in results if any(emb))
            print(f"Successfully embedded {successful}/{len(texts)} texts")

        return results

    def _send_batch_request(self, batch: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": batch}
        )
        if response.status_code != 200:
            raise ValueError(f"API returned status code {response.status_code}")
        return response.json()