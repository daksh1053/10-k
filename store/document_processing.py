"""
Preprocess 10-k statements and store them in chunks as embeddings in a vector store.
Add services to retrieve the chunks from the vector store.
"""

import re
import os
import math
import uuid
import time
import json
from typing import List, Dict, Tuple, Any
from urllib.parse import urljoin
import requests
import numpy as np
from multiprocessing import Pool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import fitz
import camelot
import pandas as pd

class TextPreProcessor:
    """Preprocesses SEC 10-K filings by loading and chunking documents."""
    
    def __init__(self, config: Dict[str, Any], company_filing_urls: List[Tuple[str, str]], filling_id: int):
        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]
        self.user_agent_header = config["user_agent_header"]
        return self.load_and_process_filings(company_filing_urls, filling_id)
        

    def load_and_process_filings(self, company_filing_urls: List[Tuple[str, str]], filling_id: int) -> Tuple[List[Document], Dict[str, str]]:
        processed_company_filings = []
        director_names_chunk = {}

        for company, url in company_filing_urls:
            try:
                filing_content = WebBaseLoader(
                    url, 
                    header_template={'User-Agent': self.user_agent_header}
                ).load()

                # Extract director signatures from the last 1000 characters
                director_names_chunk[company] = filing_content[0].page_content[-1000:]

                split_filing_content = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                ).transform_documents(filing_content)

                for split in split_filing_content:
                    split.metadata.update({
                        'company': company,
                        'filling_id': filling_id,
                    })

                processed_company_filings.extend(split_filing_content)

            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                continue

        return processed_company_filings, director_names_chunk


def process_document_part(args: Tuple[str, int, int]) -> List[Document]:
    """Process a specific range of pages from the PDF using Camelot."""
    file_path, start_page, end_page = args

    try:
        documents = []

        # Convert to 1-indexed page numbers for Camelot
        page_range = f"{start_page + 1}-{end_page}"

        # Try lattice method first (for bordered tables)
        try:
            tables_lattice = camelot.read_pdf(
                file_path,
                pages=page_range,
                flavor='lattice'
            )

            for table in tables_lattice:
                # Convert table to HTML using pandas
                html_content = table.df.to_html(index=False)

                doc = Document(
                    page_content=html_content,
                    metadata={
                        "category": "table",
                        "page": table.page,
                        "accuracy": table.accuracy,
                        "method": "lattice",
                        "shape": str(table.df.shape)
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"Lattice method failed for pages {start_page}-{end_page}: {str(e)}")

        # Try stream method (for borderless tables)
        try:
            tables_stream = camelot.read_pdf(
                file_path,
                pages=page_range,
                flavor='stream',
                edge_tol=50
            )

            for table in tables_stream:
                # Convert table to HTML using pandas
                html_content = table.df.to_html(index=False)

                # Only add if not a duplicate (check by page number)
                if not any(doc.metadata.get("page") == table.page for doc in documents):
                    doc = Document(
                        page_content=html_content,
                        metadata={
                            "category": "table",
                            "page": table.page,
                            "accuracy": table.accuracy,
                            "method": "stream",
                            "shape": str(table.df.shape)
                        }
                    )
                    documents.append(doc)
        except Exception as e:
            print(f"Stream method failed for pages {start_page}-{end_page}: {str(e)}")

        return documents

    except Exception as e:
        print(f"Error processing pages {start_page}-{end_page}: {str(e)}")
        return []


class TablePreProcessor:
    """Processes PDF documents using multiprocessing to extract tables."""

    def __init__(self, config: Dict[str, Any], num_processes: int = 3, file_path: str = None, filling_id: int = None):
        self.config = config
        self.num_processes = num_processes
        self.filling_id = filling_id
        return self.process_document(file_path, filling_id)
        

    def process_document(self, file_path: str, filling_id: int) -> List[Document]:
        try:
            with fitz.open(file_path) as pdf:
                total_pages = pdf.page_count

            # Split document into parts for parallel processing
            pages_per_part = math.ceil(total_pages / self.num_processes)
            parts = [
                (file_path, i * pages_per_part, min((i + 1) * pages_per_part, total_pages))
                for i in range(self.num_processes)
                if i * pages_per_part < total_pages
            ]

            with Pool(processes=len(parts)) as pool:
                results = pool.map(process_document_part, parts)

            # Flatten results - all documents from Camelot are already tables
            tables = []
            for result in results:
                tables.extend(result)

            return self.generate_table_summaries(tables, filling_id)

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return []


    def generate_table_summaries(self, tables: List[Document], filling_id: int) -> tuple[list[Document], list[tuple[str, str]]]:

        summaries = []
        table_list = []
        batch_size = self.config["table_batch_size"]

        for i in range(0, len(tables), batch_size):
            batch = tables[i:i + batch_size]

            if len(batch) == 1:
                prompt_template = self.config["table_single_summary_prompt"]
                content = batch[0].page_content

                single_prompt = PromptTemplate.from_template(prompt_template)

                summary_chain = (
                    single_prompt
                    | ChatOpenAI(
                        model=self.config["summary_model_name"],
                        temperature=self.config["summary_model_temperature"]
                    )
                    | StrOutputParser()
                )

                response = summary_chain.invoke({"element": content})
                batch_summaries = [response.strip()]
            else:
                combined_content = "\n\nTABLE SEPARATOR\n\n".join(
                    [f"Table {j+1}:\n{table.page_content}"
                        for j, table in enumerate(batch)]
                )

                batch_prompt = PromptTemplate.from_template(
                    self.config["table_batch_summary_prompt"]
                )

                summary_chain = (
                    batch_prompt
                    | ChatOpenAI(
                        model=self.config["summary_model_name"],
                        temperature=self.config["summary_model_temperature"]
                    )
                    | StrOutputParser()
                )

                response = summary_chain.invoke({"element": combined_content})
                batch_summaries = re.findall(r'\(([^)]+)\)', response)

                cleaned_summaries = []
                for summary in batch_summaries:
                    # Remove table references
                    for table_num in range(1, batch_size + 1):
                        summary = summary.replace(f"Table {table_num}:", "")
                    cleaned_summaries.append(summary)
                batch_summaries = cleaned_summaries

            for table, summary in zip(batch, batch_summaries):
                table_id = str(uuid.uuid4())

                # Store table in HTML format (already converted by Camelot)
                table_list.append((filling_id, table_id, table.page_content))

                summary_doc = Document(
                    page_content=summary.strip(),
                    metadata={
                        "type": "table_summary",
                        "table_id": table_id,
                        "filling_id": filling_id,
                        "source": table.metadata.get("source", "unknown"),
                        "company": table.metadata.get("company", "unknown"),
                        "page": table.metadata.get("page", "unknown"),
                        "extraction_method": table.metadata.get("method", "unknown"),
                        "accuracy": table.metadata.get("accuracy", 0.0)
                    }
                )
                summaries.append(summary_doc)

        return summaries, table_list

        

