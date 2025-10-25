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
import warnings
import pandas as pd
import logging
from tqdm import tqdm

class TextPreProcessor:
    """Preprocesses SEC 10-K filings by loading and chunking documents."""
    
    def __init__(self, config: Dict[str, Any], company_filing_urls: List[Tuple[str, str]], filling_id: int):
        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]
        self.user_agent_header = config["user_agent_header"]
        self.config = config
        self.company_filing_urls = company_filing_urls
        self.filling_id = filling_id
        

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
        seen_tables: set[tuple[int, str]] = set()

        # Create output directory for tables
        # output_dir = "extracted_tables"
        # os.makedirs(output_dir, exist_ok=True)
        # table_counter = 0

        def _add_table(table, method: str):
            # nonlocal table_counter
            # Convert table to HTML using pandas
            html_content = table.df.to_html(index=False)
            table_key = (table.page, hash(html_content))
            if table_key in seen_tables:
                return
            seen_tables.add(table_key)

            # Save table as readable text file
            # table_counter += 1
            # table_filename = f"{output_dir}/page_{table.page:04d}_table_{table_counter:03d}_{method}.txt"
            # with open(table_filename, 'w', encoding='utf-8') as f:
            #     f.write(f"Page: {table.page}\n")
            #     f.write(f"Method: {method}\n")
            #     f.write(f"Shape: {table.df.shape}\n")
            #     f.write(f"Accuracy: {getattr(table, 'accuracy', 'N/A')}\n")
            #     f.write("-" * 80 + "\n\n")
            #     f.write(table.df.to_string(index=False))

            doc = Document(
                page_content=html_content,
                metadata={
                    "category": "table",
                    "page": table.page,
                    "accuracy": getattr(table, "accuracy", None),
                    "method": method,
                    "shape": str(table.df.shape)
                }
            )
            documents.append(doc)

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
                _add_table(table, "lattice")
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
                _add_table(table, "stream")
        except Exception as e:
            print(f"Stream method failed for pages {start_page}-{end_page}: {str(e)}")

        return documents

    except Exception as e:
        print(f"Error processing pages {start_page}-{end_page}: {str(e)}")
        return []


class TablePreProcessor:
    """Processes PDF documents using optional multiprocessing to extract tables."""

    def __init__(self, config: Dict[str, Any], num_processes: int = 1, file_path: str = None, filling_id: int = None):
        warnings.filterwarnings("ignore", category=UserWarning, module="camelot.parsers.base")
        warnings.filterwarnings("ignore", category=UserWarning, module="camelot")
        logging.getLogger("camelot").setLevel(logging.ERROR)
        self.config = config
        self.num_processes = max(1, num_processes or 1)
        self.filling_id = filling_id
        self.file_path = file_path
        

    def process_document(self, file_path: str, filling_id: int) -> List[Document]:
        try:
            with fitz.open(file_path) as pdf:
                total_pages = pdf.page_count

            # Determine page chunking for Camelot processing
            pages_per_part = max(
                1,
                min(
                    self.config.get("table_pages_per_chunk", 5),
                    math.ceil(total_pages / self.num_processes)
                ),
            )

            parts: List[Tuple[str, int, int]] = []
            start = 0
            while start < total_pages:
                end = min(start + pages_per_part, total_pages)
                parts.append((file_path, start, end))
                start = end

            results = []
            if self.num_processes > 1 and len(parts) > 1:
                with Pool(processes=min(len(parts), self.num_processes)) as pool:
                    for result in tqdm(
                        pool.imap(process_document_part, parts),
                        total=len(parts),
                        desc="Processing PDF pages",
                        unit="part",
                    ):
                        results.append(result)
            else:
                for part in tqdm(parts, desc="Processing PDF pages", unit="part"):
                    results.append(process_document_part(part))

            # Flatten results - all documents from Camelot are already tables

            tables = []
            for result in results:
                tables.extend(result)

            print(f"Found {len(tables)} tables")

            return self.generate_table_summaries(tables, filling_id)

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return []


    def generate_table_summaries(self, tables: List[Document], filling_id: int) -> tuple[list[Document], list[tuple[str, str]]]:

        summaries = []
        table_list = []
        batch_size = self.config["table_batch_size"]

        for i in tqdm(range(0, len(tables), batch_size), desc="Generating table summaries", unit="batch"):
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

        
