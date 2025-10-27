"""
Tools for REACT agents - web search, director lookup, and vector retrieval.
"""

import os
import requests
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from store.persistent import PersistentStore

class CompanyDirectorsTool(BaseTool):
    """Tool for retrieving company director information with LinkedIn profiles."""
    
    name: str
    description: str
    director_strings_dict: dict
    serp_api_obj: Any
    persistent_store: PersistentStore
    llm: Any
    parser: Any
    name_extraction_prompt: Any
    name_extraction_chain: Any

    def __init__(
        self,
        config: Dict[str, Any],
        director_strings_dict: dict,
        persistent_store: Optional[PersistentStore] = None
    ):
        available_companies = director_strings_dict.keys()

        super().__init__(
            name=config["director_tool_name"],
            description=f"{config['director_tool_description']} Available companies: {', '.join(available_companies)}",
            director_strings_dict=director_strings_dict,
            serp_api_obj=SerpAPIWrapper(),
            persistent_store=persistent_store or PersistentStore(),
            llm=ChatOpenAI(
                model=config["name_extraction_model"],
                temperature=config["name_extraction_model_temperature"]
            ),
            parser=CommaSeparatedListOutputParser(),
            name_extraction_prompt=PromptTemplate.from_template(config["name_extraction_prompt"]),
            name_extraction_chain=None
        )
        self.name_extraction_chain = self.name_extraction_prompt | self.llm | self.parser

    def _run(self, query: str) -> str:
        try:
            company_name, include_linkedin = query.split(',')
            company_name = company_name.strip()
            include_linkedin = include_linkedin.strip().lower() == 'true'
        except ValueError:
            return "Invalid format. Use: company_name, true/false"
        
        company_snippet = self.director_strings_dict.get(company_name)

        if not company_snippet:
            return f"No director information found for {company_name}"

        director_names = self.name_extraction_chain.invoke({"text": company_snippet})

        if not include_linkedin:
            return f"Directors of {company_name}: {', '.join(director_names)}"

        director_handles = []
        for name in director_names:
            linkedin_handle = self._get_linkedin_handle(name, company_name)
            director_handles.append(f"{name} (LinkedIn: {linkedin_handle})")

        return f"Directors of {company_name}: {'; '.join(director_handles)}"

    def _get_linkedin_handle(self, name: str, company: str) -> str:
        cache_key = f"{name}_{company}"
        cached_result = self.persistent_store.get_director_linkedin_cache(cache_key)
        if cached_result:
            return cached_result

        try:
            results = self.serp_api_obj.results(f'"{name}" {company} site:linkedin.com/in/')
            link = results.get("organic_results", [{}])[0].get("link", "Profile not found")
            self.persistent_store.set_director_linkedin_cache(cache_key, link)
            return link
        except Exception as e:
            return f"Error finding LinkedIn profile: {str(e)}"


class WebSearchTool(BaseTool):
    """Tool for performing web searches."""
    
    name: str
    description: str
    serp_api: Any
    num_results: int

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name=config["web_tool_name"],
            description=config["web_tool_description"],
            serp_api=SerpAPIWrapper(),
            num_results=config["num_web_tool_results"]
        )

    def _run(self, query: str) -> str:
        try:
            results = self.serp_api.results(query)
            return self._format_results(results)
        except Exception as e:
            return f"Error performing web search: {str(e)}"

    def _format_results(self, results: Dict) -> str:
        formatted = []
        
        for result in results.get("organic_results", [])[:self.num_results]:
            formatted.append(
                f"Title: {result.get('title', 'N/A')}\n"
                f"Snippet: {result.get('snippet', 'N/A')}\n"
                f"Link: {result.get('link', 'N/A')}"
            )
        return "\n\n".join(formatted)


class VectorRerankerSearchTool(BaseTool):
    """Tool for retrieving documents from vector store with reranking."""

    name: str
    description: str
    vector_store: Any
    chunk_k: int
    table_k: int
    ids_to_retrieve: Any
    reranker: Any
    persistent_store: Any

    def __init__(
        self,
        config: Dict[str, Any],
        vector_store: Any,
        persistent_store: Optional[PersistentStore] = None,
    ):
        from langchain_cohere import CohereRerank

        chunk_k = config.get("chunk_k", 0)
        table_k = config.get("table_k", 0)

        super().__init__(
            name=config["retriever_tool_name"],
            description=config["retriever_tool_description"],
            vector_store=vector_store,
            chunk_k=int(chunk_k) if chunk_k is not None else 0,
            table_k=int(table_k) if table_k is not None else 0,
            ids_to_retrieve=config.get("ids_to_retrieve"),
            reranker=CohereRerank(model=config["reranker_model"]),
            persistent_store=persistent_store or PersistentStore()
        )

    def set_filter_ids(self, ids: Optional[List[int]]) -> None:
        self.ids_to_retrieve = ids

    def _run(self, query: str) -> str:
        try:
            chunk_k = int(self.chunk_k) if self.chunk_k else 0
            table_k = int(self.table_k) if self.table_k else 0
            chunk_docs, table_docs = self.vector_store.retriever(
                query,
                chunk_k,
                table_k,
                self.ids_to_retrieve
            )

            if chunk_docs and chunk_k > 0:
                chunk_docs = self.rerank_documents(chunk_docs, query)[:chunk_k]
            else:
                chunk_docs = []
            if table_docs and table_k > 0:
                table_docs = self.rerank_documents(table_docs, query)[:table_k]
            else:
                table_docs = []

            context_string = self._format_docs(chunk_docs + table_docs)
            context_string += "\n\n"
            context_string += self._format_tables(table_docs)

            return context_string.strip()
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"

    def rerank_documents(self, docs: List[Document], query: str) -> List[Document]:
        """Rerank documents using Cohere reranker"""
        if not docs:
            return docs
        try:
            reranked = self.reranker.compress_documents(docs, query)
            return reranked
        except Exception as e:
            print(f"Reranking failed: {e}")
            return docs

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
            if table_id:
                try:
                    table_data = self.persistent_store.get_table_by_id(table_id)
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


class LinkedinScraperTool(BaseTool):
    """Tool for scraping LinkedIn profiles using RapidAPI Fresh LinkedIn Profile Data API."""
    
    name: str
    description: str
    api_key: str
    persistent_store: PersistentStore

    def __init__(
        self,
        config: Dict[str, Any],
        persistent_store: Optional[PersistentStore] = None
    ):
        super().__init__(
            name=config["linkedin_scraper_tool_name"],
            description=config["linkedin_scraper_tool_description"],
            api_key=os.environ.get("RAPIDAPI_API_KEY"),
            persistent_store=persistent_store or PersistentStore()
        )

    def _run(self, query: str) -> str:
        linkedin_url = query.strip()

        cached_result = self.persistent_store.get_linkedin_scraper_cache(linkedin_url)
        if cached_result:
            return cached_result

        try:
            data = self._fetch_linkedin_data(linkedin_url)
            result = self._format_background_info(data)
            self.persistent_store.set_linkedin_scraper_cache(linkedin_url, result)
            return result
        except Exception as e:
            return f"Error retrieving background info: {str(e)}"

    def _fetch_linkedin_data(self, linkedin_url: str) -> dict:
        url = "https://fresh-linkedin-profile-data.p.rapidapi.com/get-profile-public-data"
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }
        params = {
            'linkedin_url': linkedin_url,
            'include_skills': 'false',
            'include_certifications': 'false',
            'include_publications': 'false',
            'include_honors': 'false',
            'include_volunteers': 'false',
            'include_projects': 'false',
            'include_patents': 'false',
            'include_courses': 'false',
            'include_organizations': 'false',
            'include_profile_status': 'false',
            'include_company_public_url': 'false'
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json().get("data", {})
        raise Exception(f"API call failed: {response.status_code} {response.text}")

    def _format_background_info(self, data: dict) -> str:
        companies = [exp.get("company") for exp in data.get("experiences", [])[:5] if exp and exp.get("company")]
        education = [edu.get("school") for edu in data.get("educations", [])[:3] if edu and edu.get("school")]

        return (
            f"Professional Experience: {', '.join(companies)}\n"
            f"Education: {', '.join(education)}"
        )
