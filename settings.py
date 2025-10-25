"""
Configuration settings for the 10-K Analysis System.
Contains all model, API, and processing parameters.
"""

from langchain_core.documents import Document


DEFAULT_CONFIG = {
   
    "chunk_k": 5,
    "table_k": 5,
    "ids_to_retrieve": None,

    "rag_answer_model": "gpt-5",
    "rag_answer_model_temperature": 0.7,

    "react_model_name": "gpt-5",
    "react_model_temperature": 0,

    # Document processing settings
    "chunk_size": 500,
    "chunk_overlap": 50,
    "user_agent_header": "FinTech-RAG-System/1.0 (research@company.com)",

    # Embedding settings
    "embedding_api_url": "https://api-inference.huggingface.co/models/BAAI/bge-base-en-v1.5",
    "embedding_dim": 768,

    # Default document for vector store initialization
    "default_document": Document(
        page_content="This is a default document.",
        metadata={"source": "default"}
    ),

    # Reranker settings
    "reranker_model": "rerank-english-v3.0",

    # Model settings

    "query_decomposer_model": "gpt-5-mini",
    "query_decomposer_model_temperature": 0.8,
    
    "name_extraction_model": "gpt-5-mini",
    "name_extraction_model_temperature": 0.4,
    

    
    "summary_model_name": "gpt-5-mini",
    "summary_model_temperature": 0.3,



    # RAG prompt template
    "rag_prompt_template": """
    Give an answer for the `Question` using only the given `Context`. Use information relevant to the query from the entire context.
    Provide a detailed answer with thorough explanations, avoiding summaries.

    Question: {question}

    Context: {context}

    Answer:
    """,

    # Query decomposition prompt
    "subquery_prompt_template": """
    Break down the `Question` into multiple sub-queries. Use the guidelines given below to help in the task.

    1. The set of sub-queries together capture the complete information needed to answer the question.
    2. Each sub-query should ask for just one piece of information about one specific company.
    3. For each sub-query, only mention the information you're trying to get. Don't use verbs like "retrieve" or "find".
    4. Include the company name mentioned in each sub-query.
    5. Do not include any references to data sources in your sub-queries.

    Enclose the sub-query in angle brackets. For example:
    <sub-query 1>
    <sub-query 2>

    Question: {question}

    Begin:
    """,

    # Name extraction prompt
    "name_extraction_prompt": """
    Extract and list the names of all individuals with the title 'Director' from the following text, excluding any additional information such as dates or signatures.
    Present the names as a simple, comma-separated list.

    {text}
    """,

    # Table summarization settings
    "table_batch_size": 3,

    # Table summary prompts
    "table_batch_summary_prompt": """
    You are an assistant tasked with summarizing multiple tables. For each table below, generate a concise summary enclosed in brackets ().

    {element}
    """,

    "table_single_summary_prompt": """
    You are an assistant tasked with summarizing a table. Generate a concise summary of the table below that captures the key information and structure.

    {element}
    """,

    # REACT agent settings
    "react_prompt_template": """Your task is to gather relevant information to build context for the question. Focus on collecting details related to the question.
    Gather as much context as possible before formulating your answer.

    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer

    Thought: you should always think about what to do

    Action: the action to take, should be one of [{tool_names}]

    Action Input: the input to the action

    Observation: the result of the action

    ... (this Thought/Action/Action Input/Observation can repeat N times)

    Thought: I now know the final answer

    Final Answer: the final answer to the question.

    Follow these steps:

    Begin!

    Question: {input}

    Thought:{agent_scratchpad}
    """,

    "react_verbosity": True,
    "react_agent_max_iterations": 25,

    # Reflection settings
    "reflection_prompt_template": """Your task is to analyze whether the `Answer` is missing some information related to the `Question`.
    Give feedback on the missing requirements of the answer. Mention only the essential information.

    Here is the previous interaction:
    Question: {original_input}
    Answer: {current_output}

    Reflection:
    Provide brief, concise thoughts on what additional information needs to be collected in the next iteration.

    Based on your reflection, conclude with one of the following actions:

    If the current Answer provides sufficient information for Original Input, state "STOP".
    If further refinement is needed, provide 2-3 brief thoughts for improvement, each on a new line, and end with "CONTINUE".
    """,

    "react_prompt_reflection_addition": """Improve `Previous Answer` based on `Reflections`. Don't look for information already present in `Previous Answer`.
    Formulate a new Final Answer.

    Reflections: {reflections}

    Previous Answer: {previous_iteration}
    """,

    "reflection_max_iterations": 2,

    # Tool settings
    "use_director_tool": True,
    "director_tool_name": "Company Directors Information",
    "director_tool_description": "Retrieve the names of company directors for a chosen company. Optionally, their LinkedIn handles can also be included. Use the format: company_name, true/false.",

    "use_web_tool": True,
    "web_tool_name": "WebSearch",
    "web_tool_description": "Performs a web search on the query.",
    "num_web_tool_results": 6,

    "use_retriever_tool": True,
    "retriever_tool_name": "Vector Reranker Search",
    "retriever_tool_description": "Retrieves information from an embedding based vector DB containing financial data and company information. Structure query as a sentence asking information, don't use words like retrieve.",
    "num_retriever_tool_results": 6,

    "use_linkedin_scraper_tool": True,
    "linkedin_scraper_tool_name": "Director's previous work and education",
    "linkedin_scraper_tool_description": "Retrieves director's education and work experience using their LinkedIn URL. Use the format: url",

    # Environment variable keys
    "required_env_vars": {
        "OPENAI_API_KEY": "OpenAI API key for LLM operations",
        "COHERE_API_KEY": "Cohere API key for reranking",
        "HF_API_TOKEN": "HuggingFace API token for embeddings",
        "SERPAPI_API_KEY": "SerpAPI key for web search",
        "UNSTRUCTURED_API_KEY": "Unstructured API key for document processing",
        "RAPIDAPI_API_KEY": "RapidAPI key for LinkedIn scraping",
        "LANGCHAIN_API_KEY": "LangSmith API key for tracing (optional)",
    },

    # Optional LangSmith tracing
    "langsmith_tracing": {
        "enabled": False,
        "project_name": "10K-Analysis-System",
        "endpoint": "https://api.smith.langchain.com"
    },
}