"""
Query decomposition to break down complex questions.
"""

import re
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def decompose_query(config: Dict[str, Any], question: str) -> List[str]:
    llm = ChatOpenAI(
        model=config["query_decomposer_model"], 
        temperature=config["query_decomposer_model_temperature"]
    )
    prompt = PromptTemplate.from_template(config["subquery_prompt_template"])
    chain = (prompt | llm | StrOutputParser())
    response = chain.invoke({"question": question})
    return re.findall(r'<(.*?)>', response, re.DOTALL)
