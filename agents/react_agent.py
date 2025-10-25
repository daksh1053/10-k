"""
REACT agent implementation for autonomous information gathering.
"""

from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor

from .tools import CompanyDirectorsTool, WebSearchTool, VectorRerankerSearchTool, LinkedinScraperTool


class ReactAgent:
    """Basic REACT agent for information gathering and question answering."""

    def __init__(self, config: Dict[str, Any], director_strings: dict):
        self.config = config
        self.director_strings = director_strings
        self.agent = None
        self.react_agent_executor = None

    def _initialize_tools(self, retriever, vector_store=None):
        """Initialize the tools available to the agent."""
        self.tools = []

        if self.config.get("use_director_tool", False):
            self.tools.append(CompanyDirectorsTool(self.config, self.director_strings))

        if self.config.get("use_web_tool", False):
            self.tools.append(WebSearchTool(self.config))

        if self.config.get("use_retriever_tool", False):
            self.tools.append(VectorRerankerSearchTool(self.config, retriever, vector_store))

        if self.config.get("use_linkedin_scraper_tool", False):
            self.tools.append(LinkedinScraperTool(self.config))

    def initialize_agent(self, retriever, vector_store=None):
        """Initialize the REACT agent with tools and prompt."""
        self._initialize_tools(retriever, vector_store)
        prompt = PromptTemplate.from_template(self.config["react_prompt_template"])
        llm = ChatOpenAI(
            model=self.config["react_model_name"],
            temperature=self.config["react_model_temperature"]
        )

        self.agent = create_react_agent(llm, self.tools, prompt)

        self.react_agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.config.get("react_verbosity", False)
        )

    def run(self, question: str) -> str:
        """Run the agent with a question."""
        if not self.react_agent_executor:
            raise ValueError("React agent not initialized. Call initialize_agent() first.")
        return self.react_agent_executor.invoke({"input": question})['output']