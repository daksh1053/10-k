"""
REACT agent implementation for autonomous information gathering.
"""

from typing import Dict, Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor

from .tools import CompanyDirectorsTool, WebSearchTool, VectorRerankerSearchTool, LinkedinScraperTool
from store.persistent import PersistentStore


class ReactAgent:
    """Basic REACT agent for information gathering and question answering."""

    def __init__(
        self,
        config: Dict[str, Any],
        director_strings: dict,
        persistent_store: PersistentStore,
    ):
        self.config = config
        self.director_strings = director_strings
        self.persistent_store = persistent_store
        self.agent = None
        self.react_agent_executor = None
        self.vector_tool: Optional[VectorRerankerSearchTool] = None

    def _initialize_tools(self, vector_store):
        """Initialize the tools available to the agent."""
        self.tools = []

        if self.config.get("use_director_tool", False):
            self.tools.append(
                CompanyDirectorsTool(self.config, self.director_strings, self.persistent_store)
            )

        if self.config.get("use_web_tool", False):
            self.tools.append(WebSearchTool(self.config))

        if self.config.get("use_retriever_tool", False):
            self.vector_tool = VectorRerankerSearchTool(
                self.config, vector_store, self.persistent_store
            )
            self.tools.append(self.vector_tool)

        if self.config.get("use_linkedin_scraper_tool", False):
            self.tools.append(LinkedinScraperTool(self.config, self.persistent_store))

    def initialize_agent(self, vector_store, ids_to_retrieve=None):
        """Initialize the REACT agent with tools and prompt."""
        self._initialize_tools(vector_store)
        if self.vector_tool:
            self.vector_tool.set_filter_ids(ids_to_retrieve)
        prompt = PromptTemplate.from_template(self.config["react_prompt_template"])
        llm = ChatOpenAI(
            model=self.config["react_model_name"],
            temperature=self.config["react_model_temperature"]
        )

        stop_sequence = self.config.get("react_agent_stop_sequence", True)
        self.agent = create_react_agent(llm, self.tools, prompt, stop_sequence=stop_sequence)
        self.agent.stream_runnable = False  # Avoid OpenAI streaming requirement

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
