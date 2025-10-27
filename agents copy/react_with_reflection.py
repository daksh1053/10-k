"""
REACT agent implementation with reflection mechanism for answer completeness.
"""

from typing import Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser

from .tools import CompanyDirectorsTool, WebSearchTool, VectorRerankerSearchTool, LinkedinScraperTool
from store.persistent import PersistentStore


class ReactWithReflection:
    """Enhanced REACT agent with reflection mechanism for answer completeness."""

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
        self.react_prompt_addition = PromptTemplate.from_template(
            config["react_prompt_reflection_addition"]
        )
        self.reflection_max_iterations = config["reflection_max_iterations"]
        self.react_max_iterations = config["react_agent_max_iterations"]
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
        """Initialize the REACT agent with reflection capabilities."""
        self._initialize_tools(vector_store)
        if self.vector_tool:
            self.vector_tool.set_filter_ids(ids_to_retrieve)
        prompt = PromptTemplate.from_template(self.config["react_prompt_template"])
        reflection_prompt = PromptTemplate.from_template(self.config["reflection_prompt_template"])
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
            verbose=self.config.get("react_verbosity", False),
            handle_parsing_errors=True,
            max_iterations=self.react_max_iterations
        )

        self.reflection_chain = (
            reflection_prompt
            | llm
            | StrOutputParser()
        )

    def _reflect(self, original_input: str, current_output: str) -> str:
        """Generate reflection on answer completeness."""
        reflection_input = {
            "original_input": original_input,
            "current_output": current_output
        }
        return self.reflection_chain.invoke(reflection_input)

    def run(self, question: str) -> str:
        """Run the agent with reflection loop for answer completeness."""
        if not self.react_agent_executor:
            raise ValueError("REACT agent not initialized. Call initialize_agent() first.")

        reflections = ""
        final_context = ""
        previous_iteration = ""
        iteration = ""

        for _ in range(self.reflection_max_iterations):
            # Run the REACT agent
            result = self.react_agent_executor.invoke({
                "input": question,
                "iteration": iteration,
            })
            final_context = result['output']
            previous_iteration = final_context

            # Generate reflection
            reflections = self._reflect(question, final_context)

            # Prepare next iteration with reflection
            iteration = self.react_prompt_addition.format(
                previous_iteration=previous_iteration,
                reflections=reflections
            )

            # Check if reflection indicates completion
            if "STOP" in reflections.upper():
                break

        return f"{previous_iteration}\n\n{final_context}"
