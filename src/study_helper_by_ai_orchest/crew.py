import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class StudyHelperByAiOrchest():
    """StudyHelperByAiOrchest crew"""

    agents: list[BaseAgent]
    tasks: list[Task]
    
    load_dotenv()

    llm_gemini = LLM(
        model="gemini/gemini-3.5-flash",
        api_key= os.getenv("GEMINI_API_KEY"),
        base_url=os.getenv("GEMINI_BASE_URL")
    )

    llm_openrouter= LLM(
        model="openrouter/tencent/hy3:free",
        api_key= os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL")
    )

    @agent
    def context_giver_guy(self) -> Agent:
        return Agent(
            config=self.agents_config['context_giver_guy'], # type: ignore[index]
            verbose=True,
            llm= self.llm_openrouter
        )

    @task
    def context_giver_task(self) -> Task:
        return Task(
            config=self.tasks_config['context_giver_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the StudyHelperByAiOrchest crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
