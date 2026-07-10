from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class StudyHelperByAiOrchest():
    """StudyHelperByAiOrchest crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def context_giver_guy(self) -> Agent:
        return Agent(
            config=self.agents_config['context_giver_guy'], # type: ignore[index]
            verbose=True
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
