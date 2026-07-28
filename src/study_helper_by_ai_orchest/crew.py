import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

text_source = TextFileKnowledgeSource(
    file_paths=["user_preference.txt"]
)

load_dotenv()

@CrewBase
class StudyHelperByAiOrchest():
    """StudyHelperByAiOrchest crew"""

    agents: list[BaseAgent]
    tasks: list[Task]
    
    search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))

    llm_gemini = LLM(
        model="gemini/gemini-3.1-flash-lite",
        api_key= os.getenv("GEMINI_API_KEY"),
        base_url=os.getenv("GEMINI_BASE_URL")
    )

    llm_openrouter= LLM(
        model="openrouter/tencent/hy3:free",
        api_key= os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL")
    )

    llm_nvidia= LLM(
        model="openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        api_key= os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL")
    )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst'], # type: ignore[index]
            verbose=True,
            llm= self.llm_nvidia
        )
    
    @agent
    def reasearcher(self) -> Agent:
        return Agent(
            config=self.agents_config['reasearcher'], # type: ignore[index]
            verbose=True,
            llm= self.llm_openrouter,
            tools= [self.search_tool]
        )
    
    @agent
    def project_manager(self) -> Agent:
        return Agent(
            config= self.agents_config['project_manager'],
            verbose=True,
            llm= self.llm_gemini,
            knowledge_sources=[text_source],
            embedder={  
                "provider": "google-generativeai",
                "config": {
                   "model_name": "gemini-embedding-001",
                   "api_key": os.getenv("GEMINI_API_KEY")
          }
         }
        )

    @task
    def main_task(self) -> Task:
        return Task(
            config=self.tasks_config['main_task'], # type: ignore[index]
        )
    
    # @task
    # def research_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['research_task'], # type: ignore[index]
    #         context= [self.context_giver_task()]
    #     )
    # @task
    # def adjust_task(self) -> Task:
    #     return Task(
    #         config= self.tasks_config['adjust_task'],
    #         context=[self.research_task()]
    #     ) 
    
    @crew
    def crew(self) -> Crew:
        """Creates the StudyHelperByAiOrchest crew"""
        return Crew(
            agents=[self.reasearcher(),self.data_analyst()],
            tasks= self.tasks,
            manager_agent= [self.project_manager()],
            process=Process.hierarchical,
            planning=True,
            verbose=True,
        )
