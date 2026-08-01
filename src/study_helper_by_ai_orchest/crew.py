import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import PDFSearchTool, CSVSearchTool
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

load_dotenv()

@CrewBase
class StudyHelperByAiOrchest():
    """StudyHelperByAiOrchest crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    
    embedder_config={  
            "provider": "google-generativeai",
            "config": {
            "model_name": "models/text-embedding-004",
            "api_key": os.getenv("GEMINI_API_KEY")
         }
     }

    sop_pdf_tool = PDFSearchTool(
        pdf="./database/Kebijakan_SOP_Customer_Service_NexusAIS.pdf",
        config=dict(embedder=embedder_config)
    )

    transaksi_csv_tool = CSVSearchTool(
        csv="./database/NexusAIS_Data_Transaksi.csv",
        config=dict(embedder=embedder_config)
    )

    pelanggan_csv_tool = CSVSearchTool(
        csv="./database/NexusAIS_Database_Pelanggan.csv",
        config=dict(embedder=embedder_config)
    )

    company_profile_knowledge= PDFKnowledgeSource(file_paths=['NexusAIS_Company_Profile_KnowledgeBase.pdf'],embedder=embedder_config)

    llm_gemini = LLM(
        model="gemini/gemini-3.1-flash-lite",
        api_key= os.getenv("GEMINI_API_KEY"),   
        base_url=os.getenv("GEMINI_BASE_URL")
    )

    llm_openrouter= LLM(
        model="openrouter/inclusionai/ling-3.0-flash:free",
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
            llm= self.llm_nvidia,
            tools=[self.sop_pdf_tool, self.transaksi_csv_tool,self.pelanggan_csv_tool],
            embedder=self.embedder_config
        )
    
    @agent
    def reasearcher(self) -> Agent:
        return Agent(
            config=self.agents_config['reasearcher'], # type: ignore[index]
            verbose=True,
            llm= self.llm_openrouter,
        )
    
    @agent
    def project_manager(self) -> Agent:
        return Agent(
            config= self.agents_config['project_manager'],
            verbose=True,
            llm= self.llm_gemini,
            allow_delegation=True,
            
        )

    @task
    def main_task(self) -> Task:
        return Task(
            config=self.tasks_config['main_task'], # type: ignore[index]
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the StudyHelperByAiOrchest crew"""
        return Crew(
            agents=[
                self.reasearcher(),
                self.data_analyst()
                ],
            tasks= [
                self.main_task()
                ],
            manager_agent= self.project_manager(),
            process=Process.hierarchical,
            planning=True,
            planning_llm=self.llm_gemini,
            verbose=True,
            knowledge_sources= self.company_profile_knowledge,
            embedder=self.embedder_config
            
        )
