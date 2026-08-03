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
    """StudyHelperByAiOrchest Crew"""

    agents: list[BaseAgent] 
    tasks: list[Task]

    """StudyHelperByAiOrchest Embedder"""

    tools_embedder={
        "embedding_model": {
            "provider":  "google-generativeai",
            "config": {
                "model_name": "gemini-embedding-001",
                "api_key": os.getenv("GEMINI_API_KEY")


        },
    }}

    crew_embedder = {
    "provider": "google-generativeai",
    "config": {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "model_name": "gemini-embedding-001",
    }}

    """StudyHelperByAiOrchest Knowledge"""
    
    sop_pdf_tool = PDFSearchTool(
        pdf="./database/Kebijakan_SOP_Customer_Service_NexusAIS.pdf",
        config=tools_embedder)

    transaksi_csv_tool = CSVSearchTool(
        csv="./database/NexusAIS_Data_Transaksi.csv",
        config=tools_embedder)

    pelanggan_csv_tool = CSVSearchTool(
        csv="./database/NexusAIS_Database_Pelanggan.csv",
        config=tools_embedder)

    company_profile_knowledge= PDFKnowledgeSource(file_paths=['NexusAIS_Company_Profile_KnowledgeBase.pdf'],embedder=tools_embedder)

    """StudyHelperByAiOrchest Llm Models"""

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

    """StudyHelperByAiOrchest Agents"""
    @agent
    def internal_data_inspector(self) -> Agent:
        return Agent(
            config=self.agents_config['internal_data_inspector'], # type: ignore[index]
            llm= self.llm_nvidia,
            tools=[self.transaksi_csv_tool,self.pelanggan_csv_tool],
            embedder=self.crew_embedder,
            verbose=True,
        )
    
    @agent
    def policy_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['policy_expert'], # type: ignore[index]
            llm= self.llm_openrouter,
            tools=[self.sop_pdf_tool],
            verbose=True,
        )
    
    @agent
    def custommer_service_manager(self) -> Agent:
        return Agent(
            config= self.agents_config['custommer_service_manager'],
            llm= self.llm_gemini,
            allow_delegation=True,
            verbose=True,
            
        )

    """StudyHelperByAiOrchest Task"""

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
                self.internal_data_inspector(),
                self.policy_expert()
                ],
            tasks= [self.main_task()],
            manager_agent= self.custommer_service_manager(),
            knowledge_sources= [self.company_profile_knowledge],
            planning_llm=self.llm_gemini,
            process=Process.hierarchical,
            embedder=self.crew_embedder,
            planning=True,
            verbose=True,
            tracing=True
        )
