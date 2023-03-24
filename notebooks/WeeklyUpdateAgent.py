from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQAWithSourcesChain
import os
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import PromptTemplate


class Weeklupdate:
  def __init__(self):
    self.llm = OpenAI(temperature=0)
    self.documents = self.load_documents()
    self.db = self.setup_db()
    self.agent = self.create_agent()

  def load_documents(self):
    documents = []
    file_names = os.listdir("../data/weekly_updates")
    for file_name in file_names:
      loader = UnstructuredHTMLLoader("../data/weekly_updates/" + file_name)
      document = loader.load()[0]
      print(file_name)
      document.metadata["date"] = file_name[0:10]
      documents += document
    return documents
  
  def setup_db(self):
    persist_directory = 'data'
    embeddings = OpenAIEmbeddings()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(self.documents)
    docsearch = Chroma.from_documents(texts, embeddings, collection_name="weeklyupdatesv2", persist_directory=persist_directory)
    weekly_updates = VectorDBQAWithSourcesChain.from_chain_type(llm=self.llm, chain_type="map_reduce", vectorstore=docsearch)
    return weekly_updates

  def create_agent(self):
    tools = [
        Tool(
            name = "Weekly Update QA System",
            func= self.db,
            description="useful for when you need to answer questions about what Anthony the CEO said in a weekly update emails to the company. Input should be a fully formed question."
        )
    ]
    agent = initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True)
    return agent
  
  def query_agent(self, query):
    template = """
    You are helping someone understand a startup names Hearth better. You have access to all the weekly update emails send by the CEO. These emails are dated.
    The question you receive is: {query}\n
    Please give your sources.
    """
    question_prompt = PromptTemplate(
      input_variables=["query"], 
      template=template
    )
    prompt_with_query = question_prompt.format(query=query)
    return self.agent.run(prompt_with_query)
