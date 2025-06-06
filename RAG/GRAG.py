
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector, Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI

import os
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader

class GRAG_pipeline:
   def __init__(self):
      self.load_environment_variables()
      self.initialize_directories()
      self.llm_name = self.select_llm()
      self.embeddings = OpenAIEmbeddings()
      self.vector_db = None
      self.NEO4J_URI, self.NEO4J_USERNAME, self.NEO4J_PASSWORD = self.load_environment_variables()
      self.neo4j_driver = Neo4jGraph.GraphDatabase.driver(self.NEO4J_URI,
                                          auth=(self.NEO4J_USERNAME, self.NEO4J_PASSWORD))

   # OPAI API key and Neo4j credentials
   def load_environment_variables(self):
      """Load API keys and environment variables."""
      # _ = load_dotenv(find_dotenv())
      load_dotenv(dotenv_path=".env", override=True)
      openai.api_key = os.getenv('OPENAI_API_KEY')
        
      variables = {}
      # Read the file line by line
      with open("./api/Neo4j-8f713933-Created-2025-03-06.txt", "r") as file:
         for line in file:
               line = line.strip()  # Remove whitespace and newline
               if line and "=" in line and not line.startswith("#"):  # Ignore empty lines and comments
                  key, value = line.split("=", 1)  # Split at first "="
                  variables[key.strip()] = value.strip()   
                  
      NEO4J_URI = variables.get("NEO4J_URI")
      NEO4J_USERNAME = variables.get("NEO4J_USERNAME")
      NEO4J_PASSWORD = variables.get("NEO4J_PASSWORD")
      
      return NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
   
   
   
   def load_documents(self, files):
      """Load documents from PDF and Markdown (.md) files."""
      docs = []
      seen_files = set()
      for file in files:
         ext = os.path.splitext(file)[-1].lower()
         base_name = os.path.splitext(os.path.basename(file))[0]
         
         if base_name in seen_files:
               continue
         elif ext in ((".md", ".txt")) and base_name not in seen_files:
               loader = TextLoader(file, encoding="utf-8")
               seen_files.add(base_name)
         elif ext == ".pdf" and base_name not in seen_files:
               loader = PyPDFLoader(file)
               seen_files.add(base_name)
         else:
               raise ValueError(f"Unsupported file format: {ext}")
         
         docs.extend(loader.load())
         
         print (f"Loaded {base_name}")
      return docs
   
   # Hybrid Search: combines vector search with fulltext search 
   def create_neo4j_vector_db(self, docs):
      """Create a Neo4j vector database from the loaded documents."""
      db = Neo4jVector.from_documents(
         documents=docs,
         embedding=self.embeddings,
         url=self.NEO4J_URI,
         username=self.NEO4J_USERNAME,
         password=self.NEO4J_PASSWORD,
         
      )

      return db
   
   
         
         

   

   
   
   
   



