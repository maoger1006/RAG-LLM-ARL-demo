import neo4j
from neo4j_graphrag.llm import OpenAILLM as LLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings as Embeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.generation import RagTemplate
import os
import json
import time


class Graph_RAG_Pipeline():
    
    def __init__(self):
        self.NEO4J_URI, self.NEO4J_USERNAME, self.NEO4J_PASSWORD = self.load_environment_variables()
        self.neo4j_driver = neo4j.GraphDatabase.driver(self.NEO4J_URI,
                                          auth=(self.NEO4J_USERNAME, self.NEO4J_PASSWORD))
        
        self.ex_llm=LLM(
            model_name="gpt-4o-mini",
            model_params={
                "response_format": {"type": "json_object"},
                "temperature": 0
            })
        
        self.embedder = Embeddings()
        
        # Optional Inputs: Schema & Prompt Template
        self.basic_node_labels = ["Object", "Entity", "Group", "Person", "Organization", "Place"]
        self.academic_node_labels = ["ArticleOrPaper", "PublicationOrJournal"]
        self.medical_node_labels = ["Anatomy", "BiologicalProcess", "Cell", "CellularComponent",
                              "CellType", "Condition", "Disease", "Drug",
                              "EffectOrPhenotype", "Exposure", "GeneOrProtein", "Molecule",
                              "MolecularFunction", "Pathway"]
        
        self.node_labels = self.basic_node_labels + self.academic_node_labels + self.medical_node_labels
        
        self.rel_types = ["ACTIVATES", "AFFECTS", "ASSESSES", "ASSOCIATED_WITH", "AUTHORED", "BIOMARKER_FOR"]
        
        self.vector_retriever = VectorRetriever(
            self.neo4j_driver,
            index_name="text_embeddings",
            embedder=self.embedder,
            return_properties=["text"],
        )
        
        self.vc_retriever = VectorCypherRetriever(
            self.neo4j_driver,
            index_name="text_embeddings",
            embedder=self.embedder,
            retrieval_query="""
            //1) Go out 2-3 hops in the entity graph and get relationships
            WITH node AS chunk
            MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,2}()
            UNWIND relList AS rel

            //2) collect relationships and text chunks
            WITH collect(DISTINCT chunk) AS chunks,
            collect(DISTINCT rel) AS rels

            //3) format and return context
            RETURN '=== text ===\n' + apoc.text.join([c in chunks | c.text], '\n---\n') + '\n\n=== kg_rels ===\n' +
            apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], '\n---\n') AS info
            """
            )
        
        
    def load_environment_variables(self):
        #
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

    # Clear the Neo4j database before running the pipeline
    def clear_neo4j_database(self): 
        with self.neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Neo4j database cleared!")
            
    # 1. Build KG and Store in Neo4j Database
    async def run_kg_builder_pdf(self, file_path):
        
        kg_builder_pdf = SimpleKGPipeline(
            llm=self.ex_llm,
            driver=self.neo4j_driver,
            embedder=self.embedder,
            from_pdf=True
        )
        pdf_result = await kg_builder_pdf.run_async(file_path=file_path)
        
        print(f"Result: {pdf_result}")
            
    def kg_create_vector_index(self):
        # 2. Create Vector Index
        create_vector_index(self.neo4j_driver, name="text_embeddings", label="Chunk",
                   embedding_property="embedding", dimensions=1536, similarity_fn="cosine")
        
    
    # 2. KG Retriever
    def run_vector_retriever(self, top_k):
        
        # self.vector_retriever = VectorRetriever(
        #     self.neo4j_driver,
        #     index_name="text_embeddings",
        #     embedder=self.embedder,
        #     return_properties=["text"],
        # )
        
        vector_res = self.vector_retriever.get_search_results(query_text = "How is Graph RAG being used?", top_k=top_k)
        
        return vector_res
    
    def run_VC_retriever(self):
        
        # self.vc_retriever = VectorCypherRetriever(
        #     self.neo4j_driver,
        #     index_name="text_embeddings",
        #     embedder=self.embedder,
        #     retrieval_query="""
        #     //1) Go out 2-3 hops in the entity graph and get relationships
        #     WITH node AS chunk
        #     MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,2}()
        #     UNWIND relList AS rel

        #     //2) collect relationships and text chunks
        #     WITH collect(DISTINCT chunk) AS chunks,
        #     collect(DISTINCT rel) AS rels

        #     //3) format and return context
        #     RETURN '=== text ===\n' + apoc.text.join([c in chunks | c.text], '\n---\n') + '\n\n=== kg_rels ===\n' +
        #     apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], '\n---\n') AS info
        #     """
        #     )
        vc_res = self.vc_retriever.get_search_results(query_text = "How is precision medicine applied to Lupus?", top_k=3)
        
        return vc_res
    
    def generate_answer(self, question, top_k):
        llm = LLM(model_name="gpt-4o",  model_params={"temperature": 0.3})

        rag_template = RagTemplate(template='''Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned.

        # Question:
        {query_text}

        # Context:
        {context}

        # Answer:
        ''', expected_inputs=['query_text', 'context'])

        v_rag  = GraphRAG(llm=llm, retriever=self.vector_retriever, prompt_template=rag_template)
        vc_rag = GraphRAG(llm=llm, retriever=self.vc_retriever, prompt_template=rag_template)

        vector_answer = v_rag.search(question, retriever_config={"top_k": top_k}).answer
        vc_answer = vc_rag.search(question, retriever_config={"top_k": top_k}).answer
    
        return vector_answer, vc_answer
    
    
async def main():
    # 1. Clear the Neo4j database
    analyzer = Graph_RAG_Pipeline()
    analyzer.load_environment_variables() 
    # analyzer.clear_neo4j_database()
    # await analyzer.run_kg_builder_pdf(file_path="./Test_files/Thesis Mingyang.pdf")
    # # await analyzer.run_kg_builder_pdf(file_path="./Test_files/Paper_review.")
    # analyzer.kg_create_vector_index()
    
    vector_answer , vc_answer = analyzer.generate_answer(question="How does finger tissue thickness—specifically variations in the dermis and subcutaneous layers—affect the performance of both transmission and reflection pulse oximetry?", top_k=5)
    
    # print (f"Vector Answer: {vector_answer}")
    # print ("\n===============================\n")
    print (f"Vector + Cypher Answer: {vc_answer}")
    
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())