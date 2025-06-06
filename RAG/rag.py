import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import warnings
warnings.filterwarnings("ignore")
import time
import gc



# Class definition
class RAG_pipeline:
    def __init__(self):
        self.load_environment_variables()
        self.initialize_directories()
        self.llm_name = self.select_llm()
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = None

    def load_environment_variables(self):
        """Load API keys and environment variables."""
        # _ = load_dotenv(find_dotenv())
        load_dotenv(dotenv_path=".env", override=True)
        openai.api_key = os.getenv('OPENAI_API_KEY')

    def initialize_directories(self):
        """Create necessary directories."""
        self.current_dir = os.getcwd()
        self.docs_dir = os.path.join(self.current_dir, 'docs')
        self.chroma_dir = os.path.join(self.docs_dir, 'chroma')

        # Create 'docs' and 'chroma' directories if they don't exist
        if not os.path.isdir(self.docs_dir):
            os.mkdir(self.docs_dir)
            print(f"Created directory: {self.docs_dir}")
        if not os.path.exists(self.chroma_dir):
            os.mkdir(self.chroma_dir)
            print(f"Created subfolder: {self.chroma_dir}")

    def select_llm(self):
        """Select which language model to use based on the date."""

        return "gpt-4.1-mini"

    def load_documents_simple(self, pdf_files):
        """Load documents from PDF files."""
        loaders = [PyPDFLoader(pdf) for pdf in pdf_files]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        return docs
    
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
    
    
    def split_documents(self, docs, chunk_size=1024):
        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2", 
            model_max_length=4096  # or 4096, etc.
        )
        chunk_overlap = int(chunk_size / 10)  

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(docs)


    def create_vector_db(self, docs, persist_directory):
        """Create a vector database from document chunks."""
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=persist_directory    # maintain the previous uploaded documents
        )
        self.vector_db = vectordb
        return vectordb

    def build_qa_chain(self, response_type):
        """Build a QA chain for querying the documents."""
        if response_type == "concise mode":
            
            prompt_template = """As an adaptive knowledge assistant, I provide **clear, concise, and context-aware responses** within 1 sentences (no need to be a complete sentence) based on the question type.

                                ## **Response Strategy**
                                - If the question contains **factual errors or misunderstandings**, respond with:
                                - A **direct clarification** (e.g., *"The correct author is..."*).
                                - If the question is straightforward, provide a **short, direct answer** in **1 sentence** max.
                                - Use a straightforward tone with **natural transitions** 

                                ---

                                ### **Context from Previous Conversation**  
                                {context}  

                                ### **User's Question**  
                                {question}  

                                ### **Response**  """
                                
        elif response_type == "detail mode":
            
            
            prompt_template = """As an adaptive knowledge assistant, I provide clear and natural responses that remain concise yet include extra details when needed.

                ## **Response Strategy**
                - If the question contains **factual errors or misunderstandings**, respond with a **direct clarification** (e.g., *"The correct author is..."*).
                - If the question is straightforward, provide a **short, direct answer** in **1 sentence max**.
                - For more complex questions, expand your answer with additional context, examples, and insights, while keeping the response clear and natural.

                ---

                ### **Context from Previous Conversation**
                {context}

                ### **User's Question**
                {question}

                ### **Response**
                """
                    
                      
        self.QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(model_name=self.llm_name, temperature=0.4)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vector_db.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT}
        )
        
        
    def build_qa_chain_correct(self):
        # """Build a QA chain for identifying error in the conversation."""
        
        
        prompt_template = """Follow these instructions STRICTLY:
        1. If the input is EMPTY (no text), say 'correct'.
        2. Consistency Check (Primary Rule):** If the Input **IS related** to the Context:* If the Input is **consistent** with the information presented in the Context, respond ONLY with the word `correct`.* If the Input **directly and clearly contradicts** specific information stated in the Context, respond ONLY with a brief corrected phrase (MAX 10 words) that accurately reflects the information given in the Context. Base this correction *strictly* on the provided Context.
        3. If the input contains OBVIOUS factual errors (e.g. "sun rises from west") or clearly contradicts or does not relate to the provided knowledge base context, respond ONLY with the corrected phrase (MAX 10 words).
        4. If the input is ambiguous, incomplete, or not related to the knowledge base context, respond with 'correct'.
        5. Only if the input clearly aligns with the provided context and has a factual error, provide a direct correction in one short phrase.
        6. DO NOT include explanations or prefixes like "Correction:".

        Examples:
        Input: "Boeing has been selected to build what may turn out to be the most expensive jet flighter in history." → Response: "correct"
        Input: "To address this, we developed Memoro, a wearable video-based assistant with a concise user interface." → Response: "Audio-based"
        Input: "Fire is cold" → Response: "Fire is hot"
        Input: "Cats can fly" → Response: "Cats cannot fly"
        Input: "" → Response: "correct"
        Input: "It might rain" → Response: "correct"

        ---

        ### **Context from Previous Conversation**
        {context}

        ### **User's Question or Input**
        {question}

        ### **Response**
        """
        
        

            
                      
        self.QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(model_name=self.llm_name, temperature=0.0)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vector_db.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT}
        )
    
    
    def generate_answer(self, format_question, k=3):
        
        """Retrieve context, similarity scores, and generate an answer for the given question."""
        start_time = time.time()
        
        # Retrieve the top-k most relevant documents with similarity scores
        docs_with_scores = self.vector_db.similarity_search_with_score(format_question, k=k)

        # Extract document content and similarity scores
        context_window = ' '.join([doc[0].page_content for doc in docs_with_scores])
        
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"Document {i+1} (Similarity Score: {score:.4f}):\n{doc.page_content}\n")
            
        similarity_scores = [doc[1] for doc in docs_with_scores]  # Extracting scores
        
        end_time = time.time()
        
        source_pdfs = [doc[0].metadata.get("source", "Unknown Source") for doc in docs_with_scores]  # Extract source PDF name
        
        # Format the prompt with the retrieved context
        prompt = self.QA_CHAIN_PROMPT.format(context=context_window, question=format_question)
        
        # Generate the response
        result = self.qa_chain({"query": prompt})
        

        Time_taken = end_time - start_time
        
        # Return both the answer and similarity scores
        return result['result'], similarity_scores, source_pdfs, Time_taken
    
    def get_score(self, question, k=3):   # score is the distance between two vectors, the lower the better
        """Retrieve context, similarity scores, and generate an answer for the given question."""
        # Retrieve the top-k most relevant documents with similarity scores
        docs_with_scores = self.vector_db.similarity_search_with_score(question, k=k)

        # Extract document content and similarity scores
        # context_window = ' '.join([doc[0].page_content for doc in docs_with_scores])
        
        similarity_scores = [doc[1] for doc in docs_with_scores]
        
        return min(similarity_scores)
               

    
    
def main():
    # Instantiate the class
    analyzer = RAG_pipeline()

    # Load the documents
    pdf_file_path = "./source/"
    pdf_files = pdf_file_path
    # pdf_files = [pdf_file_path+"reviewer_1.pdf", pdf_file_path+"reviewer_2.pdf"]  # Add other PDFs if needed
    docs = analyzer.load_documents(pdf_files)

    # Split documents into smaller chunks
    splits = analyzer.split_documents(docs, chunk_size=2048)

    # Create vector database
    vectordb = analyzer.create_vector_db(splits, persist_directory='docs/chroma/')
    k = vectordb._collection.count()

    # Build the QA chain
    analyzer.build_qa_chain()

    # Ask user for the question
    user_question = input("Please enter your question: ")

    # Generate the answer for the user-provided question
    answer, similarity_score = analyzer.generate_answer(user_question, k=k)

    # Print the result
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()