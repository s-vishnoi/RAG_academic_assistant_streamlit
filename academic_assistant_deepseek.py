from langchain_core.globals import set_verbose, set_debug  # Enable global debugging and verbosity to trace LangChain behavior.
from langchain_ollama import ChatOllama, OllamaEmbeddings  # Interfaces to connect to Ollama's local LLMs and their corresponding embedding models.
from langchain.schema.output_parser import StrOutputParser  # Converts LLM output to plain string for display.
from langchain_community.vectorstores import Chroma  # ChromaDB for storing and retrieving embedded text chunks.
from langchain_community.document_loaders import PyPDFLoader  # Parses PDF files into readable text for processing.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Breaks large texts into overlapping, coherent chunks.
from langchain.schema.runnable import RunnablePassthrough  # Allows data to be piped cleanly through chains.
from langchain_community.vectorstores.utils import filter_complex_metadata  # Filters extra metadata from loaded documents.
from langchain_core.prompts import ChatPromptTemplate  # Builds prompts dynamically using placeholders.
import logging
import os
import shutil

# Initialize logging and LangChain debug verbosity to monitor internal flow and debugging info.
set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatPDF:
    """
    Core class implementing a Retrieval-Augmented Generation (RAG) pipeline over PDF documents.
    It allows:
    - Ingesting and vectorizing documents
    - Querying using a local LLM with embedded context
    - Short-term memory of past Q&A for contextual coherence
    """

    def __init__(self, llm_model: str = "phi", embedding_model: str = "mxbai-embed-large"):
        """
        Initialize the RAG stack:
        - Connect to an Ollama LLM (phi3/llama3/mistral, etc.)
        - Use an Ollama-compatible embedding model to vectorize document chunks
        - Prepare a prompt template to control how context and questions are injected into the LLM
        - Set up state for memory and retrieval
        """
        self.model = ChatOllama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)  # Overlap ensures continuity between chunks

        # Prompt structure: important to control LLM behavior in RAG by delimiting context and question clearly
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an expert assistant answering questions on an uploaded document given the context.
            Context:
            {context}

            Question:
            {question}

            Answer concisely and accurately in three sentences or less.
            """
        )

        # Internal memory and retrieval objects
        self.vector_store = None  # Holds embedded chunks from PDF
        self.retriever = None     # Interfaces with vector DB to retrieve relevant context
        self.chat_history = []    # Tracks recent turns for memory injection into prompt

    def ingest(self, pdf_file_path: str):
        """
        Main ingestion step:
        1. Clears old vector DB to avoid mixing data.
        2. Extracts text using PyPDFLoader.
        3. Splits into chunks suitable for retrieval (short, overlapping).
        4. Embeds and stores them in Chroma.

        This step ensures high-recall retrieval, which is crucial in RAG to reduce hallucination.
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        db_dir = "chroma_db"

        # Clear previous document vectors to prevent context leakage
        if os.path.exists(db_dir):
            shutil.rmtree(db_dir)

        # Parse + chunk + clean
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        # Vector store built from document chunks — enables fast semantic search
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=db_dir,
        )
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Executes a RAG query:
        - Retrieves top-k most relevant chunks based on cosine similarity and a score threshold
        - Combines it with previous conversational context
        - Builds a formatted prompt
        - Sends it through the LLM pipeline

        This combines retrieval-based grounding with generative flexibility, a proven strategy to reduce hallucination.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            # Retriever acts as a search engine on the vector DB
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document."

        # Select top 3 matching segments — balance between brevity and informativeness
        context = " \n \n ".join(doc.page_content for doc in retrieved_docs[:3])

        # Retrieve prior Q&A pairs (memory window of 3 turns)
        chat_log = " \n \n ".join([
            f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in self.chat_history[-3:]
        ])

        # Build composite context block
        extended_context = f"{chat_log}\n\n{context}" if chat_log else context

        # LLM pipeline execution: prompt template → LLM → string output
        chain = self.prompt | self.model | StrOutputParser()
        try:
            logger.info("Invoking model...")
            response = chain.invoke({"context": extended_context, "question": query})
            logger.info("Model responded.")
        except Exception as e:
            logger.error(f"Model invocation failed: {e}")
            return f"Error during model response: {e}"

        # Save current exchange into memory to preserve context in future turns
        self.chat_history.append({"user": query, "assistant": response})

        return response

    def clear(self):
        """
        Full reset: removes all document data and prior context.
        Use this when switching documents or starting fresh.
        """
        self.vector_store = None
        self.retriever = None
        self.chat_history = []

    def reset_chat(self):
        """
        Reset only conversation memory, keeping document loaded.
        Useful for switching questions/topics while maintaining access to the same knowledge base.
        """
        self.chat_history = []
