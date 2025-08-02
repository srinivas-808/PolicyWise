# ingest_documents.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma # Updated import for Chroma
from langchain_core.documents import Document # Explicit import for Document creation for TXT files

load_dotenv() # Load environment variables from .env

# --- Configuration ---
DATA_DIR = "data" # Directory where your policy documents (PDF, DOCX, TXT) are stored
CHROMA_DB_DIR = "chroma_db" # Directory to store the ChromaDB vector store

# Optimal chunking parameters:
# Adjust these based on the structure of your policy documents.
# Goal: Each chunk should contain a complete, logical piece of information (e.g., a full clause or sub-section).
CHUNK_SIZE = 1500 
CHUNK_OVERLAP = 150

# --- Gemini Embedding model for semantic search ---
EMBEDDING_MODEL_NAME = "models/text-embedding-004" # Recommended, newer embedding model

def load_documents(data_dir):
    """Loads documents from the specified directory and extracts basic metadata."""
    documents = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue

        docs = [] # Temporary list for documents loaded from the current file
        if filename.endswith(".pdf"):
            print(f"Loading PDF: {filename}")
            loader = PyPDFLoader(filepath)
            docs = loader.load()
        elif filename.endswith(".docx"):
            print(f"Loading DOCX: {filename}")
            loader = Docx2txtLoader(filepath)
            docs = loader.load()
        elif filename.endswith(".txt"):
            print(f"Loading TXT: {filename}")
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                # Explicitly create a LangChain Document object for TXT
                docs = [Document(page_content=text, metadata={"source": filename})]
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        # Add essential metadata to each document loaded from the current file
        for doc in docs:
            doc.metadata["document_id"] = filename 
            if "page" in doc.metadata:
                doc.metadata["page_number"] = doc.metadata["page"] + 1 
            else:
                doc.metadata["page_number"] = None 
            
            if 'source' not in doc.metadata:
                doc.metadata['source'] = filename

            doc.metadata["doc_type"] = "Policy Document" 
            doc.metadata["effective_date"] = "2023-01-01" # Placeholder for actual parsing
            doc.metadata["version"] = "1.0" # Placeholder
            
        documents.extend(docs) # Extend the main 'documents' list with the 'docs' list (containing Document objects)
    return documents

def split_documents(documents, chunk_size, chunk_overlap):
    """Splits loaded documents into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", "\t", ",", ";", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks, persist_directory):
    """Creates a ChromaDB vector store from document chunks and persists it."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (Gemini)...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")

    print(f"Creating Chroma DB at {persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Chroma DB created and persisted.")
    return vectorstore

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Error: '{DATA_DIR}' directory not found. Please create it and place your documents inside.")
    else:
        all_documents = load_documents(DATA_DIR)
        if all_documents:
            all_chunks = split_documents(all_documents, CHUNK_SIZE, CHUNK_OVERLAP)
            vector_db = create_vector_store(all_chunks, CHROMA_DB_DIR)
            print("Document ingestion complete!")
        else:
            print("No documents found to ingest.")