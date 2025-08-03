# streamlit_app.py

import streamlit as st
import os
import json
import tempfile 
from dotenv import load_dotenv
import shutil # Still imported for Clear Data & Restart button
import chromadb 

# --- CRITICAL FIX FOR SQLITE3 COMPATIBILITY (MUST BE AT THE ABSOLUTE TOP) ---
# This block ensures pysqlite3 is loaded and replaces standard sqlite3
# BEFORE any other library (like chromadb) tries to import sqlite3.
# The 'import pysqlite3' directly is more robust than __import__
__import__('pysqlite3')
import sys # <--- THIS IS THE MISSING IMPORT!
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END CRITICAL FIX ---


# LangChain and Pydantic Imports (these should now come AFTER the above fix)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma 
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Union
from datetime import date, timedelta

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from query_parser import parse_user_query 

# --- Load Environment Variables ---
load_dotenv() 

# --- Initialize Streamlit session state variables at the top ---
if 'processed' not in st.session_state:
    st.session_state['processed'] = False
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None
if 'llm' not in st.session_state:
    st.session_state['llm'] = None
if 'uploaded_files_hash' not in st.session_state:
    st.session_state['uploaded_files_hash'] = None
if 'uploaded_raw_docs' not in st.session_state:
    st.session_state['uploaded_raw_docs'] = None

# --- Configuration Constants ---
# CHROMA_DB_DIR is no longer used for persistence by ChromaDB itself in this mode
CHROMA_DB_DIR = "chroma_db_streamlit_unused" 
GEMINI_LLM_MODEL = "models/gemini-2.5-pro"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# --- Pydantic Models (single, consistent schema for LLM output) ---
class SupportingClause(BaseModel):
    clause_text: str = Field(..., description="The exact text of the clause from the document.")
    document_id: str = Field(..., description="Identifier for the source document (e.g., policy_123.pdf).")
    page_number: Optional[int] = Field(None, description="Page number where the clause was found, if applicable.")

class PolicyDecision(BaseModel):
    Decision: str = Field(..., description="The determined decision: 'Approved', 'Rejected', 'Needs Further Review', 'Information Provided', 'Clarification Needed'.")
    Amount: Optional[float] = Field(None, description="The determined payout amount, if applicable, otherwise null.")
    Justification: str = Field(..., description="A clear explanation of the decision based on the retrieved clauses. MUST reference 'CHUNK_X' identifiers.")
    SupportingClauses: Optional[List[SupportingClause]] = Field(None, description="An array of specific clauses that led to the decision.")


# --- Helper Function to Load Documents from Uploaded Files ---
def load_documents_from_upload(uploaded_files):
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            print(f"Saving and loading: {uploaded_file.name}")

            docs = []
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif uploaded_file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
            elif uploaded_file.name.endswith(".txt"):
                content = uploaded_file.getvalue().decode("utf-8")
                docs = [Document(page_content=content, metadata={"source": uploaded_file.name})]
            else:
                st.warning(f"Skipping unsupported file: {uploaded_file.name}")
                continue

            for doc in docs:
                doc.metadata["document_id"] = uploaded_file.name
                if "page" in doc.metadata:
                    doc.metadata["page_number"] = doc.metadata["page"] + 1
                else:
                    doc.metadata["page_number"] = None
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = uploaded_file.name
                doc.metadata["doc_type"] = "Policy Document" 
                doc.metadata["effective_date"] = "2023-01-01" 
                doc.metadata["version"] = "1.0"
            documents.extend(docs)
    return documents

# --- Cached Function to Create Vector Store and Retriever (MODIFIED FOR PURE IN-MEMORY) ---
@st.cache_resource(
    hash_funcs={Document: lambda doc: (doc.page_content, tuple(sorted(doc.metadata.items())))}
)
def create_vector_store_and_retriever(): 
    # Get all documents that have been uploaded so far in this session
    # This list will accumulate across "Process Documents" clicks if not explicitly cleared
    raw_documents_list = st.session_state.uploaded_raw_docs 

    st.info(f"Processing {len(raw_documents_list)} documents for ingestion and creating knowledge base. This may take a while...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "?", "!", "\t", ",", ";", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_documents_list)
    st.write(f"Split documents into {len(chunks)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    llm = ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL, temperature=0.2)

    # --- Initialize a PURE IN-MEMORY ChromaDB client ---
    try:
        client = chromadb.Client() 
        collection_name = "policy_docs_in_memory_session" 
        
        # Always delete the collection for a fresh in-memory state for this processing run.
        # This prevents issues if the collection from a *previous click* is still hanging around in memory.
        try:
            client.delete_collection(name=collection_name)
            st.info(f"Deleted previous in-memory ChromaDB collection '{collection_name}'.")
        except: 
            pass 

        collection = client.create_collection(name=collection_name, embedding_function=embeddings)
        
        ids = [f"doc_{i}" for i in range(len(chunks))]
        documents_content = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        collection.add(
            ids=ids,
            documents=documents_content,
            metadatas=metadatas
        )
        st.write(f"ChromaDB in-memory collection '{collection_name}' populated.")
        
        vectorstore = Chroma(client=client, collection_name=collection_name, embedding_function=embeddings)
        st.success("ChromaDB initialized in-memory (no disk issues!). Data will not persist across app restarts.")

    except Exception as e:
        st.error(f"FATAL: Could not initialize even in-memory ChromaDB: {e}. Please report this issue.")
        st.stop()


    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 7}) 
    
    all_stored_data = vectorstore.get(include=['documents', 'metadatas'])
    bm25_documents = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(all_stored_data['documents'], all_stored_data['metadatas'])
    ]
    
    bm25_retriever = BM25Retriever.from_documents(bm25_documents)
    bm25_retriever.k = 7 

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever], 
        weights=[0.5, 0.5], 
        c = 100 
    )
    st.write("Hybrid Retriever initialized.")
    return llm, retriever

# --- Core Policy Decision Logic (from app.py) ---
def get_policy_decision(user_query: str, llm_instance, retriever_instance) -> PolicyDecision | None:
    print(f"\nProcessing query: '{user_query}'") 

    parsed_details = parse_user_query(user_query) 
    if not parsed_details:
        st.error("Could not parse query details. Please refine your query.")
        return None
    
    current_date_for_policy_inference = date(2025, 8, 2) 
    if parsed_details.policy_duration_months is not None and parsed_details.policy_start_date is None:
        start_date = current_date_for_policy_inference - timedelta(days=parsed_details.policy_duration_months * 30)
        parsed_details.policy_start_date = start_date.strftime("%Y-%m-%d")

    retrieval_query = f"{user_query}. " + \
                      f"Details: Age={parsed_details.age}, Gender={parsed_details.gender}, " + \
                      f"Procedure={parsed_details.procedure}, Location={parsed_details.location}, " + \
                      f"Policy Duration={parsed_details.policy_duration_months} months, " + \
                      f"Policy Start Date={parsed_details.policy_start_date}."

    relevant_chunks = retriever_instance.invoke(retrieval_query)

    if not relevant_chunks:
        st.warning("No highly relevant clauses found for the query.")
        return PolicyDecision(
            Decision="Needs Further Review",
            Amount=None,
            Justification="No relevant policy clauses could be retrieved. The query might be too vague or outside the scope of uploaded documents. Please try rephrasing or providing more details.",
            SupportingClauses=[] 
        )
    
    context_for_llm = []
    supporting_clauses_for_output = [] 
    for i, doc in enumerate(relevant_chunks):
        chunk_id = f"CHUNK_{i+1}"
        doc_id = doc.metadata.get('document_id', 'Unknown Document')
        page_num = doc.metadata.get('page_number', 'N/A')
        
        cleaned_clause_text = doc.page_content.replace('\n', ' ').strip() 
        cleaned_clause_text = ' '.join(cleaned_clause_text.split()) 
        
        context_for_llm.append(
            f"### {chunk_id} (Source: {doc_id}, Page: {page_num})\n"
            f"{cleaned_clause_text}\n" 
        )
        
        supporting_clauses_for_output.append(
            SupportingClause(
                clause_text=cleaned_clause_text, 
                document_id=doc_id,
                page_number=page_num
            )
        )
    
    full_context_str = "\n\n".join(context_for_llm)

    prompt_template = PromptTemplate(
        template="""You are an expert insurance policy evaluator.
        Your task is to analyze the user's query and the provided relevant policy clauses (context) to determine a decision and justification.
        You MUST return ONLY a JSON object that strictly adheres to the following `PolicyDecision` schema.

        **Instructions for Populating `PolicyDecision` Schema Fields:**

        1.  **For 'Decision':**
            * 'Approved': Only if all conditions for coverage for a *claim scenario* are explicitly met AND no exclusions apply, based *solely* on the provided context.
            * 'Rejected': Only if an explicit exclusion applies OR a required condition is clearly not met for a *claim scenario*, based *solely* on the provided context.
            * 'Information Provided': If the query is purely informational (e.g., asking about a waiting period, general terms, definitions, process overviews, benefits) AND you find relevant information in the context to answer it.
            * 'Clarification Needed': If the query is ambiguous, incomplete, or requires more specific details from the user to provide a definitive answer (e.g., missing specific type of treatment, unclear dates, or needing to know if an optional plan is selected).
            * 'Needs Further Review': If the provided context is insufficient or irrelevant to address the query at all, or if you cannot make a definitive decision/provide information based on the rules.

        2.  **For 'Amount':**
            * If the policy clauses explicitly state a precise numerical payout amount for the specific procedure/query (e.g., "up to INR 3,750,000", "USD 100,000"), extract this numerical value ONLY as a float. You MUST convert any currency (INR, USD) to a simple float number. For example, "INR 3,750,000" should be `3750000.0`. "USD 100,000" should be `100000.0`.
            * If a method to calculate the amount is provided, perform the calculation if possible.
            * Otherwise, you MUST set 'Amount' to `null`. DO NOT fabricate amounts or include currency symbols in the number.

        3.  **For 'Justification':**
            * Provide a clear, concise explanation of your decision/information.
            * You MUST explicitly reference the 'CHUNK_X' identifiers from the 'Relevant Policy Clauses (Context)' that directly support your justification. For example: "The claim is rejected because the policy (CHUNK_1) states that procedures performed within the initial 6 months are not covered."
            * If 'Decision' is 'Needs Further Review' or 'Clarification Needed', explain why the context is insufficient or what information is missing.

        **General Rule:** DO NOT include any other text, preambles, or explanations outside the JSON block.
        `SupportingClauses` field will be populated programmatically by the system. You do NOT need to fill this in the JSON output.

        ---

        **PolicyDecision Schema (Use double curly braces to escape literal braces):**
        ```json
        {{
          "Decision": "string (Approved|Rejected|Needs Further Review|Information Provided|Clarification Needed)",
          "Amount": "float | null",
          "Justification": "string (Explanation, reference CHUNK_X IDs)"
        }}
        ```

        ---

        **User Query:** {user_query}

        **Parsed Query Details:**
        {parsed_details_json}

        **Relevant Policy Clauses (Context):**
        {context}

        **Your JSON Response (ONLY the JSON, adhering to the PolicyDecision schema):**
        """,
        input_variables=["user_query", "parsed_details_json", "context"]
    )

    chain_for_decision = prompt_template | llm_instance | StrOutputParser()

    try:
        parsed_details_json_str = parsed_details.model_dump_json(indent=2)
        
        llm_raw_response = chain_for_decision.invoke({
            "user_query": user_query,
            "parsed_details_json": parsed_details_json_str,
            "context": full_context_str
        })

        json_string_match = llm_raw_response.strip().removeprefix("```json").removesuffix("```")
        
        llm_structured_output = json.loads(json_string_match)
        
        final_decision_obj = PolicyDecision(**llm_structured_output) 
        
        text_to_check_for_citations = final_decision_obj.Justification
        
        cited_clauses = []
        if text_to_check_for_citations:
            for i, chunk in enumerate(relevant_chunks):
                chunk_id = f"CHUNK_{i+1}"
                if chunk_id.lower() in text_to_check_for_citations.lower(): 
                    cited_clauses.append(
                        SupportingClause(
                            clause_text=supporting_clauses_for_output[i].clause_text, 
                            document_id=supporting_clauses_for_output[i].document_id,
                            page_number=supporting_clauses_for_output[i].page_number
                        )
                    )
        
        if not cited_clauses and final_decision_obj.Decision not in ["Needs Further Review", "Clarification Needed", "Information Provided"]:
             st.warning("Warning: LLM did not explicitly cite chunks. Including all retrieved chunks as general supporting context.")
             final_decision_obj.SupportingClauses = supporting_clauses_for_output
        else:
            final_decision_obj.SupportingClauses = cited_clauses


        return final_decision_obj

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse LLM's JSON response: {e}")
        st.code(f"Raw LLM response:\n{llm_raw_response}", language="json")
        return None
    except ValidationError as e:
        st.error(f"Validation Error with LLM's structured output: {e}")
        st.code(f"Raw LLM response:\n{llm_raw_response}", language="json")
        st.json(e.errors()) # Show Pydantic's detailed validation errors
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during decision generation: {e}")
        st.write("Please ensure your GOOGLE_API_KEY is set and valid.")
        return None

# --- Streamlit UI Layout ---
st.set_page_config(page_title="PolicyWise AI Assistant", layout="wide")

st.title("📄 PolicyWise AI Assistant")
st.markdown("Upload your policy documents (PDF, DOCX, TXT), process them, and then ask questions!")

# --- Document Upload Section ---
st.header("1. Upload Documents")
uploaded_files = st.file_uploader(
    "Upload your policy documents", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    files_hash = hash(tuple((f.name, f.size) for f in uploaded_files))
    
    if st.button("Process Documents", key="process_docs_button"):
        # --- Delete previous ChromaDB data (no longer critical for in-memory) ---
        # This will still clear the directory if it was created by a previous attempt,
        # but the ChromaDB itself is now in-memory.
        if os.path.exists(CHROMA_DB_DIR):
            try:
                shutil.rmtree(CHROMA_DB_DIR)
                st.info(f"Cleared previous data in {CHROMA_DB_DIR} for fresh processing.")
            except OSError as e:
                st.error(f"Error removing ChromaDB directory: {e}. Please manually delete '{CHROMA_DB_DIR}' if this persists.")
                st.stop() # Stop execution if unable to clear old data

        # --- CRUCIAL FIX: Explicitly clear the cache for this specific function ---
        st.cache_resource.clear() 

        st.session_state['processed'] = False 
        st.session_state['retriever'] = None
        st.session_state['llm'] = None
        st.session_state['uploaded_files_hash'] = files_hash 
        st.session_state['uploaded_raw_docs'] = load_documents_from_upload(uploaded_files) 

        if st.session_state['uploaded_raw_docs']:
            try:
                llm_instance, retriever_instance = create_vector_store_and_retriever() 
                st.session_state['llm'] = llm_instance
                st.session_state['retriever'] = retriever_instance
                st.session_state['processed'] = True
                st.success(f"Successfully processed {len(st.session_state['uploaded_raw_docs'])} documents and built knowledge base!")
            except Exception as e:
                st.error(f"Error during document processing: {e}")
                st.warning("Please ensure your Google API Key is correctly set in your .env file.")
        else:
            st.warning("No documents were loaded from the uploaded files.")
elif 'processed' in st.session_state and st.session_state['processed']:
    current_files_hash = hash(tuple((f.name, f.size) for f in uploaded_files)) if uploaded_files else None
    if 'uploaded_files_hash' in st.session_state and st.session_state['uploaded_files_hash'] != current_files_hash:
        st.warning("Uploaded files have changed. Click 'Process Documents' to update the knowledge base.")
    else:
        st.info("Documents are already processed.")


# --- Query Section ---
st.header("2. Ask a Question")
if 'processed' in st.session_state and st.session_state['processed'] and st.session_state['retriever'] is not None:
    user_query = st.text_area("Enter your query about the policy documents:", height=100, key="user_query_input")

    if st.button("Get Policy Decision", key="get_decision_button"):
        if user_query:
            with st.spinner("Getting decision..."):
                llm_instance = st.session_state['llm']
                retriever_instance = st.session_state['retriever']
                
                decision = get_policy_decision(user_query, llm_instance, retriever_instance)
                
                if decision:
                    st.subheader("Decision Summary:")
                    if decision.Decision in ["Information Provided", "Clarification Needed"]:
                        st.info(f"**Output:** {decision.Decision}")
                        st.write(f"**Details:** {decision.Justification}") 
                    else: 
                        st.success(f"**Decision:** {decision.Decision}")
                        if decision.Amount is not None:
                            st.write(f"**Amount:** {decision.Amount:,.2f}") 
                        st.write(f"**Justification:** {decision.Justification}") 
                    
                    with st.expander("View Complete JSON Response"):
                        st.json(decision.model_dump_json(indent=2))
                    
                    st.subheader("Supporting Clauses:")
                    if decision.SupportingClauses: 
                        for i, clause in enumerate(decision.SupportingClauses):
                            st.markdown(f"**Clause {i+1}** (Source: `{clause.document_id}`, Page: `{clause.page_number if clause.page_number else 'N/A'}`):")
                            st.text(clause.clause_text) # Reverted to st.text for simple display
                            st.markdown("---")
                    else:
                        st.info("No specific supporting clauses cited by the LLM for this response, or the information was synthesized.")
                else:
                    st.error("Failed to get a decision. Please check the console/logs for errors displayed above.")
        else:
            st.warning("Please enter a query.")
else:
    st.info("Please upload and process documents first to enable the query section.")

# --- Clean up cached resources on app rerun or close ---
if st.button("Clear Processed Data & Restart", key="clear_data_button"):
    if os.path.exists(CHROMA_DB_DIR):
        try:
            shutil.rmtree(CHROMA_DB_DIR)
            st.success("Cleaned up processed data and restarted session.")
        except OSError as e:
            st.error(f"Error removing ChromaDB directory: {e}. Please manually delete '{CHROMA_DB_DIR}' if this persists.")
    st.session_state.clear()
    st.rerun() # Rerun the app from top to reset UI
