# app.py
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma # Updated import for Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser # Correct import for StrOutputParser
from pydantic import BaseModel, Field, ValidationError # Direct Pydantic V2 import
from typing import List, Optional, Union # Added Union for dynamic types
from datetime import date, timedelta

# --- UPDATED IMPORTS for Hybrid Search ---
from langchain_community.retrievers import BM25Retriever # Correct import for BM25
from langchain.retrievers import EnsembleRetriever # Correct import for EnsembleRetriever
from langchain_core.documents import Document # Explicit import for Document objects

from query_parser import parse_user_query, QueryDetails 

load_dotenv() # Load environment variables from .env

# --- Configuration ---
CHROMA_DB_DIR = "chroma_db" 

# --- Gemini LLM Configuration ---
GEMINI_LLM_MODEL = "models/gemini-2.5-pro" # Recommended for strong reasoning and structured output

# --- Gemini Embeddings Configuration (MUST match ingest_documents.py) ---
EMBEDDING_MODEL_NAME = "models/text-embedding-004" # Recommended for higher quality embeddings

# --- Pydantic model for structured output (SINGLE, CONSISTENT SCHEMA) ---
class SupportingClause(BaseModel):
    clause_text: str = Field(..., description="The exact text of the clause from the document.")
    document_id: str = Field(..., description="Identifier for the source document (e.g., policy_123.pdf).")
    page_number: Optional[int] = Field(None, description="Page number where the clause was found, if applicable.")

# PolicyDecision schema with SupportingClauses marked as Optional
class PolicyDecision(BaseModel):
    Decision: str = Field(..., description="The determined decision: 'Approved', 'Rejected', 'Needs Further Review', 'Information Provided', 'Clarification Needed'.")
    Amount: Optional[float] = Field(None, description="The determined payout amount, if applicable, otherwise null.")
    Justification: str = Field(..., description="A clear explanation of the decision based on the retrieved clauses. MUST reference 'CHUNK_X' identifiers.")
    # --- CRUCIAL FIX HERE: SupportingClauses is now Optional ---
    SupportingClauses: Optional[List[SupportingClause]] = Field(None, description="An array of specific clauses that led to the decision.")


# --- Initialize Components ---
def initialize_components():
    """Initializes the LLM, loads the ChromaDB, and sets up a Hybrid Retriever."""
    print("Initializing Gemini LLM and Embeddings...")
    
    llm = ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL, temperature=0.2) 
    
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    print(f"Loading Chroma DB from {CHROMA_DB_DIR}...")
    try:
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        
        # --- Configure Hybrid Retriever ---
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
            weights=[0.5, 0.5], # 50% keyword, 50% semantic
            c = 100 # Maximum number of unique documents to return (after deduplication)
        )

        print("Components initialized successfully.")
        return llm, retriever
    except Exception as e:
        print(f"Error loading Chroma DB: {e}")
        print("Please ensure 'rank_bm25' is installed (`pip install rank_bm25`),")
        print("you have run 'python ingest_documents.py' first, and your GOOGLE_API_KEY is set and valid.")
        exit(1)

# --- Decision-making and Justification Function ---
def get_policy_decision(user_query: str, llm, retriever) -> PolicyDecision | None:
    """
    Processes a user query, retrieves relevant policy clauses, and generates a structured decision.
    Returns a PolicyDecision object or None on error.
    """
    print(f"\nProcessing query: '{user_query}'")

    parsed_details = parse_user_query(user_query)
    if not parsed_details:
        print("Could not parse query details. Please refine your query.")
        return None
    
    print(f"Parsed Details: {parsed_details.model_dump_json(indent=2)}") 

    current_date_for_policy_inference = date(2025, 8, 2) 
    if parsed_details.policy_duration_months is not None and parsed_details.policy_start_date is None:
        start_date = current_date_for_policy_inference - timedelta(days=parsed_details.policy_duration_months * 30)
        parsed_details.policy_start_date = start_date.strftime("%Y-%m-%d")
        print(f"Inferred policy start date: {parsed_details.policy_start_date}")

    retrieval_query = f"{user_query}. " + \
                      f"Details: Age={parsed_details.age}, Gender={parsed_details.gender}, " + \
                      f"Procedure={parsed_details.procedure}, Location={parsed_details.location}, " + \
                      f"Policy Duration={parsed_details.policy_duration_months} months, " + \
                      f"Policy Start Date={parsed_details.policy_start_date}."

    relevant_chunks = retriever.invoke(retrieval_query)

    if not relevant_chunks:
        print("No relevant clauses found for the query.")
        return PolicyDecision(
            Decision="Needs Further Review", # Default decision if no chunks found
            Amount=None,
            Justification="No relevant policy clauses could be retrieved to make a decision. The query might be too vague or outside the scope of available documents. Please try rephrasing or providing more details.",
            SupportingClauses=[] # Still provide an empty list for consistency
        )

    print(f"Retrieved {len(relevant_chunks)} relevant chunks.")
    
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

    chain_for_decision = prompt_template | llm | StrOutputParser()

    try:
        parsed_details_json_str = parsed_details.model_dump_json(indent=2)
        
        llm_raw_response = chain_for_decision.invoke({
            "user_query": user_query,
            "parsed_details_json": parsed_details_json_str,
            "context": full_context_str
        })

        json_string_match = llm_raw_response.strip().removeprefix("```json").removesuffix("```")
        
        llm_structured_output = json.loads(json_string_match)
        
        # This will now succeed even if 'SupportingClauses' is missing in llm_structured_output
        final_decision_obj = PolicyDecision(**llm_structured_output) 
        
        # Populate SupportingClauses based on citations found in Justification
        text_to_check_for_citations = final_decision_obj.Justification
        
        cited_clauses = []
        if text_to_check_for_citations:
            for i, chunk in enumerate(relevant_chunks):
                chunk_id = f"CHUNK_{i+1}"
                if chunk_id.lower() in text_to_check_for_citations.lower(): 
                    cited_clauses.append(
                        SupportingClause(
                            clause_text=supporting_clauses_for_output[i].clause_text, # Use the already cleaned text
                            document_id=supporting_clauses_for_output[i].document_id,
                            page_number=supporting_clauses_for_output[i].page_number
                        )
                    )
        
        # Fallback: if no clauses were explicitly referenced, but a decision was made (not error/needs review)
        if not cited_clauses and final_decision_obj.Decision not in ["Needs Further Review", "Clarification Needed"]:
             print("Warning: LLM did not explicitly cite chunks. Including all retrieved chunks as general supporting context.")
             final_decision_obj.SupportingClauses = supporting_clauses_for_output
        else:
            final_decision_obj.SupportingClauses = cited_clauses # Assign the populated list


        return final_decision_obj

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM's JSON response: {e}")
        print(f"Raw LLM response: \n---\n{llm_raw_response}\n---")
        return None
    except ValidationError as e:
        print(f"Validation Error with LLM's structured output: {e}")
        print(f"Raw LLM response: \n---\n{llm_raw_response}\n---")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during decision generation: {e}")
        print(f"Please ensure your GOOGLE_API_KEY is set and valid, and the '{GEMINI_LLM_MODEL}' model is accessible.")
        return None

if __name__ == "__main__":
    llm_instance, retriever_instance = initialize_components()

    queries_to_test = [
        "46-year-old male, knee surgery in Pune, 3-month-old insurance policy", # Claim-like
        "What is the waiting period for critical illness coverage for a 30 year old female?", # Informational
        "Can I get dental treatment reimbursed if my policy started 1 year ago?", # Claim-like / Clarification
        "What is the maximum coverage for an emergency hospitalization in Mumbai for a 55-year-old?", # Informational / Amount
        "Is my 2-month-old policy valid for a pre-existing condition treatment?", # Claim-like
        "What if I need treatment in a location not covered by the policy?", # Informational
        "Tell me about the policy's terms regarding non-network hospitals.", # Informational / Process
        "I need to claim for a fractured arm from an accident. My policy started 2 months ago. What's the process?", # Process
        "What is the definition of 'Medical Expenses' in the policy?", # Informational
        "My policy document is unclear on exclusions for cosmetic surgery.", # Informational
        "What are the steps to renew my policy?", # Process (New Example)
        "What are the benefits of the Imperial Plus Plan?", # Informational (New Example)
    ]

    for q in queries_to_test:
        decision = get_policy_decision(q, llm_instance, retriever_instance)
        if decision:
            print("\n--- Final Decision ---")
            # --- DYNAMIC DISPLAY LOGIC IS HERE ---
            if decision.Decision in ["Information Provided", "Clarification Needed"]:
                print(f"Output: {decision.Decision}")
                print(f"Details: {decision.Justification}")
            else: # Approved, Rejected, Needs Further Review (for claims)
                print(f"Decision: {decision.Decision}")
                # Only print amount if it's not null and decision is claim-related
                if decision.Amount is not None:
                    print(f"Amount: {decision.Amount}")
                print(f"Justification: {decision.Justification}")
            
            # Always print Supporting Clauses, as requested
            print("\n--- Supporting Clauses ---")
            # Ensure decision.SupportingClauses is not None before iterating
            if decision.SupportingClauses:
                for i, clause in enumerate(decision.SupportingClauses):
                    print(f"Clause {i+1} (Source: {clause.document_id}, Page: {clause.page_number}):")
                    print(clause.clause_text)
                    print("-" * 10) # Separator for clauses
            else:
                print("No specific supporting clauses cited by the LLM for this response.")
            print("-" * 30)
        else:
            print(f"Failed to get a decision for query: {q}")
            print("-" * 30)