# query_parser.py
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, ValidationError # Direct Pydantic V2 import
import os
from datetime import date

load_dotenv() # Load environment variables from .env

# --- Gemini Configuration for LLM Inference ---
GEMINI_LLM_MODEL = "models/gemini-2.5-pro" # Recommended for robust parsing

# Define the Pydantic model for structured query details
class QueryDetails(BaseModel):
    age: int | None = Field(None, description="Age of the person, if mentioned in years.")
    gender: str | None = Field(None, description="Gender of the person (e.g., 'male', 'female', 'other'), if mentioned.")
    procedure: str | None = Field(None, description="Medical procedure, type of claim, or event (e.g., 'knee surgery', 'dental treatment', 'emergency hospitalization').")
    location: str | None = Field(None, description="Geographic location related to the procedure or claim (e.g., 'Pune', 'Mumbai').")
    policy_duration_months: int | None = Field(None, description="Duration of the insurance policy in months, if explicitly mentioned (e.g., '3-month-old policy' -> 3).")
    policy_start_date: str | None = Field(None, description="Infer the policy start date in YYYY-MM-DD format based on policy_duration_months and the provided current date. If policy duration is not clear, set to null.")

def parse_user_query(user_query: str) -> QueryDetails | None:
    """
    Parses a natural language user query to extract structured details using an LLM.
    """
    llm = ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL, temperature=0.0) # Low temperature for deterministic parsing

    parser = PydanticOutputParser(pydantic_object=QueryDetails)

    prompt = PromptTemplate(
        template="You are an expert at extracting key details from natural language insurance queries.\n"
                 "Extract the following information from the user's query and return a JSON object.\n"
                 "If a detail is not explicitly mentioned or cannot be reliably inferred, return null for that field.\n"
                 "{format_instructions}\n\n"
                 "User Query: {query}\n"
                 "Current Date: {current_date}\n" # Explicitly provide current date
                 "Extracted Details:",
        input_variables=["query", "current_date"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    # --- UPDATED: Hardcoded current date to match latest context (August 2, 2025) ---
    current_date_str = date(2025, 8, 2).strftime("%Y-%m-%d") # August 2, 2025

    try:
        parsed_details = chain.invoke({"query": user_query, "current_date": current_date_str})
        return parsed_details
    except ValidationError as e:
        print(f"Validation Error in parsing query: {e}")
        print(f"Gemini model might not have returned valid JSON for this query. Raw error: {e.errors()}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during query parsing: {e}")
        print(f"Please ensure your GOOGLE_API_KEY is set and valid, and the '{GEMINI_LLM_MODEL}' model is accessible.")
        return None

if __name__ == "__main__":
    test_queries = [
        "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
        "My 50-year-old sister needs heart bypass in Mumbai, policy just started last month",
        "Is cataract surgery covered for a 70 year old?",
        "What about dental work?"
    ]

    for query in test_queries:
        print(f"\nOriginal Query: '{query}'")
        parsed_info = parse_user_query(query)
        if parsed_info:
            print(f"Parsed Information: {parsed_info.model_dump_json(indent=2)}")
        else:
            print("Failed to parse query.")