import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv() # Load your GOOGLE_API_KEY from .env

# Configure with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available Gemini models for your account:")
for m in genai.list_models():
    # Only print models capable of generating content (text generation)
    if 'generateContent' in m.supported_generation_methods:
        print(f"  Name: {m.name}")
        print(f"  Display Name: {m.display_name}")
        print(f"  Description: {m.description}")
        print(f"  Input Token Limit: {m.input_token_limit}")
        print(f"  Output Token Limit: {m.output_token_limit}")
        print(f"  Supported Methods: {m.supported_generation_methods}")
        print("-" * 30)

print("\nAvailable Embedding models for your account:")
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(f"  Name: {m.name}")
        print(f"  Display Name: {m.display_name}")
        print(f"  Description: {m.description}")
        print(f"  Supported Methods: {m.supported_generation_methods}")
        print("-" * 30)