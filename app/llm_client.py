# app/llm_client.py
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def call_llm(prompt: str, system_prompt: str = None) -> str:
    contents = prompt
    if system_prompt:
        contents = f"{system_prompt}\n\nUser: {prompt}"
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents,
    )
    return response.text