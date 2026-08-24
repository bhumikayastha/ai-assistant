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

def call_llm_json(prompt: str, system_prompt: str = None) -> str:
    """Same as call_llm but instructs the model to return valid JSON only."""
    json_instruction = "\n\nRespond with ONLY valid JSON, no markdown, no extra text."
    full_prompt = prompt + json_instruction
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{full_prompt}"

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=full_prompt,
        config={"response_mime_type": "application/json"},
    )
    return response.text    