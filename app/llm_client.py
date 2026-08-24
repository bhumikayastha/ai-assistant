# app/llm_client.py
import os
from dotenv import load_dotenv
from google import genai

from tenacity import retry, stop_after_attempt, wait_exponential
load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(prompt: str, system_prompt: str = None) -> str:
    contents = prompt
    if system_prompt:
        contents = f"{system_prompt}\n\nUser: {prompt}"
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents,
    )
    return response.text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
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

def call_llm_with_fallback(prompt: str, system_prompt: str = None) -> str:
    try:
        return call_llm(prompt, system_prompt)
    except Exception as e:
        print(f"Primary model failed after retries: {e}")
        try:
            contents = prompt
            if system_prompt:
                contents = f"{system_prompt}\n\nUser: {prompt}"
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",  # smaller fallback model
                contents=contents,
            )
            return response.text
        except Exception as e2:
            return f"Sorry, the assistant is temporarily unavailable. ({e2})"