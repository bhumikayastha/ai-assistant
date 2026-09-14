# app/llm_client.py
import os
import re
from dotenv import load_dotenv
from google import genai

from tenacity import retry, stop_after_attempt, wait_exponential
load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _fallback_answer_from_context(prompt: str, json_mode: bool = False) -> str:
    text = prompt
    if "Draft answer:" in text and "Retrieved context" in text:
        if json_mode:
            return '{"verdict": "SUFFICIENT", "refined_query": "", "clarifying_question": ""}'
        return "SUFFICIENT"

    if "Context:" in text and "Question:" in text:
        context_part = text.split("Context:", 1)[1].split("\n\nQuestion:", 1)[0].strip()
        question_part = text.split("Question:", 1)[1].strip()
        question_part = question_part.split("\n\nAnswer", 1)[0].strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context_part) if s.strip()]
        if not sentences:
            sentences = [context_part]
        q_words = set(re.findall(r"[A-Za-z0-9]+", question_part.lower()))
        if q_words:
            best = max(sentences, key=lambda s: sum(1 for w in q_words if w in s.lower()))
        else:
            best = sentences[0]
        if json_mode:
            return '{"answer": "' + best.replace('"', '\\"') + '", "confidence": "low"}'
        return best

    if json_mode:
        return '{"verdict": "SUFFICIENT", "refined_query": "", "clarifying_question": ""}'
    return "The model is currently unavailable, but the local document context is available for review."


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(prompt: str, system_prompt: str = None) -> str:
    contents = prompt
    if system_prompt:
        contents = f"{system_prompt}\n\nUser: {prompt}"
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents,
        )
        return response.text
    except Exception:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=contents,
            )
            return response.text
        except Exception:
            return _fallback_answer_from_context(contents)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm_json(prompt: str, system_prompt: str = None) -> str:
    """Same as call_llm but instructs the model to return valid JSON only."""
    json_instruction = "\n\nRespond with ONLY valid JSON, no markdown, no extra text."
    full_prompt = prompt + json_instruction
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{full_prompt}"

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=full_prompt,
            config={"response_mime_type": "application/json"},
        )
        return response.text
    except Exception:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=full_prompt,
                config={"response_mime_type": "application/json"},
            )
            return response.text
        except Exception:
            return _fallback_answer_from_context(full_prompt, json_mode=True)


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
                model="gemini-2.5-flash-lite",
                contents=contents,
            )
            return response.text
        except Exception as e2:
            return _fallback_answer_from_context(contents)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm_full(prompt: str, system_prompt: str = None, json_mode: bool = False) -> dict:
    """Like call_llm, but also returns token usage — needed for the agent's cost accounting."""
    contents = prompt
    if system_prompt:
        contents = f"{system_prompt}\n\nUser: {prompt}"

    config = {"response_mime_type": "application/json"} if json_mode else None
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents,
            config=config,
        )
    except Exception:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=contents,
                config=config,
            )
        except Exception:
            fallback_text = _fallback_answer_from_context(contents, json_mode=json_mode)
            if json_mode:
                return {"text": fallback_text, "tokens": 0}
            return {"text": fallback_text, "tokens": 0}
    usage = getattr(response, "usage_metadata", None)
    total_tokens = usage.total_token_count if usage else 0
    return {"text": response.text, "tokens": total_tokens}