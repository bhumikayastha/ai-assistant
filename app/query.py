# app/query.py
from app.rag import similarity_search
from app.llm_client import call_llm
import json
from pydantic import BaseModel

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY
the provided context below. If the answer isn't in the context, say you don't know.
Be concise."""

def rag_query(question: str) -> str:
    chunks = similarity_search(question, k=3)
    context = "\n\n".join(chunks)

    prompt = f"""Context:
{context}

Question: {question}

Answer using only the context above."""

    return call_llm(prompt, system_prompt=SYSTEM_PROMPT)


class RagAnswer(BaseModel):
    answer: str
    confidence: str  # e.g. "high", "medium", "low"

def rag_query_structured(question: str) -> RagAnswer:
    from app.llm_client import call_llm_json
    chunks = similarity_search(question, k=3)
    context = "\n\n".join(chunks)

    prompt = f"""Context:
{context}

Question: {question}

Return JSON with fields: "answer" (string) and "confidence" (one of "high", "medium", "low")."""

    raw = call_llm_json(prompt, system_prompt=SYSTEM_PROMPT)
    data = json.loads(raw)
    return RagAnswer(**data)


from google.genai import types
from app.tools import AVAILABLE_TOOLS, search_knowledge_base, get_word_count

def rag_query_with_tools(question: str) -> str:
    from app.llm_client import client

    tool_declarations = types.Tool(function_declarations=[
        {
            "name": "search_knowledge_base",
            "description": "Search the ingested documents for relevant information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "get_word_count",
            "description": "Count words in a piece of text",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    ])

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=question,
        config=types.GenerateContentConfig(tools=[tool_declarations]),
    )

    part = response.candidates[0].content.parts[0]

    if part.function_call:
        fn_name = part.function_call.name
        fn_args = dict(part.function_call.args)
        print(f"[Tool called: {fn_name}({fn_args})]")

        result = AVAILABLE_TOOLS[fn_name](**fn_args)

        followup = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=f"Question: {question}\nTool result: {result}\n\nAnswer the question using this tool result.",
        )
        return followup.text

    return response.text


if __name__ == "__main__":
    answer = rag_query("What is a transformer?")
    print(answer)

    structured = rag_query_structured("What is a transformer?")
    print(structured)

    tool_answer = rag_query_with_tools("Search the knowledge base for information about transformers")
    print(tool_answer)