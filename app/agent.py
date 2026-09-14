# app/agent.py

from tenacity import RetryError
import json
from app.tools import search_knowledge_base
from app.llm_client import call_llm_full

MAX_ITERATIONS = 3

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY
the provided context below. If the answer isn't in the context, say you don't know."""

VERIFY_PROMPT = """Question: {question}
Draft answer: {answer}
Retrieved context (condensed): {notes}

Decide if the draft answer is fully supported by the context.
Respond with ONLY this JSON shape:
{{"verdict": "SUFFICIENT" or "NEEDS_MORE_SEARCH" or "NEEDS_CLARIFICATION",
  "refined_query": "<a narrower/different search query, only if NEEDS_MORE_SEARCH>",
  "clarifying_question": "<only if NEEDS_CLARIFICATION>"}}
"""

def condense(raw_context: str, max_chars_per_chunk: int = 150) -> str:
    """Context-engineering step: compaction.
    search_knowledge_base joins up to 3 full 800-char chunks with '\\n\\n'.
    Passing that raw text back into the prompt on every retry would let the
    scratchpad balloon after 2-3 loop iterations. Instead we keep only a
    short summary of each chunk across iterations."""
    chunks = raw_context.split("\n\n")
    return "\n".join(c[:max_chars_per_chunk] for c in chunks if c.strip())



def run_agentic_query(question: str, inject_failure: bool = False) -> dict:
    trace = []
    current_query = question
    final_answer = None
    total_tokens = 0
    stopped_reason = "max_iterations_reached"
    tool_calls = []

    for step in range(1, MAX_ITERATIONS + 1):
        try:
            if inject_failure and step == 1:
                raise RuntimeError("Injected tool failure for evaluation")
            raw_context = search_knowledge_base(current_query)
            tool_calls.append({"tool": "search_knowledge_base", "query": current_query})
        except Exception as e:
            return {
                "answer": None,
                "clarifying_question": None,
                "trace": trace,
                "tokens": total_tokens,
                "stopped_reason": f"tool_error: {e}",
                "tool_calls": tool_calls,
            }

        notes = condense(raw_context)

        try:
            draft_prompt = f"Context:\n{notes}\n\nQuestion: {question}\n\nAnswer using only the context."
            draft = call_llm_full(draft_prompt, system_prompt=SYSTEM_PROMPT)
            final_answer = draft["text"]
            total_tokens += draft["tokens"]

            verify = call_llm_full(
                VERIFY_PROMPT.format(question=question, answer=final_answer, notes=notes),
                json_mode=True,
            )
            total_tokens += verify["tokens"]
        except RetryError as e:
            return {
                "answer": None,
                "clarifying_question": None,
                "trace": trace,
                "tokens": total_tokens,
                "stopped_reason": f"llm_error: {e}",
                "tool_calls": tool_calls,
            }

        try:
            verdict = json.loads(verify["text"])
            if not isinstance(verdict, dict):
                verdict = {"verdict": "SUFFICIENT"}
        except (json.JSONDecodeError, TypeError, ValueError):
            verdict = {"verdict": "SUFFICIENT"}

        decision = verdict.get("verdict", "SUFFICIENT")

        trace.append({
            "step": step,
            "query_used": current_query,
            "verdict": decision,
            "tool_called": "search_knowledge_base",
        })

        if decision == "SUFFICIENT":
            stopped_reason = "converged"
            break
        elif decision == "NEEDS_CLARIFICATION":
            return {
                "answer": None,
                "clarifying_question": verdict.get("clarifying_question"),
                "trace": trace,
                "tokens": total_tokens,
                "stopped_reason": "clarification_needed",
                "tool_calls": tool_calls,
            }
        elif decision == "NEEDS_MORE_SEARCH":
            current_query = verdict.get("refined_query", question)
            continue

    return {
        "answer": final_answer,
        "clarifying_question": None,
        "trace": trace,
        "tokens": total_tokens,
        "stopped_reason": stopped_reason,
        "tool_calls": tool_calls,
    }

    