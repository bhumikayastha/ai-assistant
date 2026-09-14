# eval_harness.py
import time
from typing import Any, Dict, List

from app.agent import run_agentic_query


TEST_QUERIES = [
    "What is a transformer?",
    "How does attention differ from convolution?",
    "asdkjaskjd nonsense query zzz",
    "Compare RNNs and transformers for long sequences",
]


def classify_failure(result: Dict[str, Any]) -> str | None:
    if result["stopped_reason"] == "clarification_needed":
        return None
    if str(result["stopped_reason"]).startswith("tool_error"):
        return "soft_failure"
    if str(result["stopped_reason"]).startswith("llm_error"):
        return "soft_failure"
    if result["answer"] is None:
        return "hard_failure"
    if result["stopped_reason"] == "max_iterations_reached":
        return "cascading_soft_failure"
    return None


def tool_call_correctness(result: Dict[str, Any]) -> bool:
    calls = result.get("tool_calls", [])
    if not calls:
        return False
    tool_names = [str(call.get("tool", "")).lower() for call in calls]
    return any("search" in name or "answer" in name for name in tool_names)


def run_eval() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(TEST_QUERIES):
        if i > 0:
            time.sleep(0.25)
        start = time.time()
        result = run_agentic_query(q)
        latency = round(time.time() - start, 2)
        rows.append({
            "query": q,
            "completed": result["answer"] is not None or result["stopped_reason"] == "clarification_needed",
            "iterations": len(result["trace"]),
            "tokens": result["tokens"],
            "latency_s": latency,
            "tool_call_ok": tool_call_correctness(result),
            "failure_type": classify_failure(result),
        })
    return rows


def run_failure_injection_test() -> Dict[str, Any]:
    result = run_agentic_query("What is a transformer?", inject_failure=True)
    return {
        "query": "What is a transformer?",
        "stopped_reason": result["stopped_reason"],
        "answer": result["answer"],
        "tool_calls": result.get("tool_calls", []),
        "failure_type": classify_failure(result),
    }


def print_report(rows: List[Dict[str, Any]]) -> None:
    completed = sum(1 for r in rows if r["completed"])
    total = len(rows)
    print(f"Task completion rate: {completed}/{total}\n")
    print("| Query | Completed | Iterations | Tokens | Latency(s) | Tool call OK | Failure |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        print(f"| {r['query'][:30]} | {r['completed']} | {r['iterations']} | {r['tokens']} | {r['latency_s']} | {r['tool_call_ok']} | {r['failure_type'] or '-'} |")

    print("\nFailure injection test:")
    print(run_failure_injection_test())


if __name__ == "__main__":
    print_report(run_eval())