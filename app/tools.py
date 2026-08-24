# app/tools.py
from app.rag import similarity_search

def search_knowledge_base(query: str) -> str:
    """Tool: searches the ingested documents."""
    results = similarity_search(query, k=3)
    return "\n\n".join(results)

def get_word_count(text: str) -> int:
    """Tool: simple example of a non-RAG tool."""
    return len(text.split())

# Registry so main.py / query.py can look tools up by name
AVAILABLE_TOOLS = {
    "search_knowledge_base": search_knowledge_base,
    "get_word_count": get_word_count,
}