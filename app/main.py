# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.query import rag_query, rag_query_structured, rag_query_with_tools

app = FastAPI(title="AI Assistant")

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    answer = rag_query(request.query)
    return {"answer": answer}

@app.post("/chat/structured")
def chat_structured(request: ChatRequest):
    result = rag_query_structured(request.query)
    return result.model_dump()

@app.post("/chat/tools")
def chat_tools(request: ChatRequest):
    answer = rag_query_with_tools(request.query)
    return {"answer": answer}