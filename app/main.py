# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio

from app.query import rag_query, rag_query_structured, rag_query_with_tools

app = FastAPI(title="AI Assistant")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong, please try again.", "detail": str(exc)},
    )


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def health_check():
    return {"status": "ok"}


from app.cache import cache_get, cache_set

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, chat_request: ChatRequest):
    cached = cache_get(chat_request.query)
    if cached:
        return {"answer": cached, "cached": True}

    answer = await asyncio.to_thread(rag_query, chat_request.query)
    cache_set(chat_request.query, answer)
    return {"answer": answer, "cached": False}


@app.post("/chat/structured")
async def chat_structured(request: ChatRequest):
    result = await asyncio.to_thread(rag_query_structured, request.query)
    return result.model_dump()


@app.post("/chat/tools")
async def chat_tools(request: ChatRequest):
    answer = await asyncio.to_thread(rag_query_with_tools, request.query)
    return {"answer": answer}