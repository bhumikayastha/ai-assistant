# AI Assistant — Week 15 Submission

## 1. Project Summary
This project is a RAG-based AI assistant built with FastAPI, ChromaDB, and Google Gemini. It supports:
- standard document-grounded Q&A
- structured JSON responses
- function/tool calling
- an agentic verification loop for more reliable answers

## 2. Documentation Requirements
### a) Overview and project purpose
The assistant retrieves relevant context from a local document store, injects that context into a Gemini prompt, and answers using only the retrieved content. This reduces hallucinations and makes the system grounded in user-provided or project-provided documents.

### b) Setup and run instructions
#### Install dependencies
```bash
pip install -r requirements.txt
```

#### Configure environment
Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_key_here
```

#### Ingest documents
```bash
python -m app.rag
```

#### Run the API
```bash
uvicorn app.main:app --reload
```

#### Run the web UI
```bash
streamlit run app/ui_streamlit.py
```

#### Docker
```bash
docker compose up --build
```

### c) API and usage details
Available endpoints:
- `GET /` — health check
- `POST /chat` — standard RAG answer
- `POST /chat/structured` — JSON response with confidence
- `POST /chat/tools` — tool-calling route
- `POST /chat/agentic` — self-checking agentic flow

Example request:
```json
{
  "query": "What is a transformer?"
}
```

## 3. Additional Requirements
- Local vector database using ChromaDB
- Embedding model via `sentence-transformers`
- Google Gemini as the LLM backend
- Dockerized deployment
- ONNX conversion example via `convert_onnx.py`
- Evaluation harness for test queries and failure injection

## 4. Updated Source Code
The current implementation includes:
- `app/main.py` — FastAPI routes and request handling
- `app/query.py` — standard RAG and tool-calling logic
- `app/agent.py` — agentic loop with verification and retry behavior
- `app/rag.py` — ingestion and ChromaDB retrieval
- `app/tools.py` — knowledge-base search and utility tools
- `app/cache.py` — lightweight in-memory cache
- `app/ui_streamlit.py` — basic user interface
- `eval_harness.py` — evaluation report generator

## 5. Agentic Feature
The agentic workflow now does the following:
1. Search the knowledge base for relevant chunks
2. Condense the retrieved context
3. Generate a draft answer from Gemini
4. Verify whether the answer is supported by the context
5. Re-query if needed or ask for clarification when required

This loop is implemented in `app/agent.py` and is exposed through the `POST /chat/agentic` endpoint.

## 6. Architecture Diagram
The repository includes an updated architecture diagram showing the agentic loop and the retrieval/verification flow.

## 7. Evaluation Harness
The project includes `eval_harness.py`, which runs a small set of test queries and prints a markdown-style report including:
- task completion
- iteration count
- token usage
- latency
- tool usage validity
- failure classification

Example output format:
```text
Task completion rate: 1/4

| Query | Completed | Iterations | Tokens | Latency(s) | Tool call OK | Failure |
|---|---:|---:|---:|---:|---:|---|
| What is a transformer? | True | 1 | 865 | 8.75 | True | - |
```

## 8. Runtime Notes
- The project expects a valid `GEMINI_API_KEY` in the environment.
- The vector store is local and persisted under `chroma_db/`.
- The current setup is intended for local development and submission review.

## 9. Submission Status
This README reflects the current submission-ready branch in the repository and is aligned with the project code and evaluation artifacts currently present in the workspace.