# AI Assistant — RAG-based Chat API

# AI Assistant — RAG-based Chat API

## Overview
A FastAPI-based AI assistant using Google Gemini, with a Retrieval-Augmented
Generation (RAG) pipeline for document-grounded answers, structured JSON
output, and tool calling.

## Tech Stack
- **LLM:** Google Gemini (gemini-3.6-flash) — chosen for free-tier access, no billing required
- **Backend:** FastAPI + Uvicorn
- **Vector DB:** ChromaDB (embedded/local, no extra infra needed)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2) — runs locally, no API cost
- **Chunking:** LangChain's RecursiveCharacterTextSplitter (800 chars, 100 overlap)
- **Containerization:** Docker

## Project Structure
```
ai-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI backend
│   ├── rag.py             # ingestion, chunking, embeddings, vector store
│   ├── llm_client.py       # Gemini API wrapper
│   ├── tools.py             # tool/function definitions
│   ├── query.py               # RAG query logic (plain, structured, tool-calling)
│   └── ui_streamlit.py         # Streamlit web UI
├── data/
│   └── sample.txt          # source document(s) for ingestion
├── chroma_db/                # vector store (generated, not committed)
├── Dockerfile
├── requirements.txt
├── .env                        # API key (not committed)
├── .gitignore
├── architecture.png
└── README.md
```

## Setup
1. Clone the repo:
   ```
   git clone <your-repo-url>
   cd ai-assistant
   ```
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your-key-here
   ```
   Get a free key at https://aistudio.google.com/apikey
5. Ingest documents into the vector store:
   ```
   python -m app.rag
   ```
6. Run the API server:
   ```
   uvicorn app.main:app --reload
   ```
7. Open `http://127.0.0.1:8000/docs` to test endpoints interactively

## Running the Web UI
With the API server already running (step 6 above), in a separate terminal:
```
streamlit run app/ui_streamlit.py
```
Opens automatically at `http://localhost:8501`.

## Running with Docker
```
docker build -t ai-assistant .
docker run -p 8000:8000 --env-file .env ai-assistant
```
Then visit `http://127.0.0.1:8000/docs` as usual.

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/chat` | Plain-text RAG-grounded answer |
| POST | `/chat/structured` | JSON-structured answer with confidence level |
| POST | `/chat/tools` | Answer generated via tool-calling (knowledge base search) |

Example request:
```json
POST /chat
{ "query": "What is a transformer?" }
```

Example response:
```json
{ "answer": "A transformer is a deep learning neural network architecture..." }
```

## Architecture
See `architecture.png`. Data flow:

```
User → FastAPI backend → similarity search (ChromaDB)
                              ↓
                   context injected into Gemini prompt
                              ↓
                        response returned to user
```

## Design Decisions & Notes
- **Provider choice:** Chose Gemini over Anthropic/OpenAI for this assignment
  specifically because it offers a usable free tier without requiring billing
  setup, while still meeting the assignment's "major LLM provider" requirement.
- **Prompting over sampling parameters:** Temperature/top_p tuning was
  initially planned, but current-generation model APIs (both Anthropic's newer
  Claude models and this project's Gemini setup) increasingly favor
  prompt-based behavior control over manual sampling parameters. Output
  consistency here is instead controlled through explicit system prompt
  instructions (e.g. "answer only from the provided context").
- **Vector DB choice:** ChromaDB was used in embedded/local mode rather than a
  hosted vector DB (Pinecone, Weaviate) since it requires zero extra
  infrastructure — appropriate for this project's scope.
- **Local model serving (vLLM):** [Fill in honestly — e.g. "Attempted serving
  Mistral-7B-Instruct via vLLM; not completed due to no local GPU access. In a
  production deployment this would run on a GPU-backed instance (e.g. AWS
  g5.xlarge)." OR describe what you actually did if you completed this step.]

## Known Limitations
- Single-document ingestion demoed; scaling to many documents would need
  batched ingestion and metadata-based filtering.
- No persistent conversation history between requests (each call is stateless).
- Tool calling currently supports two example tools (`search_knowledge_base`,
  `get_word_count`); easily extensible via the `AVAILABLE_TOOLS` registry in
  `app/tools.py`.

- **Local model serving (vLLM):** Not implemented in this submission due to
  lack of local GPU access — vLLM requires a CUDA-capable GPU with sufficient
  VRAM (typically 16GB+ for a 7B model like Mistral-7B or Llama 3 8B) to serve
  models efficiently. On available hardware (CPU-only), inference through
  vLLM would be impractically slow and isn't representative of its intended
  use case. In a production or GPU-backed environment, this would be
  implemented as:
  python -m vllm.entrypoints.openai.api_server
--model mistralai/Mistral-7B-Instruct-v0.2
--port 8001
  exposing an OpenAI-compatible endpoint that the existing `llm_client.py`
  could call as a fallback or alternative provider, following the same
  pattern already used for the Gemini API integration.

## Deliverables Checklist
- [x] Source code
- [x] Dockerfile
- [x] README (this file)
- [x] Architecture diagram (`architecture.png`)

**ONNX conversion:** The main LLM (Gemini) is a hosted API — ONNX conversion
  isn't applicable since we don't have access to its weights. Instead, the
  RAG pipeline's embedding model (all-MiniLM-L6-v2) was converted to ONNX
  using Hugging Face Optimum, which reduces embedding inference latency by
  removing the PyTorch runtime dependency.

## Deployment
1. Ensure Docker Desktop is installed and running
2. Create `.env` with your `GEMINI_API_KEY`
3. Run: `docker compose up --build`
4. Visit `http://localhost:8501` for the UI, `http://localhost:8000/docs` for the API
5. To stop: `docker compose down`