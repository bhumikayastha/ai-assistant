# app/rag.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_or_create_collection("docs", embedding_function=embed_fn)

def ingest_document(path: str, doc_id: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = splitter.split_text(text)
    collection.add(
        documents=chunks,
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        metadatas=[{"source": doc_id} for _ in chunks],
    )
    print(f"Ingested {len(chunks)} chunks from {doc_id}")

def similarity_search(query: str, k: int = 3):
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]

if __name__ == "__main__":
    ingest_document("data/sample.txt", "sample")
    results = similarity_search("your test query here")
    for r in results:
        print("---", r[:200])