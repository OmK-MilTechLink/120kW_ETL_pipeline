import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from src.path import OUTPUT_DIR, VECTOR_DB_DIR

# =========================================================
# CONFIG
# =========================================================

SCOPE_DIR = OUTPUT_DIR / "scope"
VECTOR_DB_INFO = VECTOR_DB_DIR / "vector_db_info"

COLLECTION_NAME = "standards_scope"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 64

# =========================================================
# CORE EMBEDDING LOGIC (AI + FUNCTIONALITY UNCHANGED)
# =========================================================

def embed_all_scopes() -> int:
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_INFO),
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"level": "document_scope"}
    )

    ids, documents, metadatas = [], [], []
    total = 0

    for scope_file in sorted(SCOPE_DIR.glob("*_scope.json")):
        data = json.loads(scope_file.read_text(encoding="utf-8"))

        document_id = data.get("document_id")
        title = data.get("document_title") or ""
        summary = data.get("summary", "")
        scope = data.get("scope", [])
        tests = data.get("tests", [])

        if not document_id:
            continue

        # ---- EMBEDDING TEXT (UNCHANGED) ----
        embedding_text = "\n\n".join([title] + scope + tests).strip()
        if not embedding_text:
            continue

        ids.append(document_id)
        documents.append(embedding_text)

        # ---- ONLY CHANGE: RICH METADATA STORAGE ----
        metadatas.append({
            "document_id": document_id,
            "document_title": title,
            "summary": summary,
            "tests": "\n".join(tests) if tests else ""
        })

        if len(ids) >= BATCH_SIZE:
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=model.encode(
                    documents,
                    normalize_embeddings=True,
                    batch_size=BATCH_SIZE
                ).tolist(),
                metadatas=metadatas
            )
            total += len(ids)
            ids, documents, metadatas = [], [], []

    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=model.encode(
                documents,
                normalize_embeddings=True,
                batch_size=BATCH_SIZE
            ).tolist(),
            metadatas=metadatas
        )
        total += len(ids)

    return total

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="Information Embedding API", version="2.1")

@app.post("/scope/embed")
def embed_scope_api():
    if not SCOPE_DIR.exists():
        raise HTTPException(status_code=404, detail="Scope directory not found")

    count = embed_all_scopes()

    return {
        "status": "success",
        "documents_embedded": count,
        "vector_db": str(VECTOR_DB_INFO),
        "collection": COLLECTION_NAME
    }

# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    total = embed_all_scopes()
    print(f"[OK] Embedded {total} scope documents")