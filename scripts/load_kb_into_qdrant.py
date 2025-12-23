import os
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import torch

# ---------- Config ----------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "infra_kb"
KB_FILE = os.path.join("data", "kb_docs", "infra_sample_kb.txt")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384  # bge-small-en-v1.5 dimension
# ----------------------------

def load_kb_snippets(path: str):
    """Very simple loader: splits on blank lines / [DOC] markers."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by blank lines or [DOC] markers – quick & dirty for now
    raw_chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    snippets = []
    for chunk in raw_chunks:
        # You can keep it simple: whole chunk as one snippet
        snippets.append(chunk)
    return snippets

def main():
    # 1) Connect to Qdrant
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

    # 2) Create collection if not exists
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection {COLLECTION_NAME} already exists")

    # 3) Load KB snippets
    snippets = load_kb_snippets(KB_FILE)
    print(f"Loaded {len(snippets)} snippets from KB")

    # 4) Load embedding model (GPU if available)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Model moved to GPU")
    else:
        print("GPU not available, using CPU")

    # 5) Embed snippets
    embeddings = model.encode(snippets, show_progress_bar=True)

    # 6) Upsert into Qdrant
    points = []
    for idx, (text, vector) in enumerate(zip(snippets, embeddings)):
        points.append(
            models.PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={"text": text},
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    print(f"Inserted {len(points)} points into {COLLECTION_NAME}")

    # 7) Quick test: search with a heartbeat query
    test_query = "SCOM agent heartbeat failure troubleshooting"
    q_vec = model.encode([test_query])[0].tolist()
    
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=3,
    )

    print("\nSearch results for:", test_query)
    for r in search_result.points:
        print("Score:", r.score)
        print("Text:", r.payload["text"])
        print("-" * 40)

if __name__ == "__main__":
    main()