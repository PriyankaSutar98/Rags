from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

def retrieve(query: str, top_k:int =5) -> list[dict]:
    query_embedding = embedder.encode(query).tolist()
    client = QdrantClient(url="http://localhost:6333")
    hits = client.query_points(
        collection_name="rag_docs",
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    return [{**hit.payload, "score": round(hit.score, 4)} for hit in hits.points]

