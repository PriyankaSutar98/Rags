import os
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

GITHUB_RAW_URL = "https://raw.githubusercontent.com/tnahddisttud/sample-doc/refs/heads/main/atliqai_hr_policies.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)
# Loading Document
def load_document(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.text
    #  response.raise_for_status() works same as this code if response.status_code == 200:
     #   return response.text
   # else:
     #   raise Exception(f"Failed to load document from {url}. Status code: {response.status_code}")



#Chunking the document into smaller pieces
def chunking(text: str, chunk_size: int = 50) -> list[dict]:
    clean_lines =[]
    for line in text.splitlines():
        clean_line = line.strip().lstrip("#").strip()  # Remove leading/trailing whitespace and leading hyphens
        if clean_line:  # Only add non-empty lines
            clean_lines.append(clean_line)     

    words = " ".join(clean_lines).split()  # Split the cleaned text into words
    chunks = []
    index=0
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append({
            "chunk_index": index,
            "content": chunk
            })
        index += 1
    return chunks


#Embedding the chunks
def embedding() -> tuple[list[dict], np.ndarray]:
    
    # Load the document from GitHub
    document_text = load_document(GITHUB_RAW_URL)
    print("Document loaded successfully.")

    # Chunk the document into smaller pieces
    chunks = chunking(document_text, chunk_size=50)
    print(f"Document chunked into {len(chunks)} pieces.")

    chunk_texts = [chunk["content"] for chunk in chunks]

    print("Performing embedding on the chunks...")
    embeddings = embedder.encode(chunk_texts)
    print(f"Embedding completed with Shape: {embeddings.shape}.")
    return chunks,embeddings
    

#Storing the chunks and embeddings in Qdrant (Vector Database) 
def store_in_qdrant(chunks: list[dict], embeddings) -> None:
    COLLECTION_NAME = "rag_docs"
    DIM = embedder.get_embedding_dimension()
    qdrant = QdrantClient(url="http://localhost:6333")

    # Create collection similar to table in relational database
    if qdrant.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    else:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
        )

    # Create points to be inserted into the collection similar to a row in a relational database
    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "content": chunk["content"],
            },
        )
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    # Inserting the points into the collection
    response = qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True  # waits till all the points are inserted into the collection
    )
    print(response.status)
    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"Points     : {info.points_count}")
    print(f"Dimensions : {info.config.params.vectors.size}")