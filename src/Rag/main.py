import getpass
import os
from Rag.ingestion import embedding, store_in_qdrant
from Rag.retrieval import retrieve
from Rag.generation import generation

def rag(query: str, top_k: int = 5):
    print("**************Ingestion**************")
    chunks, embeddings = embedding()
    print("**************Store embedding in DB**************")
    store_in_qdrant(chunks, embeddings)
    print("**************Retrieval**************")
    retrieved_chunks = retrieve(query, top_k)

    print("**************Building Context**************")
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(f"[Source {i}]\n{chunk['content']}")

    context = "\n\n".join(parts)

    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    answer = generation(user_message)
    print("**************Generation**************")
    return answer,context

def main() -> None:
    answer, context = rag("What are the main topics covered in this document?")
    print(answer)
    print(f"{250*'='}")
    print(f"\n\nSOURCES:\n {context}")
    

    



if __name__ == "__main__":
    main()
