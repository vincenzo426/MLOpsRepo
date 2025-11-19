#
# rag_orchestrator.py
#
import os
import requests
import json
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from qdrant_client import QdrantClient
from huggingface_hub import InferenceClient
from typing import List, Optional

# --- CONFIGURAZIONE ---
# URL del servizio di embedding (interno al cluster Kubernetes)
# Nota: La porta 80 è quella del Service di KServe, che gira il traffico al pod sulla 8080
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service.kubeflow-user-example-com.svc.cluster.local/v1/models/embedding-model:predict")

# Configurazione Qdrant (URL interno al cluster)
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

# Configurazione LLM (Hugging Face)
HF_API_KEY = os.getenv("HF_API_KEY")
HF_LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

if not HF_API_KEY:
    print("⚠️ ATTENZIONE: HF_API_KEY non trovato.")

# --- CLIENTS ---
# Qdrant
try:
    qdrant_client = QdrantClient(url=QDRANT_URL)
    print(f"✓ Client Qdrant configurato su {QDRANT_URL}")
except Exception as e:
    print(f"⚠️ Errore config Qdrant: {e}")
    qdrant_client = None

# Hugging Face
hf_client = InferenceClient(token=HF_API_KEY)

# --- MODELLI DATI ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    retrieved_sources: list[str]

app = FastAPI(title="RAG Orchestrator")

def get_embedding_remote(text: str) -> List[float]:
    """Chiama il microservizio KServe per ottenere l'embedding."""
    payload = {"instances": [text]}
    
    try:
        print(f"Richiesta embedding a: {EMBEDDING_SERVICE_URL}")
        response = requests.post(EMBEDDING_SERVICE_URL, json=payload)
        response.raise_for_status()
        
        # KServe restituisce: {"predictions": [[0.1, 0.2, ...]]}
        data = response.json()
        embedding = data["predictions"][0]
        return embedding
    except Exception as e:
        print(f"❌ Errore chiamata Embedding Service: {e}")
        raise HTTPException(status_code=503, detail=f"Embedding Service error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest = Body(...)):
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant non disponibile")

    # 1. Ottieni Embedding (Chiamata remota)
    query_vector = get_embedding_remote(request.query)

    # 2. Cerca in Qdrant
    print(f"Ricerca Qdrant per: '{request.query}'")
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=request.top_k
    )

    context_text = ""
    sources = set()
    
    for result in search_results:
        payload = result.payload or {}
        if 'content' in payload:
            context_text += f"\n---\n{payload['content']}"
        if 'source' in payload:
            sources.add(payload['source'])

    if not context_text:
        return QueryResponse(answer="Nessun documento rilevante trovato.", retrieved_sources=[])

    # 3. Genera risposta con LLM
    system_message = "Sei un assistente utile. Rispondi alla domanda usando solo il contesto fornito."
    user_message = f"Contesto:{context_text}\n\nDomanda: {request.query}"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    print("Invio a LLM...")
    try:
        response = hf_client.chat_completion(
            model=HF_LLM_MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.7
        )
        answer = response.choices[0].message.content
        return QueryResponse(answer=answer, retrieved_sources=list(sources))

    except Exception as e:
        print(f"❌ Errore LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # KServe si aspetta che ascoltiamo sulla 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)