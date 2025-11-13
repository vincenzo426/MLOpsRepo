import os
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import uvicorn

# Carica le variabili dal tuo file .env (per HF_API_KEY, ecc.)
load_dotenv()

# --- 1. Configurazione ---
# Legge le variabili d'ambiente.
# Per il test locale, usa localhost per Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "documents" # Come definito in document_pipeline.yaml
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # DEVE corrispondere a document_pipeline.yaml
HF_API_KEY = os.getenv("HF_API_KEY")
HF_LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct" # <-- MODIFICATO

if not HF_API_KEY:
    print("⚠️ ATTENZIONE: HF_API_KEY non trovato nel file .env.")
    print("Le chiamate all'LLM di Hugging Face falliranno.")

# --- 2. Modelli Pydantic per Richiesta/Risposta ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    retrieved_sources: list[str]

# --- 3. Inizializzazione Globale (Modelli caricati all'avvio) ---
# Questi oggetti vengono creati una sola volta all'avvio del server.
try:
    print("Inizializzazione servizio RAG...")
    
    print(f"Caricamento modello di embedding: {EMBEDDING_MODEL_NAME}...")
    # 1. Carica il modello di embedding (SentenceTransformer)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✓ Modello di embedding caricato.")

    print(f"Connessione a Qdrant: {QDRANT_URL}...")
    # 2. Si connette a Qdrant
    qdrant_client = QdrantClient(url=QDRANT_URL)
    #qdrant_client.health_check()
    print("✓ Connesso a Qdrant.")

    print("Inizializzazione client Hugging Face...")
    # 3. Inizializza il client per l'LLM
    hf_client = InferenceClient(token=HF_API_KEY)
    print(f"Utilizzo chiave: {HF_API_KEY}")
    print("✓ Client Hugging Face inizializzato.")
    
    print("\n🚀 Servizio RAG pronto.")
    
except Exception as e:
    print(f"❌ Errore critico durante l'inizializzazione: {e}")
    print("Il servizio potrebbe non funzionare. Controlla che Qdrant sia in esecuzione.")
    embedding_model = None
    qdrant_client = None
    hf_client = None

# --- 4. Creazione dell'App FastAPI ---
app = FastAPI(
    title="RAG Inference API (Local)",
    description="API per interrogare un sistema RAG con Qdrant e Hugging Face"
)

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest = Body(...)):
    """
    Esegue una query RAG completa:
    1. Vettorizza la query.
    2. Cerca contesti simili in Qdrant.
    3. Invia query + contesti a un LLM per la risposta.
    """
    if not all([embedding_model, qdrant_client, hf_client]):
        raise HTTPException(status_code=503, detail="Servizio non inizializzato correttamente. Controlla i log all'avvio.")
        
    try:
        # --- Step 1: Vettorizza la query ---
        print(f"\nQuery ricevuta: {request.query}")
        query_vector = embedding_model.encode(request.query).tolist()

        # --- Step 2: Cerca in Qdrant ---
        print(f"Ricerca in Qdrant (top_k={request.top_k})...")
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=request.top_k,
            with_payload=True
        )
        
        # --- Step 3: Estrai contesto e sorgenti ---
        context = ""
        sources = set()
        
        if not search_results:
            print("Nessun risultato trovato in Qdrant.")
            return QueryResponse(answer="Non ho trovato informazioni rilevanti nei documenti per rispondere a questa domanda.", retrieved_sources=[])

        print(f"Trovati {len(search_results)} chunk rilevanti.")
        for result in search_results:
            if 'content' in result.payload:
                context += f"\n---\n{result.payload['content']}"
            if 'source' in result.payload:
                sources.add(result.payload['source'])

        # --- Step 4: Costruisci i Messaggi per 'chat_completion' ---
        # (Questo sostituisce il vecchio prompt in formato [INST])
        
        # 4.1. Definiamo le istruzioni di sistema
        system_message = (
            "Sei un assistente AI. Usa esclusivamente il seguente contesto per rispondere alla domanda."
            " Rispondi solo in base al contesto fornito."
            " Se la risposta non è nel contesto, dì 'Non ho trovato informazioni sufficienti per rispondere'."
        )
        
        # 4.2. Creiamo il prompt dell'utente con contesto e domanda
        user_message = f"""
        Contesto:
        {context}

        Domanda:
        {request.query}
        """
        
        # 4.3. Formattiamo i messaggi come richiesto dall'API
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # --- Step 5: Chiama l'LLM via Hugging Face (con chat_completion) ---
        print(f"Invio prompt all'LLM ({HF_LLM_MODEL}) con il task 'chat_completion'...")
        
        # Sostituiamo hf_client.text_generation con hf_client.chat_completion
        response = hf_client.chat_completion(
            model=HF_LLM_MODEL,
            messages=messages,
            max_tokens=512,  # 'max_new_tokens' diventa 'max_tokens' in questa API
            temperature=0.7
        )
        
        # Estraiamo la risposta (il formato è diverso)
        response_text = response.choices[0].message.content
        
        print(f"Risposta LLM: {response_text.strip()}")
        return QueryResponse(
            answer=response_text.strip(),
            retrieved_sources=list(sources)
        )

    except Exception as e:
        print(f"❌ Errore durante l'elaborazione della query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Controlla la salute del servizio e la connessione a Qdrant."""
    try:
        qdrant_client.health_check()
        return {"status": "ok", "qdrant_connection": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant non raggiungibile: {e}")

# --- 5. Blocco di Esecuzione ---
# Questo permette di eseguire lo script direttamente con: python rag_api_local.py
if __name__ == "__main__":
    print("Avvio server FastAPI locale su http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)