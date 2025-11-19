FROM python:3.10-slim

WORKDIR /app

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa dipendenze Python
COPY model_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il predictor
COPY model_server/predictor.py .

# Scarica il modello durante la build (opzionale, per velocizzare lo startup)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Esponi la porta
EXPOSE 8081

# Comando di avvio
ENTRYPOINT ["python", "predictor.py"]