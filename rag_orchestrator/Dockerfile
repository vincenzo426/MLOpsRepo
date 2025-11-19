FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_orchestrator.py .

# Espone la porta 8080 (standard KServe/Knative)
EXPOSE 8080

CMD ["python", "rag_orchestrator.py"]