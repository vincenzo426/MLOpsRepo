#!/usr/bin/env python3
import kfp

# Compila la pipeline
from kubeflow_pipeline import document_processing_pipeline
from kfp import compiler

compiler.Compiler().compile(
    pipeline_func=document_processing_pipeline,
    package_path='document_pipeline.yaml'
)

# Esegui la pipeline
client = kfp.Client(host='http://localhost:8080')  # URL del tuo Kubeflow

run = client.create_run_from_pipeline_func(
    document_processing_pipeline,
    arguments={
        'minio_bucket': 'dvc-bucket',
        'minio_endpoint': 'minio:9000',  # Nome servizio in Kubernetes
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'qdrant_url': 'http://qdrant:6333',  # Nome servizio in Kubernetes
        'collection_name': 'documents',
        'vector_size': 384
    }
)

print(f"Pipeline eseguita: {run.run_id}")