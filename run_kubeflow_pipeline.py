#!/usr/bin/env python3
import kfp
import os
import sys
from kubeflow_pipeline import document_processing_pipeline

def run_pipeline():
    try:
        # Client Kubeflow
        client = kfp.Client(host='http://localhost:8080')
        
        # HF API Key da environment
        hf_api_key = os.getenv('HF_API_KEY', 'default')
        
        # Esegui pipeline
        run = client.create_run_from_pipeline_func(
            document_processing_pipeline,
            arguments={
                'minio_bucket': 'dvc-storage',
                'minio_endpoint': 'minio-service.kubeflow.svc.cluster.local:9000',
                'minio_access_key': 'minio',
                'minio_secret_key': 'minio123',
                'hf_api_key': hf_api_key,
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'qdrant_url': 'http://qdrant:6333',
                'collection_name': 'documents',
                'vector_size': 384
            }
        )
        
        print(f"✓ Pipeline triggered: {run.run_id}")
        print(f"View at: http://localhost:8080/#/runs/details/{run.run_id}")
        return 0
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(run_pipeline())