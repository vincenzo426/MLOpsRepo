def run_pipeline():
    try:
        # Verifica connessione Kubeflow API
        if not ensure_port_forward():
            print("\n❌ Impossibile connettersi a Kubeflow Pipelines API")
            print("   Verifica che:")
            print("   1. Minikube sia in esecuzione: minikube status")
            print("   2. Kubeflow sia installato: kubectl get pods -n kubeflow")
            print("   3. Servizio ml-pipeline esista: kubectl get svc ml-pipeline -n kubeflow")
            print("   4. Esegui manualmente: ./start_portforward.sh")
            return 1
        
        # Client Kubeflow - PORTA API CORRETTA
        print("🔗 Connessione a Kubeflow Pipelines API...")
        client = kfp.Client(host='http://localhost:8888')
        
        # HF API Key da environment
        hf_api_key = os.getenv('HF_API_KEY', 'default')
        
        if hf_api_key == 'default':
            print("⚠️  HF_API_KEY non configurato, usando valore default")
        
        # Import pipeline
        from kubeflow_pipeline import document_processing_pipeline
        
        # Esegui pipeline
        print("🚀 Avvio pipeline...")
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
        print(f"  View at: http://localhost:8080/#/runs/details/{run.run_id}")
        print(f"  (Assicurati che anche il port-forward UI sia attivo per visualizzare)")
        return 0
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        print("\n🔍 Debug Info:")
        print("   - Il client KFP richiede accesso all'API ml-pipeline (porta 8888)")
        print("   - La UI è su porta 8080 (separata dall'API)")
        print("   - Esegui: ./start_portforward.sh per configurare entrambe")
        print("   - Test API: curl http://localhost:8888/apis/v2beta1/healthz")
        return 1

if __name__ == '__main__':
    sys.exit(run_pipeline())


