"""
Script per deployare ed eseguire la pipeline su Kubeflow
"""
import os
import sys
import argparse
from datetime import datetime
import kfp

def upload_pipeline(client, pipeline_file, pipeline_name):
    """Upload della pipeline su Kubeflow"""
    print(f"\n📦 Upload pipeline...")
    print(f"   Nome: {pipeline_name}")
    
    try:
        pipeline_upload = client.upload_pipeline(
            pipeline_package_path=pipeline_file,
            pipeline_name=pipeline_name,
            description=f"processing-data-for-agenticRAG {datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        
        pipeline_id = None
        if hasattr(pipeline_upload, 'pipeline_id'):
            pipeline_id = pipeline_upload.pipeline_id
        elif hasattr(pipeline_upload, 'id'):
            pipeline_id = pipeline_upload.id
        elif hasattr(pipeline_upload, 'name'):
            pipeline_id = pipeline_upload.name
        
        print(f"✅ Pipeline caricata con successo!")
        if pipeline_id:
            print(f"   Pipeline ID: {pipeline_id}")
        
        return pipeline_id
        
    except Exception as e:
        print(f"⚠️  Metodo standard fallito: {str(e)}")
        print(f"   Provo metodo alternativo...")
        
        result = client._upload_api.upload_pipeline(
            uploadfile=pipeline_file,
            name=pipeline_name,
            description=f"processing-data-for-agenticRAG - {datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        print(f"✅ Pipeline caricata con metodo alternativo!")
        return None


def run_pipeline(client, pipeline_file, pipeline_name=None):
    """Esegui una pipeline run con parametri da environment"""
    print(f"\n🚀 Esecuzione pipeline...")
    
    # Recupera credenziali da environment
    hf_api_key = os.getenv('HF_API_KEY', '')
    minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'minio123')
    
    if not hf_api_key:
        print("⚠️  HF_API_KEY non trovata nelle environment variables")
    
    # Parametri per la pipeline
    arguments = {
        'hf_api_key': hf_api_key,
        'minio_secret_key': minio_secret_key,
    }
    
    try:
        run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Usa create_run_from_pipeline_package per passare parametri
        run = client.create_run_from_pipeline_package(
            pipeline_file=pipeline_file,
            arguments=arguments,
            run_name=run_name
        )
        
        run_id = getattr(run, 'id', None) or getattr(run, 'run_id', None)
        print(f"✅ Pipeline run avviato!")
        if run_id:
            print(f"   Run ID: {run_id}")
            print(f"   Visualizza su: http://localhost:8080/#/runs/details/{run_id}")
        
        return run_id
        
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Deploy e Run Kubeflow Pipeline')
    parser.add_argument('--upload', action='store_true', 
                       help='Upload della pipeline su Kubeflow')
    parser.add_argument('--run', action='store_true',
                       help='Esecuzione della pipeline')
    parser.add_argument('--endpoint', default=None,
                       help='Endpoint Kubeflow')
    
    args = parser.parse_args()
    
    # Se non specificato nessun flag, esegui entrambi
    if not args.upload and not args.run:
        args.upload = True
        args.run = True
    
    endpoint = args.endpoint or os.getenv('KUBEFLOW_ENDPOINT', 'http://localhost:8888')
    pipeline_file = 'document_pipeline.yaml'
    
    print("\n" + "="*60)
    print("🚀 KUBEFLOW PIPELINE MANAGER")
    print("="*60)
    print(f"📍 Endpoint: {endpoint}")
    print(f"📄 Pipeline: {pipeline_file}")
    
    if not os.path.exists(pipeline_file):
        print(f"❌ ERRORE: File {pipeline_file} non trovato!")
        sys.exit(1)
    
    try:
        print("\n🔌 Connessione a Kubeflow...")
        client = kfp.Client(host=endpoint)
        print("✅ Connessione stabilita")
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        pipeline_name = f"processing-data-for-agenticRAG-{timestamp}"
        
        if args.upload:
            upload_pipeline(client, pipeline_file, pipeline_name)
        
        if args.run:
            run_pipeline(client, pipeline_file, pipeline_name)
        
        print("\n" + "="*60)
        print("✅ OPERAZIONE COMPLETATA CON SUCCESSO!")
        print("="*60)
        print(f"\n💡 Dashboard Kubeflow: http://localhost:8080")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERRORE: {str(e)}")
        print(f"🔍 Tipo errore: {type(e).__name__}")
        
        import traceback
        print(f"\n📋 Stack trace:")
        traceback.print_exc()
        
        print("\n" + "="*60)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)