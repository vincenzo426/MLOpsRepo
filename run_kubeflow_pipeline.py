"""
Script per deployare la pipeline su Kubeflow
Usato dal workflow GitHub Actions
"""
import os
import sys
from datetime import datetime
import kfp

def deploy_pipeline():
    """Deploy della pipeline su Kubeflow"""
    
    # Leggi configurazione da environment variables
    endpoint = os.getenv('KUBEFLOW_ENDPOINT', 'http://localhost:8888')
    pipeline_file = 'document_pipeline.yaml'
    
    print("\n" + "="*60)
    print("🚀 DEPLOY PIPELINE SU KUBEFLOW")
    print("="*60)
    print(f"📍 Endpoint: {endpoint}")
    print(f"📄 Pipeline: {pipeline_file}")
    
    # Verifica che il file esista
    if not os.path.exists(pipeline_file):
        print(f"❌ ERRORE: File {pipeline_file} non trovato!")
        sys.exit(1)
    
    try:
        # Connetti a Kubeflow
        print("\n🔌 Connessione a Kubeflow...")
        client = kfp.Client(host='http://localhost:8888')
        
        # Verifica connessione
        print("✅ Connessione stabilita")
        
        # Nome pipeline con timestamp per versioning
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        pipeline_name = f"processing-data-for-agenticRAG-{timestamp}"
        
        print(f"\n📦 Upload pipeline...")
        print(f"   Nome: {pipeline_name}")
        
        # Carica la pipeline - usa upload_pipeline_version per le nuove API
        try:
            # Prova prima con il nuovo metodo
            pipeline_upload = client.upload_pipeline(
                pipeline_package_path=pipeline_file,
                pipeline_name=pipeline_name,
                description=f"processing-data-for-agenticRAG {timestamp}"
            )
            
            # Gestisci diversi tipi di risposta
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
            print(f"   Nome: {pipeline_name}")
            
        except Exception as e:
            print(f"⚠️  Metodo standard fallito, provo metodo alternativo...")
            print(f"   Dettagli errore: {str(e)}")
            
            # Metodo alternativo: carica come file
            with open(pipeline_file, 'rb') as f:
                pipeline_content = f.read()
            
            # Questo metodo è più robusto
            print(f"   Usando metodo di upload alternativo...")
            result = client._upload_api.upload_pipeline(
                uploadfile=pipeline_file,
                name=pipeline_name,
                description=f"processing-data-for-agenticRAG - {timestamp}"
            )
            print(f"✅ Pipeline caricata con metodo alternativo!")
        
        # Lista tutte le pipeline per conferma
        print(f"\n📋 Pipeline disponibili su Kubeflow:")
        try:
            pipelines = client.list_pipelines(page_size=10)
            if pipelines and hasattr(pipelines, 'pipelines'):
                for i, p in enumerate(pipelines.pipelines[:5], 1):
                    name = getattr(p, 'display_name', None) or getattr(p, 'name', 'N/A')
                    print(f"   {i}. {name}")
            else:
                print("   (Lista pipeline non disponibile)")
        except Exception as e:
            print(f"   ⚠️  Impossibile listare pipeline: {str(e)}")
        
        print("\n" + "="*60)
        print("✅ DEPLOY COMPLETATO CON SUCCESSO!")
        print("="*60)
        print(f"\n💡 Puoi visualizzare la pipeline su: http://localhost:8080")
        print(f"   Vai su: Pipelines → cerca '{pipeline_name}'")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERRORE durante il deploy:")
        print(f"   {str(e)}")
        print(f"\n🔍 Tipo errore: {type(e).__name__}")
        
        # Debug info
        import traceback
        print(f"\n📋 Stack trace completo:")
        traceback.print_exc()
        
        print("\n" + "="*60)
        return 1


if __name__ == "__main__":
    exit_code = deploy_pipeline()
    sys.exit(exit_code)