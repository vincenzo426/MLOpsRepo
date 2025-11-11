"""
Script per deployare, versionare ed eseguire la pipeline su Kubeflow
"""
import os
import sys
import argparse
from datetime import datetime
import kfp
from kfp.exceptions import KFPException

# Nomi statici per pipeline ed esperimento
PIPELINE_NAME = "document-processing-pipeline"
EXPERIMENT_NAME = "RAG Document Processing"
PIPELINE_FILE = "document_pipeline.yaml"

def get_or_create_experiment(client: kfp.Client, experiment_name: str):
    """Recupera o crea un esperimento su Kubeflow"""
    try:
        experiment = client.get_experiment(experiment_name=experiment_name)
        print(f"🧪 Esperimento '{experiment_name}' trovato (ID: {experiment.id})")
        return experiment
    except KFPException as e:
        if "No experiment" in str(e):
            print(f"🧪 Esperimento '{experiment_name}' non trovato. Creazione in corso...")
            experiment = client.create_experiment(name=experiment_name)
            print(f"✅ Esperimento creato (ID: {experiment.id})")
            return experiment
        else:
            raise e
    except Exception as e:
        # Gestisce altri possibili errori di connessione o API
        print(f"Errore nel recupero esperimento: {e}")
        # Prova a crearlo come fallback
        try:
            experiment = client.create_experiment(name=experiment_name)
            print(f"✅ Esperimento creato (ID: {experiment.id})")
            return experiment
        except Exception as create_e:
            print(f"❌ Fallita anche la creazione dell'esperimento: {create_e}")
            raise create_e


def upload_pipeline_version(client: kfp.Client, pipeline_file: str, pipeline_name: str):
    """
    Carica una pipeline. Se esiste, carica una nuova versione.
    Se non esiste, crea la pipeline.
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    version_name = f"version-{timestamp}"
    
    try:
        # 1. Controlla se la pipeline esiste
        pipeline = client.get_pipeline(pipeline_name=pipeline_name)
        pipeline_id = pipeline.id
        print(f"\n📦 Pipeline '{pipeline_name}' trovata (ID: {pipeline_id}).")
        print(f"   Caricamento nuova versione: {version_name}...")
        
        # 2. Se esiste, carica una nuova versione
        client.upload_pipeline_version(
            pipeline_package_path=pipeline_file,
            pipeline_version_name=version_name,
            pipeline_id=pipeline_id
        )
        print(f"✅ Nuova versione '{version_name}' caricata con successo.")
        
    except KFPException as e:
        if "No pipeline" in str(e) or "not found" in str(e):
            # 3. Se non esiste, crea una nuova pipeline
            print(f"\n📦 Pipeline '{pipeline_name}' non trovata. Creazione nuova pipeline...")
            pipeline = client.upload_pipeline(
                pipeline_package_path=pipeline_file,
                pipeline_name=pipeline_name,
                description=f"Pipeline per processing documenti AgenticRAG"
            )
            pipeline_id = pipeline.id
            print(f"✅ Pipeline creata con successo (ID: {pipeline_id}).")
        else:
            print(f"❌ Errore KFP durante l'upload: {str(e)}")
            raise e
    except Exception as e:
        print(f"❌ Errore imprevisto durante l'upload: {str(e)}")
        raise e
    
    return pipeline_id


def run_pipeline(client: kfp.Client, experiment_id: str, pipeline_name: str):
    """
    Esegue l'ultima versione della pipeline specificata.
    """
    print(f"\n🚀 Avvio run per l'ultima versione di '{pipeline_name}'...")
    
    # Recupera credenziali da environment
    hf_api_key = os.getenv('HF_API_KEY', '')
    minio_secret_key = os.getenv('MINIO_SECRET_KEY', '')
    
    if not hf_api_key:
        print("⚠️  HF_API_KEY non trovata nelle environment variables")
    
    # Parametri per la pipeline
    arguments = {
        'hf_api_key': hf_api_key,
        'minio_secret_key': minio_secret_key,
    }
    
    try:
        run_name = f"run-{pipeline_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Esegue la pipeline per NOME (KFP userà l'ultima versione di default)
        run = client.run_pipeline(
            experiment_id=experiment_id,
            run_name=run_name,
            pipeline_name=pipeline_name,
            params=arguments
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
                       help='Upload nuova versione della pipeline')
    parser.add_argument('--run', action='store_true',
                       help='Esecuzione ultima versione della pipeline')
    parser.add_argument('--endpoint', default=None,
                       help='Endpoint Kubeflow')
    
    args = parser.parse_args()
    
    # Se non specificato nessun flag, esegui entrambi
    if not args.upload and not args.run:
        args.upload = True
        args.run = True
    
    endpoint = args.endpoint or os.getenv('KUBEFLOW_ENDPOINT', 'http://localhost:8888')
    
    print("\n" + "="*60)
    print("🚀 KUBEFLOW PIPELINE MANAGER (v2.0 - Versioning)")
    print("="*60)
    print(f"📍 Endpoint:  {endpoint}")
    print(f"📄 Pipeline:  {PIPELINE_FILE}")
    print(f"🏷️  Nome Pipe: {PIPELINE_NAME}")
    print(f"🧪 Esperim.:  {EXPERIMENT_NAME}")
    
    if (args.upload or args.run) and not os.path.exists(PIPELINE_FILE):
        print(f"❌ ERRORE: File {PIPELINE_FILE} non trovato!")
        print("   (Necessario per --upload o --run se la pipeline non è mai stata caricata)")
        if not args.upload:
             print("   (Tentativo di --run senza --upload su una pipeline forse inesistente)")
        # Non esce se è solo --run, potrebbe esistere già
        if args.upload:
            sys.exit(1)
            
    try:
        print("\n🔌 Connessione a Kubeflow...")
        client = kfp.Client(host=endpoint)
        print("✅ Connessione stabilita")
        
        # 1. Recupera o crea l'esperimento
        experiment = get_or_create_experiment(client, EXPERIMENT_NAME)
        
        # 2. Se richiesto, compila e carica la versione
        if args.upload:
            if not os.path.exists(PIPELINE_FILE):
                 print(f"❌ ERRORE: {PIPELINE_FILE} non trovato. Esegui 'make compile-pipeline' prima.")
                 sys.exit(1)
            upload_pipeline_version(client, PIPELINE_FILE, PIPELINE_NAME)
        
        # 3. Se richiesto, esegui l'ultima versione
        if args.run:
            run_pipeline(client, experiment.id, PIPELINE_NAME)
        
        print("\n" + "="*60)
        print("✅ OPERAZIONE COMPLETATA CON SUCCESSO!")
        print("="*60)
        print(f"\n💡 Dashboard Kubeflow: http://localhost:8080")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERRORE: {str(e)}")
        import traceback
        print(f"\n📋 Stack trace:")
        traceback.print_exc()
        print("\n" + "="*60)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)