"""
Script per deployare, versionare ed eseguire la pipeline su Kubeflow
Compatibile con KFP SDK v2.5.0
Include la specifica del Namespace
Utilizza 'Exception' generiche per la gestione errori API.
"""
import os
import sys
import argparse
from datetime import datetime
import kfp
# NON importa ApiException, come da richiesta

# Nomi statici per pipeline ed esperimento
PIPELINE_NAME = "document-processing-pipeline"
EXPERIMENT_NAME = "RAG Document Processing"
PIPELINE_FILE = "document_pipeline.yaml"
# Namespace Kubeflow (richiesto da KFP 2.5.0)
KUBEFLOW_NAMESPACE = "kubeflow-user-example-com" 

def get_or_create_experiment(client: kfp.Client, experiment_name: str):
    """
    Recupera o crea un esperimento su Kubeflow.
    Logica aggiornata: controlla prima l'ID, poi crea se non esiste.
    """
    
    print(f"Verifica esistenza esperimento '{experiment_name}'...")
    
    # 1. Recupera l'ID dell'esperimento (restituisce None se non trovato)
    # Passiamo il namespace come richiesto
    experiment = client.get_experiment(experiment_name=experiment_name, namespace=KUBEFLOW_NAMESPACE)

    # 2. Logica di controllo sull'ID
    # Scenario: Esperimento NON trovato
    if experiment is None or experiment.id == "":
        print(f"🧪 Esperimento '{experiment_name}' non trovato. Creazione in corso...")
        try:
            experiment = client.create_experiment(name=experiment_name, namespace=KUBEFLOW_NAMESPACE)
            print(f"✅ Esperimento creato (ID: {experiment.id})")
            return experiment
        except Exception as e:
            print(f"❌ Errore durante la *creazione* dell'esperimento: {str(e)}")
            raise e
    
    # Scenario: Esperimento TROVATO
    else:
        print(f"🧪 Esperimento '{experiment_name}' trovato (ID: {experiment.id}).")
        # Ora recuperiamo l'oggetto esperimento completo usando l'ID
        return experiment


def upload_pipeline_version_function(client: kfp.Client, pipeline_file: str, pipeline_name: str):
    """
    Carica una pipeline. Se esiste, carica una nuova versione.
    Se non esiste, crea la pipeline.
    Logica aggiornata per controllare se pipeline_id è None o stringa vuota.
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    version_name = f"version-{timestamp}"
    
    print(f"Verifica esistenza pipeline '{pipeline_name}'...")
    
    # 1. Recupera l'ID della pipeline (restituisce None se non trovata)
    pipeline_id = client.get_pipeline_id(name=pipeline_name)
    
    # 2. Logica di controllo su pipeline_id
    # Scenario: Pipeline NON trovata
    if pipeline_id is None or pipeline_id == "":
        print(f"\n📦 Pipeline '{pipeline_name}' non trovata. Creazione nuova pipeline...")
        try:
            pipeline = client.upload_pipeline(
                pipeline_package_path=pipeline_file,
                pipeline_name=pipeline_name,
                description=f"Pipeline per processing documenti AgenticRAG"
            )

            # Gestisci diversi tipi di risposta
            pipeline_id = None
            if hasattr(pipeline, 'pipeline_id'):
                pipeline_id = pipeline.pipeline_id
            elif hasattr(pipeline, 'id'):
                pipeline_id = pipeline.id
            elif hasattr(pipeline_upload, 'name'):
                pipeline_id = pipeline.name
            
            print(f"✅ Pipeline caricata con successo!")
            if pipeline_id:
                print(f"   Pipeline ID: {pipeline_id}")
            print(f"   Nome: {pipeline_name}")
        except Exception as e:
            print(f"❌ Errore durante la *creazione* della pipeline: {str(e)}")
            raise e # Rilancia l'errore se la creazione fallisce
    # Scenario: Pipeline TROVATA
    else:
        print(f"\n📦 Pipeline '{pipeline_name}' trovata (ID: {pipeline_id}).")
        print(f"   Caricamento nuova versione: {version_name}...")
        try:
            client.upload_pipeline_version(
                pipeline_package_path=pipeline_file,
                pipeline_version_name=version_name,
                pipeline_id=pipeline_id
            )
            print(f"✅ Nuova versione '{version_name}' caricata con successo.")
        except Exception as e:
            print(f"❌ Errore durante l'upload della *versione*: {str(e)}")
            raise e # Rilancia l'errore se l'upload della versione fallisce
    return pipeline_id


def run_pipeline(client: kfp.Client, experiment_id: str, pipeline_name: str):
    """
    Esegue l'ultima versione della pipeline specificata.
    """
    print(f"\n🚀 Avvio run per l'ultima versione di '{pipeline_name}'...")
    
    hf_api_key = os.getenv('HF_API_KEY', '')
    minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'minio123')
    
    if not hf_api_key:
        print("⚠️  HF_API_KEY non trovata nelle environment variables")
    
    arguments = {
        'hf_api_key': hf_api_key,
        'minio_secret_key': minio_secret_key,
    }
    
    try:
        run_name = f"run-{pipeline_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        pipeline_id = client.get_pipeline_id(name=pipeline_name)
        
        run = client.run_pipeline(
            experiment_id=experiment_id,
            job_name=run_name, 
            pipeline_id=pipeline_id,
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
    
    if not args.upload and not args.run:
        args.upload = True
        args.run = True
    
    endpoint = args.endpoint or os.getenv('KUBEFLOW_ENDPOINT', 'http://localhost:8888')
    
    print("\n" + "="*60)
    print("🚀 KUBEFLOW PIPELINE MANAGER (v2.0 - KFP 2.5.0 - Generic Exc.)")
    print("="*60)
    print(f"📍 Endpoint:  {endpoint}")
    print(f"📦 Namespace: {KUBEFLOW_NAMESPACE}")
    print(f"📄 Pipeline:  {PIPELINE_FILE}")
    print(f"🏷️  Nome Pipe: {PIPELINE_NAME}")
    print(f"🧪 Esperim.:  {EXPERIMENT_NAME}")
    
    if (args.upload or args.run) and not os.path.exists(PIPELINE_FILE):
        print(f"❌ ERRORE: File {PIPELINE_FILE} non trovato!")
        if args.upload:
            sys.exit(1)
            
    try:
        print("\n🔌 Connessione a Kubeflow...")
        client = kfp.Client(host=endpoint)
        print("✅ Connessione stabilita")
        
        experiment = get_or_create_experiment(client, EXPERIMENT_NAME)
        
        if args.upload:
            if not os.path.exists(PIPELINE_FILE):
                 print(f"❌ ERRORE: {PIPELINE_FILE} non trovato. Esegui 'make compile-pipeline' prima.")
                 sys.exit(1)
            upload_pipeline_version_function(client, PIPELINE_FILE, PIPELINE_NAME)
        
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