"""
Script per deployare, versionare ed eseguire la pipeline su Kubeflow
Compatibile con KFP SDK v2.5.0
Modificato per passare credenziali e commit hash
"""
import os
import sys
import argparse
from datetime import datetime
import kfp

PIPELINE_NAME = "document-processing-pipeline"
EXPERIMENT_NAME = "RAG Document Processing"
PIPELINE_FILE = "document_pipeline.yaml"
KUBEFLOW_NAMESPACE = "kubeflow-user-example-com" 

# ... (le funzioni get_or_create_experiment e upload_pipeline_version_function 
#      restano invariate, puoi copiarle da prima) ...

def get_or_create_experiment(client: kfp.Client, experiment_name: str):
    print(f"Verifica esistenza esperimento '{experiment_name}'...")
    experiment = client.get_experiment(experiment_name=experiment_name, namespace=KUBEFLOW_NAMESPACE)
    if experiment is None or experiment.experiment_id == "":
        print(f"🧪 Esperimento '{experiment_name}' non trovato. Creazione in corso...")
        try:
            experiment = client.create_experiment(name=experiment_name, namespace=KUBEFLOW_NAMESPACE)
            print(f"✅ Esperimento creato (ID: {experiment.experiment_id})")
            return experiment
        except Exception as e:
            print(f"❌ Errore durante la *creazione* dell'esperimento: {str(e)}")
            raise e
    else:
        print(f"🧪 Esperimento '{experiment_name}' trovato (ID: {experiment.experiment_id}).")
        return experiment

def upload_pipeline_version_function(client: kfp.Client, pipeline_file: str, pipeline_name: str):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    version_name = f"version-{timestamp}"
    print(f"Verifica esistenza pipeline '{pipeline_name}'...")
    pipeline_id = client.get_pipeline_id(name=pipeline_name)
    pipeline_version_to_return = None
    if pipeline_id is None or pipeline_id == "":
        print(f"\n📦 Pipeline '{pipeline_name}' non trovata. Creazione nuova pipeline...")
        try:
            pipeline = client.upload_pipeline(
                pipeline_package_path=pipeline_file,
                pipeline_name=pipeline_name,
                description=f"Pipeline per processing documenti AgenticRAG"
            )
             # Estrai l'ID della pipeline
            pipeline_id = pipeline.pipeline_id

            if not pipeline_id:
                raise ValueError(f"Nessuna pipeline trovata con nome: {pipeline_name}")

            # 2. Elenca le versioni della pipeline
            versions = client.list_pipeline_versions(pipeline_id=pipeline_id)

            if not versions.pipeline_versions:
                raise ValueError(f"Nessuna versione trovata per la pipeline {pipeline_name}")

            # 3. Seleziona la versione più recente (di solito la default)
            default_version = versions.pipeline_versions[0]
            
            # --- MODIFICA CHIAVE ---
            # Assegna la versione di default al valore di ritorno
            pipeline_version_to_return = default_version.pipeline_version_id
            print(f"✅ Pipeline creata con successo! ID: {pipeline_id}, Versione ID: {pipeline_version_to_return}")
        except Exception as e:
            print(f"❌ Errore durante la *creazione* della pipeline: {str(e)}")
            raise e
    else:
        print(f"\n📦 Pipeline '{pipeline_name}' trovata (ID: {pipeline_id}).")
        print(f"   Caricamento nuova versione: {version_name}...")
        try:
            new_version = client.upload_pipeline_version(
                pipeline_package_path=pipeline_file,
                pipeline_version_name=version_name,
                pipeline_id=pipeline_id
            )
            pipeline_version_to_return = new_version.pipeline_version_id
            print(f"✅ Nuova versione '{version_name}' caricata (ID: {pipeline_version_to_return}).")
        except Exception as e:
            print(f"❌ Errore during l'upload della *versione*: {str(e)}")
            raise e
    return pipeline_version_to_return


def run_pipeline(
    client: kfp.Client, 
    experiment_id: str, 
    pipeline_name: str, 
    version_id: str,
    git_repo: str,
    new_commit: str, # MODIFICATO
    old_commit: str, # NUOVO
    minio_key: str,  # NUOVO
    minio_secret: str # NUOVO
):
    """
    Esegue l'ultima versione della pipeline specificata.
    """
    print(f"\n🚀 Avvio run per l'ultima versione di '{pipeline_name}'...")
    
    hf_api_key = os.getenv('HF_API_KEY', '')
    
    if not hf_api_key:
        print("⚠️  HF_API_KEY non trovata nelle environment variables")
    
    arguments = {
        # --- MODIFICATO ---
        'git_repo_url': git_repo,
        'new_commit_hash': new_commit,
        'old_commit_hash': old_commit,
        'hf_api_key': hf_api_key,
        'minio_access_key': minio_key,
        'minio_secret_key': minio_secret,
    }
    
    try:
        run_name = f"run-{pipeline_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        pipeline_id = client.get_pipeline_id(name=pipeline_name)
        
        run = client.run_pipeline(
            experiment_id=experiment_id,
            job_name=run_name, 
            pipeline_id=pipeline_id,
            version_id=version_id,
            params=arguments
        )
        
        run_id = getattr(run, 'id', None) or getattr(run, 'run_id', None)
        print(f"✅ Pipeline run avviato!")
        if run_id:
            print(f"   Run ID: {run_id}")
            print(f"   Visualizza su: http://localhost:8080/#/runs/details/{run_id}")
        
        return run_id
        
    except Exception as e:
        print(f"❌ Errore during l'esecuzione: {str(e)}")
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
    
    # --- NUOVI ARGOMENTI ---
    parser.add_argument('--new_commit_hash', default='main',
                       help='Git commit hash NUOVO (es. github.sha)')
    parser.add_argument('--git_repo', default='https://github.com/vincenzo426/MLOpsRepo.git',
                       help='URL del repository Git')
    parser.add_argument('--minio_access_key', required=True,
                       help='MinIO Access Key (da GitHub Secrets)')
    parser.add_argument('--minio_secret_key', required=True,
                       help='MinIO Secret Key (da GitHub Secrets)')
    
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
    print(f"🔗 Git Repo:  {args.git_repo}")
    print(f"🔑 Git Commit (Nuovo): {args.new_commit_hash}")
    print(f"🔑 Git Commit (Vecchio): {args.old_commit_hash}")
    print(f"🔒 MinIO Key:  {'*' * len(args.minio_access_key)}")
    
    # ... (il resto della funzione main con connessione, get_experiment, etc.
    #      resta invariato, eccetto la chiamata a run_pipeline) ...
            
    try:
        print("\n🔌 Connessione a Kubeflow...")
        token = os.getenv('KUBEFLOW_PIPELINE_TOKEN')
        if token:
            print("   (Autenticazione con Service Account Token trovata)")
        else:
            print("   (Nessun Token di autenticazione, tentativo di connessione anonima)")
            
        client = kfp.Client(host=endpoint, namespace=KUBEFLOW_NAMESPACE, existing_token=token)
        print("✅ Connessione stabilita")
        
        experiment = get_or_create_experiment(client, EXPERIMENT_NAME)
        
        verision_pipeline = None 
        
        if args.upload:
            if not os.path.exists(PIPELINE_FILE):
                 print(f"❌ ERRORE: {PIPELINE_FILE} non trovato. Esegui 'make compile-pipeline' prima.")
                 sys.exit(1)
            verision_pipeline = upload_pipeline_version_function(client, PIPELINE_FILE, PIPELINE_NAME)
        
        if args.run:
            run_pipeline(
                client, 
                experiment.experiment_id, 
                PIPELINE_NAME, 
                verision_pipeline,
                args.git_repo,
                args.new_commit_hash, # Passa il nuovo hash
                args.minio_access_key, # Passa la chiave
                args.minio_secret_key  # Passa il segreto
            )
        
        print("\n" + "="*60)
        print("✅ OPERAZIONE COMPLETATA CON SUCCESSO!")
        print("="*60)
        
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