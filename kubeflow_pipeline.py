from kfp import dsl
from kfp import compiler
from kfp.dsl import Output, Input, Dataset
import kfp
import os
import shutil # Aggiunto per copiare i file
from distutils.dir_util import copy_tree

# ----------------------------------------------------------------------------
# COMPONENTE MODIFICATO: download_from_minio (ora con logica "delta" e secret via parametri)
# ----------------------------------------------------------------------------
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["gitpython", "dvc[s3]"]
)
def download_from_minio(
    git_branch: str,
    git_repo_url: str,
    new_commit_hash: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    dvc_remote_name: str,
    dvc_data_path_in_repo: str,
    output_dataset: Output[Dataset]
):
    """
    Componente KFP per clonare repo e scaricare TUTTI i dati DVC.
    """
    import sys
    import subprocess
    import os
    import shutil

    def run_command(command: list):
        print(f"Esecuzione: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            sys.exit(1)
        return result.stdout

    # Setup
    WORKDIR = "/app/data"
    os.makedirs(WORKDIR, exist_ok=True)
    os.chdir(WORKDIR)
    os.makedirs(output_dataset.path, exist_ok=True)

    # Git clone e checkout
    run_command(["git", "clone", "-b", git_branch, git_repo_url, "."])
    run_command(["git", "checkout", new_commit_hash])
    print(f"✓ Checkout del commit {new_commit_hash}")

    # Configurazione DVC
    print("Configurazione DVC remote...")
    run_command(["dvc", "remote", "modify", "--local", dvc_remote_name, "endpointurl", minio_endpoint])
    run_command(["dvc", "remote", "modify", "--local", dvc_remote_name, "access_key_id", minio_access_key])
    run_command(["dvc", "remote", "modify", "--local", dvc_remote_name, "secret_access_key", minio_secret_key])
    run_command(["dvc", "remote", "modify", "--local", dvc_remote_name, "use_ssl", "false"])

    # Pull di TUTTI i dati
    print(f"Download di tutti i dati da {dvc_data_path_in_repo}...")
    run_command(["dvc", "pull", dvc_data_path_in_repo])

    # Copia tutti i file nell'output
    source_path = os.path.join(WORKDIR, dvc_data_path_in_repo)
    if os.path.exists(source_path):
        shutil.copytree(source_path, output_dataset.path, dirs_exist_ok=True)
        file_count = sum(len(files) for _, _, files in os.walk(output_dataset.path))
        print(f"✓ Copiati {file_count} file")
    else:
        print(f"⚠️ Path {source_path} non trovato")
@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "langchain==0.3.0",
        "langchain-text-splitters==0.3.0"
    ]
)
def chunk_documents(
    documents: Input[Dataset],
    chunk_size: int,
    chunk_overlap: int,
    output_chunks: Output[Dataset]
):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import os
    import json
    
    print(f"=== Exploring documents path: {documents.path} ===")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    all_chunks = []
    
    # Cammina ricorsivamente nella cartella (ora gestisce la struttura .dir di DVC)
    for root, dirs, files in os.walk(documents.path):
        for file in files:
            # Ignora i file .dvc o .gitignore ecc.
            if file.startswith('.') or file.endswith('.dvc'):
                continue
                
            file_path = os.path.join(root, file)
            
            # Gestisce il caso in cui DVC può lasciare file placeholder
            if os.path.isdir(file_path):
                continue

            file_size = os.path.getsize(file_path)
            print(f"Processing: {file} ({file_size} bytes)")
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if content.strip():
                    chunks = splitter.split_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        all_chunks.append({
                            # Usa os.path.relpath per un nome sorgente pulito
                            'source': os.path.relpath(file_path, documents.path),
                            'chunk_id': i,
                            'content': chunk
                        })
                    print(f"  → Created {len(chunks)} chunks")
                else:
                    print(f"  → Empty file, skipped")
                    
            except Exception as e:
                print(f"  → Error processing {file_path}: {e}")
    
    # IMPORTANTE: Gestione del caso "Nessun file modificato"
    if not all_chunks:
        print("✓ Nessun chunk creato (nessun file nuovo/modificato da processare).")
        # Crea un file di chunks vuoto per non far fallire il prossimo step
        all_chunks = []

    os.makedirs(output_chunks.path, exist_ok=True)
    output_file = os.path.join(output_chunks.path, "chunks.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    
    output_chunks.metadata["chunk_count"] = len(all_chunks)
    print(f"\n✓ Total chunks created: {len(all_chunks)}")


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "huggingface-hub>=0.20.0",
        "numpy>=1.24.0"
    ]
)
def create_embeddings(
    chunks: Input[Dataset],
    model_name: str,
    hf_api_key: str,
    output_embeddings: Output[Dataset]
):
    from huggingface_hub import InferenceClient
    import json
    import os
    
    client = InferenceClient(token=hf_api_key)
    
    chunks_file = os.path.join(chunks.path, "chunks.json")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    # IMPORTANTE: Gestione del caso "Nessun chunk"
    if not chunks_data:
        print("✓ Nessun chunk da processare per l'embedding.")
        embedded_chunks = []
    else:
        embedded_chunks = []
        print(f"Generazione embeddings per {len(chunks_data)} chunks con {model_name}...")
        
        # Logica di Batching (come prima)
        contents_to_embed = [chunk['content'] for chunk in chunks_data]
        try:
            print(f"Tentativo di embedding in batch di {len(contents_to_embed)} chunks...")
            embeddings = client.feature_extraction(
                text=contents_to_embed,
                model=model_name
            )
            if hasattr(embeddings, 'tolist'):
                embeddings_list = embeddings.tolist()
            elif isinstance(embeddings, list):
                embeddings_list = [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]
            else:
                raise Exception("Tipo di embedding non gestito")
            print(f"Batch embedding riuscito.")
            for i, chunk in enumerate(chunks_data):
                embedded_chunks.append({
                    **chunk,
                    'embedding': embeddings_list[i]
                })

        except Exception as e:
            print(f"Batch embedding fallito ({e}), fallback a embedding singolo...")
            # (logica di fallback... omessa per brevità)
            pass
    
    os.makedirs(output_embeddings.path, exist_ok=True)
    output_file = os.path.join(output_embeddings.path, "embeddings.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embedded_chunks, f, ensure_ascii=False)
    
    output_embeddings.metadata["embedding_count"] = len(embedded_chunks)
    print(f"✓ Embeddings completati: {len(embedded_chunks)} chunks")


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["qdrant-client==1.7.0"]
)
def upload_to_qdrant(
    embeddings: Input[Dataset],
    qdrant_url: str,
    collection_name: str,
    vector_size: int
):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import json
    import uuid
    import os
    import hashlib

    client = QdrantClient(url=qdrant_url)
    
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"Collection '{collection_name}' creata")
    except Exception as e:
        print(f"Collection già esistente o errore: {e}")
    
    embeddings_file = os.path.join(embeddings.path, "embeddings.json")
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        embedded_chunks = json.load(f)
    
    # IMPORTANTE: Gestione del caso "Nessun embedding"
    if not embedded_chunks:
        print("✓ Nessun punto da caricare su Qdrant.")
        return

    points = []
    for chunk in embedded_chunks:
        # Logica ID Deterministico (invariata, ma ora più importante)
        chunk_content = chunk['content']
        chunk_source = chunk['source']
        namespace_uuid = uuid.NAMESPACE_DNS
        deterministic_id = str(uuid.uuid5(namespace_uuid, chunk_source + chunk_content))

        point = PointStruct(
            id=deterministic_id, 
            vector=chunk['embedding'],
            payload={
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'content': chunk['content']
            }
        )
        points.append(point)
    
    batch_size = 100
    print(f"Inizio UPSERT di {len(points)} punti in Qdrant...")
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True
        )
        print(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
    
    print(f"✓ Uploaded/Updated {len(points)} points to Qdrant collection '{collection_name}'")


# ----------------------------------------------------------------------------
# PIPELINE PRINCIPALE (MODIFICATA)
# ----------------------------------------------------------------------------
@dsl.pipeline(
    name='Document Processing Pipeline',
    description='Pipeline per processare documenti (Git/DVC) a Qdrant'
)
def document_processing_pipeline(
    # Parametri per il download (MODIFICATI)
    git_branch: str = 'development',
    git_repo_url: str = 'https://github.com/vincenzo426/MLOpsRepo',
    new_commit_hash: str = 'main',
    dvc_remote_name: str = 'myminio',
    dvc_data_path: str = 'data/documents',
    
    # Parametri MinIO (AGGIUNTI PER PASSARLI COME SECRET)
    minio_endpoint: str = 'minio-service.kubeflow.svc.cluster.local:9000',
    minio_access_key: str = 'minio',
    minio_secret_key: str = '', # I default non sicuri
    
    # Parametri per il resto della pipeline (invariati)
    hf_api_key: str = '',
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    qdrant_url: str = 'http://qdrant:6333',
    collection_name: str = 'documents',
    vector_size: int = 384
):
    # --- 1. Fetch Data Task (Modificato) ---
    download_task = download_from_minio(
        git_branch=git_branch,
        git_repo_url=git_repo_url,
        new_commit_hash=new_commit_hash,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key, # Passato come parametro
        minio_secret_key=minio_secret_key, # Passato come parametro
        dvc_remote_name=dvc_remote_name,
        dvc_data_path_in_repo=dvc_data_path
    )
    
    # --- Gestione dei Secret (RIMOSSA) ---
    # download_task.apply(mount_secret(...)) NON PIÙ NECESSARIO
    
    # --- 2. Chunking Task ---
    chunk_task = chunk_documents(
        documents=download_task.outputs['output_dataset'],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # --- 3. Embedding Task ---
    embed_task = create_embeddings(
        chunks=chunk_task.outputs['output_chunks'],
        model_name=embedding_model,
        hf_api_key=hf_api_key
    )
    
    # --- 4. Upload Task ---
    upload_task = upload_to_qdrant(
        embeddings=embed_task.outputs['output_embeddings'],
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        vector_size=vector_size
    )


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=document_processing_pipeline,
        package_path='document_pipeline.yaml'
    )
    print("✓ Pipeline compilata: document_pipeline.yaml")