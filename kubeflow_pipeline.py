from kfp import dsl
from kfp import compiler
from kfp.dsl import Output, Input, Dataset
import kfp

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["dvc==3.48.0", "dvc-s3==3.2.0", "gitpython"]
)
def download_from_minio(
    git_repo_url: str,
    git_branch: str,
    minio_endpoint: str,
    access_key: str,
    secret_key: str,
    output_dataset: Output[Dataset]
):
    import os
    import shutil
    import subprocess
    
    # Clone del repository
    repo_dir = "/tmp/repo"
    subprocess.run(["git", "clone", "-b", git_branch, git_repo_url, repo_dir], check=True)
    
    os.chdir(repo_dir)
    
    # Configura credenziali MinIO per DVC
    subprocess.run(["dvc", "remote", "modify", "myminio", "access_key_id", access_key], check=True)
    subprocess.run(["dvc", "remote", "modify", "myminio", "secret_access_key", secret_key], check=True)
    subprocess.run(["dvc", "remote", "modify", "myminio", "endpointurl", f"http://{minio_endpoint}"], check=True)
    
    # DVC pull
    subprocess.run(["dvc", "pull"], check=True)
    
    # Copia i documenti scaricati nell'output
    docs_path = os.path.join(repo_dir, "data/documents")
    os.makedirs(output_dataset.path, exist_ok=True)
    
    for item in os.listdir(docs_path):
        src = os.path.join(docs_path, item)
        dst = os.path.join(output_dataset.path, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    
    file_count = len([f for f in os.listdir(output_dataset.path) if os.path.isfile(os.path.join(output_dataset.path, f))])
    output_dataset.metadata["file_count"] = file_count
    print(f"✓ DVC pull completato: {file_count} files")

@dsl.component(
    base_image="python:3.10",
    # --- 1. MODIFICA DIPENDENZE ---
    # Aggiungiamo 'langchain-community' e 'pypdf'
    packages_to_install=[
        "langchain-community==0.2.10", # Per PyPDFLoader
        "langchain-text-splitters==0.2.2", # Per RecursiveCharacterTextSplitter
        "pypdfium2"                # Per la lettura dei PDF
    ]
)
def chunk_documents(
    documents: Input[Dataset],
    chunk_size: int,
    chunk_overlap: int,
    output_chunks: Output[Dataset]
):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFium2Loader
    import os
    import json
    
    print(f"=== Exploring documents path: {documents.path} ===")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    all_chunks = []
    
    for root, dirs, files in os.walk(documents.path):
        for file in files:
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"Processing: {file} ({file_size} bytes)")
            
            try:
                content = None
                
                # --- 3. MODIFICA LOGICA DI LETTURA ---
                if file.lower().endswith('.pdf'):
                    print(f"  → Loading as PDF (using PyPDFium2)...")
                    loader = PyPDFium2Loader(file_path)
                    pdf_pages = loader.load()
                    # Uniamo il testo di tutte le pagine
                    content = "\n".join([doc.page_content for doc in pdf_pages])
                else:
                    # Assumiamo che gli altri file siano di testo
                    # (logica originale)
                    print(f"  → Loading as Text...")
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                # --- FINE MODIFICA ---                
                if content and content.strip():
                    chunks = splitter.split_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        all_chunks.append({
                            'source': file,
                            'chunk_id': i,
                            'content': chunk
                        })
                    print(f"  → Created {len(chunks)} chunks")
                else:
                    print(f"  → Empty file or no content, skipped")
                    
            except Exception as e:
                print(f"  → Error processing {file}: {e}")
    
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
    output_embeddings: Output[Dataset],
    batch_size: int = 128  # Aggiungi un parametro per la dimensione del batch
):
    from huggingface_hub import InferenceClient
    import json
    import os

    client = InferenceClient(token=hf_api_key)

    chunks_file = os.path.join(chunks.path, "chunks.json")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    embedded_chunks = []
    
    print(f"Generazione embeddings per {len(chunks_data)} chunks con {model_name}...")

    # Raccogli solo i testi per il batching
    all_texts = [chunk['content'] for chunk in chunks_data]
    
    # Lista per raccogliere tutti gli embedding calcolati
    all_embeddings = []

    for i in range(0, len(all_texts), batch_size):
        # 1. Crea un batch di testi
        batch_texts = all_texts[i:i + batch_size]
        
        # 2. Esegui UNA chiamata API per l'INTERO batch
        try:
            batch_embeddings = client.feature_extraction(
                text=batch_texts,  # <-- SOLUZIONE: invia una lista di testi
                model=model_name
            )
            
            # Gestione della struttura di output
            if isinstance(batch_embeddings, list):
                all_embeddings.extend([e.tolist() if hasattr(e, 'tolist') else e for e in batch_embeddings])
            else:
                # Se l'output è un singolo tensore/array per il batch
                all_embeddings.extend([e.tolist() if hasattr(e, 'tolist') else e for e in batch_embeddings])

            print(f"Processato batch {i//batch_size + 1}/{(len(all_texts) - 1)//batch_size + 1}")

        except Exception as e:
            print(f"Errore durante l'elaborazione del batch {i}: {e}")
            # Aggiungi N embedding vuoti o gestisci l'errore
            all_embeddings.extend([None] * len(batch_texts))

    # 3. Ricombina i metadati con gli embedding
    for i, chunk in enumerate(chunks_data):
        embedding_value = all_embeddings[i]
        # Salta i chunk che hanno fallito (se ce ne sono)
        if embedding_value is not None:
            embedded_chunks.append({
                **chunk,
                'embedding': embedding_value
            })

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
    import hashlib
    import os
    
    client = QdrantClient(url=qdrant_url)
    
    # Crea collection se non esiste
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
        print(f"Collection già esistente: {e}")
    
    # Carica embeddings
    embeddings_file = os.path.join(embeddings.path, "embeddings.json")
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        embedded_chunks = json.load(f)
    
    # Genera ID deterministici per i nuovi chunk
    new_points = []
    new_ids = set()
    
    for chunk in embedded_chunks:
        chunk_identifier = f"{chunk['source']}_{chunk['chunk_id']}"
        chunk_id = hashlib.md5(chunk_identifier.encode()).hexdigest()
        new_ids.add(chunk_id)
        
        point = PointStruct(
            id=chunk_id,
            vector=chunk['embedding'],
            payload={
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'content': chunk['content']
            }
        )
        new_points.append(point)
    
    # Recupera tutti gli ID esistenti in Qdrant
    existing_ids = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=False,
            with_vectors=False
        )
        existing_ids.update([record.id for record in records])
        if offset is None:
            break
    
    # Identifica chunk obsoleti da eliminare
    obsolete_ids = existing_ids - new_ids
    
    if obsolete_ids:
        print(f"Rimozione {len(obsolete_ids)} chunk obsoleti...")
        client.delete(
            collection_name=collection_name,
            points_selector=list(obsolete_ids)
        )
        print(f"✓ Chunk obsoleti rimossi")
    
    # Upsert nuovi/aggiornati chunk
    batch_size = 100
    for i in range(0, len(new_points), batch_size):
        batch = new_points[i:i+batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"Uploaded batch {i//batch_size + 1}/{(len(new_points)-1)//batch_size + 1}")
    
    print(f"✓ Upload completato: {len(new_points)} chunks, {len(obsolete_ids)} rimossi")

@dsl.pipeline(
    name='Document Processing Pipeline',
    description='Pipeline per processare documenti da MinIO a Qdrant con LangChain'
)
def document_processing_pipeline(
    minio_bucket: str = 'dvc-storage',
    minio_endpoint: str = 'minio-service.kubeflow.svc.cluster.local:9000',
    minio_access_key: str = 'minio',
    minio_secret_key: str = '',
    hf_api_key: str = '',
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    qdrant_url: str = 'http://qdrant:6333',
    collection_name: str = 'documents',
    vector_size: int = 384
):
    download_task = download_from_minio(
        git_repo_url='https://github.com/vincenzo426/MLOpsRepo',  # Inserisci URL del tuo repo
        git_branch='main',
        minio_endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key
    ).set_caching_options(enable_caching=False)
    
    chunk_task = chunk_documents(
        documents=download_task.outputs['output_dataset'],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    embed_task = create_embeddings(
        chunks=chunk_task.outputs['output_chunks'],
        model_name=embedding_model,
        hf_api_key=hf_api_key
    )
    
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