from kfp import dsl
from kfp import compiler
from kfp.dsl import Output, Input, Dataset
import kfp

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["boto3==1.34.0", "minio==7.2.0"]
)
def download_from_minio(
    bucket_name: str,
    minio_endpoint: str,
    access_key: str,
    secret_key: str,
    output_dataset: Output[Dataset]
):
    from minio import Minio
    import os
    
    client = Minio(
        minio_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )
    
    os.makedirs(output_dataset.path, exist_ok=True)
    
    objects = client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        target_path = os.path.join(output_dataset.path, obj.object_name)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        client.fget_object(
            bucket_name,
            obj.object_name,
            target_path
        )
    
    print(f"Downloaded files to {output_dataset.path}")
    output_dataset.metadata["bucket"] = bucket_name
    output_dataset.metadata["file_count"] = len(list(client.list_objects(bucket_name, recursive=True)))


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
    
    for root, dirs, files in os.walk(documents.path):
        for file in files:
            # Skip hidden files e metadata DVC
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"Processing: {file} ({file_size} bytes)")
            
            try:
                # Prova a leggere come testo (file DVC sono raw text senza estensione)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if content.strip():
                    chunks = splitter.split_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        all_chunks.append({
                            'source': file,
                            'chunk_id': i,
                            'content': chunk
                        })
                    print(f"  → Created {len(chunks)} chunks")
                else:
                    print(f"  → Empty file, skipped")
                    
            except Exception as e:
                print(f"  → Error: {e}")
    
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
    
    embedded_chunks = []
    
    print(f"Generazione embeddings per {len(chunks_data)} chunks con {model_name}...")
    
    for i, chunk in enumerate(chunks_data):
        embedding = client.feature_extraction(
            text=chunk['content'],
            model=model_name
        )
        
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        elif isinstance(embedding, list) and len(embedding) > 0 and hasattr(embedding[0], 'tolist'):
            embedding = [e.tolist() if hasattr(e, 'tolist') else e for e in embedding][0]
        
        embedded_chunks.append({
            **chunk,
            'embedding': embedding
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processati {i + 1}/{len(chunks_data)} chunks")
    
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
    
    points = []
    for chunk in embedded_chunks:
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk['embedding'],
            payload={
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'content': chunk['content']
            }
        )
        points.append(point)
    
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
    
    print(f"✓ Uploaded {len(points)} points to Qdrant collection '{collection_name}'")


@dsl.pipeline(
    name='Document Processing Pipeline',
    description='Pipeline per processare documenti da MinIO a Qdrant con LangChain'
)
def document_processing_pipeline(
    minio_bucket: str = 'dvc-storage',
    minio_endpoint: str = 'minio-service.kubeflow.svc.cluster.local:9000',
    minio_access_key: str = 'minio',
    minio_secret_key: str = 'minio123',
    hf_api_key: str = '',
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    qdrant_url: str = 'http://qdrant:6333',
    collection_name: str = 'documents',
    vector_size: int = 384
):
    import os 
    download_task = download_from_minio(
        bucket_name=minio_bucket,
        minio_endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key
    )
    
    chunk_task = chunk_documents(
        documents=download_task.outputs['output_dataset'],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    embed_task = create_embeddings(
        chunks=chunk_task.outputs['output_chunks'],
        model_name=embedding_model,
        hf_api_key=os.getenv('HF_API_KEY', hf_api_key)
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