from kfp import dsl
from kfp import compiler
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
    output_path: dsl.OutputPath(str)
):
    from minio import Minio
    import os
    
    client = Minio(
        minio_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )
    
    os.makedirs(output_path, exist_ok=True)
    
    objects = client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        client.fget_object(
            bucket_name,
            obj.object_name,
            f"{output_path}/{obj.object_name}"
        )
    
    print(f"Downloaded files to {output_path}")


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "langchain==0.3.0",
        "langchain-text-splitters==0.3.0"
    ]
)
def chunk_documents(
    documents_path: dsl.InputPath(str),
    chunk_size: int,
    chunk_overlap: int,
    output_chunks: dsl.OutputPath(str)
):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import os
    import json
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    all_chunks = []
    
    for root, dirs, files in os.walk(documents_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                chunks = splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        'source': file,
                        'chunk_id': i,
                        'content': chunk
                    })
    
    with open(output_chunks, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    
    print(f"Created {len(all_chunks)} chunks")


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "langchain==0.3.0",
        "langchain-huggingface==0.1.0",
        "sentence-transformers==2.5.0"
    ]
)
def create_embeddings(
    chunks_path: dsl.InputPath(str),
    model_name: str,
    output_embeddings: dsl.OutputPath(str)
):
    from langchain_huggingface import HuggingFaceEmbeddings
    import json
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    embedded_chunks = []
    
    for chunk in chunks:
        embedding = embeddings.embed_query(chunk['content'])
        embedded_chunks.append({
            **chunk,
            'embedding': embedding
        })
    
    with open(output_embeddings, 'w', encoding='utf-8') as f:
        json.dump(embedded_chunks, f, ensure_ascii=False)
    
    print(f"Created embeddings for {len(embedded_chunks)} chunks")


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["qdrant-client==1.7.0"]
)
def upload_to_qdrant(
    embeddings_path: dsl.InputPath(str),
    qdrant_url: str,
    collection_name: str,
    vector_size: int
):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import json
    import uuid
    
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
    except Exception as e:
        print(f"Collection già esistente: {e}")
    
    with open(embeddings_path, 'r', encoding='utf-8') as f:
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
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Uploaded {len(points)} points to Qdrant")


@dsl.pipeline(
    name='Document Processing Pipeline',
    description='Pipeline per processare documenti da MinIO a Qdrant con LangChain'
)
def document_processing_pipeline(
    minio_bucket: str = 'dvc-storage',
    minio_endpoint: str = 'localhost:9000',
    minio_access_key: str = 'minioadmin',
    minio_secret_key: str = 'minioadmin',
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    qdrant_url: str = 'http://localhost:6333',
    collection_name: str = 'documents',
    vector_size: int = 384
):
    # Step 1: Download da MinIO
    download_task = download_from_minio(
        bucket_name=minio_bucket,
        minio_endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key
    )
    
    # Step 2: Chunking
    chunk_task = chunk_documents(
        documents_path=download_task.output,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Step 3: Embedding
    embed_task = create_embeddings(
        chunks_path=chunk_task.output,
        model_name=embedding_model
    )
    
    # Step 4: Upload a Qdrant
    upload_task = upload_to_qdrant(
        embeddings_path=embed_task.output,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        vector_size=vector_size
    )


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=document_processing_pipeline,
        package_path='document_pipeline.yaml'
    )