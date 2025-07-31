from qdrant_client import QdrantClient, AsyncQdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from config import settings


def create_async_qdrant_client():
    qdrant_client = AsyncQdrantClient(
        url=settings.qdrant_cluster_url,
        api_key=settings.qdrant_api_key,
        timeout=30
    )
    return qdrant_client

def create_sync_qdrant_client():
    qdrant_client = QdrantClient(
        url=settings.qdrant_cluster_url,
        api_key=settings.qdrant_api_key,
        timeout=30
    )
    return qdrant_client

def init_sync_qdrant_vectorstore(qdrant_client: QdrantClient, embedding, collection_name: str):
    qdrant = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embedding,
        retrieval_mode=RetrievalMode.DENSE,
    )
    return qdrant