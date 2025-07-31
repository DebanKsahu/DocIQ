from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams
from app.routes.qna_route import qna_router

from app.database.qdrant import create_sync_qdrant_client
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n Lifespan Started \n")
    qdrant_client = create_sync_qdrant_client()
    existing_collections = qdrant_client.get_collections().collections
    collections_names = [collection.name for collection in existing_collections]
    if settings.collection_name not in collections_names:
        qdrant_client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=settings.collection_vector_dimension, distance=Distance.COSINE)
        )
    schema = qdrant_client.get_collection(collection_name=settings.collection_name).payload_schema
    if "doc_id" not in schema:
        qdrant_client.create_payload_index(
            collection_name=settings.collection_name,
            field_name="doc_id",
            field_schema=PayloadSchemaType.INTEGER
        )
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)
app.include_router(qna_router)