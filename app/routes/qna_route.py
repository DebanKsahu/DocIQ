from fastapi import APIRouter, Depends, HTTPException
from httpx import AsyncClient
from pypdf import PdfReader
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from uuid import uuid4

from app.database.models.graph_states.qna_agent import QnaAgentInputState
from app.database.models.qna_request_model import QnaAnswer, QnaRequest
from app.database.qdrant import create_async_qdrant_client
from app.utils.dependencies import DepeendencyContainer
from qdrant_client.models import PointStruct
from app.utils.utility_functions import UtilityContainer
from app.utils.embedding_models import google_embedding
from config import settings
from app.agents.qna import qna_agent

qna_router = APIRouter()

@qna_router.post("/hackrx/run")
async def qna_bot(input_data: QnaRequest, http_client: AsyncClient = Depends(DepeendencyContainer.get_httpx_client)):
    response = await http_client.get(input_data.documents)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch PDF")
    pdf_reader = PdfReader(BytesIO(response.content))
    full_text = ("\n").join([ page.extract_text() or "" for page in pdf_reader.pages])
    if not full_text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from PDF")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=1000)
    text_chunks = text_splitter.split_text(full_text)
    documents = []
    uuids = []
    document_id = UtilityContainer.create_unique_id(blob_url=input_data.documents)
    for index, chunk in enumerate(text_chunks):
        new_document = Document(
            page_content=chunk,
            metadata = {
                "doc_id": document_id,
                "original_chunk": chunk
            }
        )
        documents.append(new_document)
        uuids.append(str(uuid4()))
    embeddings = google_embedding.embed_documents(texts=[page.page_content for page in documents], task_type="RETRIEVAL_DOCUMENT")
    points = [
        PointStruct(
            id=uuids[i],
            vector=embeddings[i],
            payload=documents[i].metadata
        )
        for i in range(len(documents))
    ]
    async_qdrant_client = create_async_qdrant_client()
    await async_qdrant_client.upsert(
        collection_name=settings.collection_name,
        points=points
    )
    responses = []
    for index,question in enumerate(input_data.questions):
        agent_response = await qna_agent.ainvoke(input=QnaAgentInputState(user_query=question,doc_id=document_id))
        answer = agent_response.get("agent_answer","")
        responses.append(answer)
    return QnaAnswer(answers=responses)
    