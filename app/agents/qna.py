from langgraph.graph import END, START, StateGraph

from app.agents.prompts.qna_agent_prompts import qna_chat_prompt_template
from app.database.models.graph_states.qna_agent import (
    QnaAgentInputState,
    QnaAgentIntermediateState,
    QnaAgentOutputState,
    QnaAgentOverallState,
)
from app.database.qdrant import create_async_qdrant_client
from app.utils.embedding_models import google_embedding
from app.utils.llms import google_gemini
from app.utils.utility_functions import UtilityContainer
from config import settings


async def data_fetch(state: QnaAgentInputState) -> QnaAgentIntermediateState:
    qdrant_client = create_async_qdrant_client()
    query_vector = google_embedding.embed_query(state.user_query)
    results = await qdrant_client.search(
        collection_name=settings.collection_name,
        query_vector=query_vector,
        limit=5
    )
    original_chunks = []
    for result in results:
        if (result.payload is not None):
            original_chunks.append(result.payload.get("original_chunk",""))
    return QnaAgentIntermediateState(
        user_query=state.user_query,
        doc_info=UtilityContainer.format_document(original_chunks)
    )

async def final_answer(state: QnaAgentIntermediateState) -> QnaAgentOutputState:
    qna_chain = (qna_chat_prompt_template | google_gemini)
    result = await qna_chain.ainvoke(
        {
            "user_query": state.user_query,
            "document_data": state.doc_info
        }
    )
    return QnaAgentOutputState(
        agent_answer=str(result.content)
    )

graph_builder = StateGraph(
    state_schema=QnaAgentOverallState,
    input_schema=QnaAgentInputState,
    output_schema=QnaAgentOutputState
)
graph_builder.add_node("data_fetch",data_fetch)
graph_builder.add_node("final_answer",final_answer)

graph_builder.add_edge(START,"data_fetch")
graph_builder.add_edge("data_fetch","final_answer")
graph_builder.add_edge("final_answer",END)

qna_agent = graph_builder.compile()