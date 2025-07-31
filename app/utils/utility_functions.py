import hashlib
import json
from typing import List
from langchain_core.documents import Document
from app.agents.prompts.qna_agent_prompts import multi_query_prompt_template
from app.utils.llms import google_gemini

class DocumentUtils():

    @staticmethod
    def format_document(docs: List[str]):
        return "\n".join(doc for doc in docs)

    @staticmethod
    def create_unique_id(blob_url: str):
        return int.from_bytes(hashlib.sha256(blob_url.encode()).digest()[:8], 'big')
    
class AgentUtils():

    @staticmethod
    async def generate_multiple_query(user_query: str, chat_model):
        query_generating_chain = (multi_query_prompt_template | google_gemini)
        response = await query_generating_chain.ainvoke({
            "question": user_query
        })

        try:
            result = json.loads(response.content if hasattr(response, "content") else str(response)) # type: ignore
            query_list = [result["query_1"], result["query_2"],user_query]
            print(query_list)
            return query_list
        except:
            return [user_query]

class UtilityContainer(
    DocumentUtils,
    AgentUtils
):
    pass