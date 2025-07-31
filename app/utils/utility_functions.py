import hashlib
from typing import List
from langchain_core.documents import Document

class DocumentUtils():

    @staticmethod
    def format_document(docs: List[str]):
        return "\n".join(doc for doc in docs)

    @staticmethod
    def create_unique_id(blob_url: str):
        return int.from_bytes(hashlib.sha256(blob_url.encode()).digest()[:8], 'big')

class UtilityContainer(
    DocumentUtils
):
    pass