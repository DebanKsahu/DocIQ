from typing import List
from pydantic import BaseModel

class QnaRequest(BaseModel):
    documents: str
    questions: List[str]

class QnaAnswer(BaseModel):
    answers: List[str]