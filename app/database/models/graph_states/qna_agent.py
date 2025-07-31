from pydantic import BaseModel

class QnaAgentInputState(BaseModel):
    user_query: str
    doc_id: int

class QnaAgentIntermediateState(BaseModel):
    user_query: str
    doc_info: str

class QnaAgentOutputState(BaseModel):
    agent_answer: str

class QnaAgentOverallState(
    QnaAgentInputState,
    QnaAgentIntermediateState,
    QnaAgentOutputState
):
    pass