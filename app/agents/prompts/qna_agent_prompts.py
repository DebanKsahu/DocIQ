from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

qna_system_message = """
You are an intelligent assistant that answers user questions based strictly on the content of a provided document.

Your job is to:
1. Read and understand the content of the given document.
2. Answer the user's question **only using the information present in the document**.
3. Keep your answer clear, concise, and factual.
4. If the answer is not found in the document, clearly say: "The answer is not available in the provided document."

Instructions:
- Only return a plain text string as the final answer.
- Do not return lists, JSON, bullet points, or markdown unless explicitly asked.
- Do not hallucinate or assume anything beyond the given content.

You will be given:
- A document (text extracted from a user-uploaded file)
- A user question related to that document
"""

qna_human_prompt = """
<user_query>
{user_query}
</user_query>
<document_data>
{document_data}
</document_data>
<answer>

</answer>
"""

qna_chat_prompt_template = ChatPromptTemplate([
    SystemMessage(content=qna_system_message),
    ("human",qna_human_prompt)
])

multi_query_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates multiple search queries from a user question."),
    ("human", "{question}")
])