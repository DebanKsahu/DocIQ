from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import settings
from pydantic import SecretStr

google_embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=SecretStr(settings.gemini_api_key)
)