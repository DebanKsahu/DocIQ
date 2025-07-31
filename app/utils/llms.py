from config import settings
from langchain_google_genai import ChatGoogleGenerativeAI

google_gemini = ChatGoogleGenerativeAI(
    model = settings.gemini_model_name,
    temperature = 0.4,
    google_api_key = settings.gemini_api_key
)

google_gemini_output_limit = ChatGoogleGenerativeAI(
    model = settings.gemini_model_name,
    temperature = 0.4,
    google_api_key = settings.gemini_api_key,
    max_output_tokens = 100
)