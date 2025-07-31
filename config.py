from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_cluster_url: str
    qdrant_api_key: str
    collection_name: str
    collection_vector_dimension: int
    gemini_model_name: str
    gemini_api_key: str
    redis_host: str
    redis_username: str
    redis_password: str
    redis_port: int
    model_config = {
        "env_file": ".env"
    }

settings = Settings() # type: ignore