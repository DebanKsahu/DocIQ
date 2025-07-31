import redis.asyncio as redis
from config import settings

def init_redis():
    redis_connection_pool = redis.ConnectionPool(
        host = settings.redis_host,
        port = settings.redis_port,
        decode_responses=True,
        username = settings.redis_username,
        password = settings.redis_password,
        max_connections=5
    )

    redis_client = redis.Redis(
        connection_pool=redis_connection_pool
    )
    return redis_client