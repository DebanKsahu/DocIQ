import redis.asyncio as redis

def init_redis():
    redis_connection_pool = redis.ConnectionPool(
        host = "localhost",
        port = 6379,
        decode_response = True,
        max_connection = 5
    )

    redis_client = redis.Redis(
        connection_pool=redis_connection_pool
    )
    return redis_client