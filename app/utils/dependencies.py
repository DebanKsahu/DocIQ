from fastapi.security import OAuth2PasswordBearer
import httpx

class TokenDependency():
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="hackrx/runs")

class IODependency():

    @staticmethod
    async def get_httpx_client():
        async with httpx.AsyncClient() as client:
            yield client

class DepeendencyContainer(
    TokenDependency,
    IODependency
):
    pass