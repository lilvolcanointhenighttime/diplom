from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import aiohttp

from src.backend.fastapi_checks.routers import router


aiohttp_clientsession: aiohttp.ClientSession = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global aiohttp_clientsession
    aiohttp_clientsession = aiohttp.ClientSession()
    yield
    await aiohttp_clientsession.close()

def get_app() -> FastAPI:
    app = FastAPI(title="FNSChecks", lifespan=lifespan, root_path="/api/v1")

    origins = ["*"]
    app.add_middleware(
        CORSMiddleware, 
        allow_origins=origins,
        allow_credentials=True, 
        allow_methods=['*'], 
        allow_headers=['*'])

    app.include_router(router)
    return app