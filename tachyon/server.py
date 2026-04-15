## to-do currently we dont have streaming working

import asyncio
import uuid
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tachyon.engine.llm import Engine, Request as EngineRequest


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class Model(BaseModel):
    id: str
    object: str = "model"


class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_memory_used: Optional[int] = None
    gpu_memory_total: Optional[int] = None


_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    return _engine


def create_engine(model_name: str):
    global _engine
    if _engine is not None:
        return
    _engine = Engine(model_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_engine("meta-llama/Llama-3.2-1B-Instruct")
    yield # before yield run everything, after yield is shutdown part
    global _engine
    _engine = None


app = FastAPI(title="tachyon", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_available = torch.cuda.is_available()
    gpu_memory_used = None
    gpu_memory_total = None

    if gpu_available:
        GB = 1024 ** 3

        gpu_memory_used = torch.cuda.memory_allocated() / GB
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / GB

    return HealthResponse(
        status="healthy",
        gpu_available=gpu_available,
        gpu_memory_used=gpu_memory_used,
        gpu_memory_total=gpu_memory_total,
    )


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(
        data=[
            Model(id="llama-3.2-1b-instruct"),
        ]
    )


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    return "\n".join([f"{m.role}: {m.content}" for m in messages])


async def generate_stream(request: EngineRequest, engine: Engine):
    while not request.is_completed:
        engine.current_batch = [request]
        if request.is_prefill:
            engine.prefill_batch([request])
        else:
            engine.decode_batch([request])
        yield f"data: {request.tokens[-1]}\n\n"
        await asyncio.sleep(0)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(
            generate_stream_chat(request),
            media_type="text/event-stream",
        )

    engine = get_engine()
    prompt = messages_to_prompt(request.messages)
    engine_request = EngineRequest(
        id=int(str(uuid.uuid4().int)[:8]),
        prompt=prompt,
        max_tokens=request.max_tokens or 100,
        temperature=request.temperature or 0.1,
    )

    result = await asyncio.to_thread(
        engine.generate_text,
        engine_request.prompt,
        engine_request.max_tokens,
        engine_request.temperature,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result.response or ""),
                finish_reason="stop" if result.is_completed else "length",
            )
        ],
        usage=Usage(
            prompt_tokens=len(result.prompt_tokens),
            completion_tokens=len(result.tokens),
            total_tokens=len(result.prompt_tokens) + len(result.tokens),
        ),
    )


async def generate_stream_chat(request: ChatCompletionRequest):
    engine = get_engine()
    prompt = messages_to_prompt(request.messages)

    engine_request = EngineRequest(
        id=int(str(uuid.uuid4().int)[:8]),
        prompt=prompt,
        max_tokens=request.max_tokens or 100,
        temperature=request.temperature or 0.1,
    )

    async for token in generate_stream(engine_request, engine):
        yield token


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    if request.stream:
        return StreamingResponse(
            generate_stream_completion(request),
            media_type="text/event-stream",
        )

    engine = get_engine()

    result = await asyncio.to_thread(
        engine.generate_text,
        request.prompt,
        request.max_tokens or 100,
        request.temperature or 0.1,
    )

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=[
            CompletionChoice(
                index=0,
                text=result.response or "",
                finish_reason="stop" if result.is_completed else "length",
            )
        ],
        usage=Usage(
            prompt_tokens=len(result.prompt_tokens),
            completion_tokens=len(result.tokens),
            total_tokens=len(result.prompt_tokens) + len(result.tokens),
        ),
    )


async def generate_stream_completion(request: CompletionRequest):
    engine = get_engine()

    engine_request = EngineRequest(
        id=int(str(uuid.uuid4().int)[:8]),
        prompt=request.prompt,
        max_tokens=request.max_tokens or 100,
        temperature=request.temperature or 0.1,
    )

    async for token in generate_stream(engine_request, engine):
        yield token


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
