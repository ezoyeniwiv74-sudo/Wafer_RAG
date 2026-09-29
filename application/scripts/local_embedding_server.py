"""Small OpenAI-compatible Qwen3 embedding service for local CPU demos.

It uses only packages already present in ``wafer_search_project_env`` and
exposes the endpoint expected by ``utils.embedding.EmbeddingGenerator``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Local Qwen3 Embedding Service")
_model = None


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "qwen3-embedding"


def model():
    global _model
    if _model is None:
        from utils.embedding11 import LocalEmbeddingGenerator
        _model = LocalEmbeddingGenerator()
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "device": "cpu"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "qwen3-embedding", "object": "model"}]}


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest):
    texts = request.input if isinstance(request.input, list) else [request.input]
    vectors = model().get_embeddings(texts)
    return {
        "object": "list",
        "model": request.model,
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8333, workers=1)
