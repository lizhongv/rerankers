from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import os
import time

app = FastAPI(
    title="BGE-Reranker CPU Service",
    description="API for document reranking using BGE-Reranker model"
)

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = CrossEncoder('./bge-reranker-base', device='cpu')
MODEL_NAME="bge-reranker-base"

# model = CrossEncoder('./bge-reranker-v2-m3', device='cpu')
# MODEL_NAME="bge-reranker-v2-m3"

# Request/Response models
class QueryDocumentPair(BaseModel):
    query: str
    document: str

class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = None # Optional top-k filtering
    batch_size: int = 16  # Default batch size for processing

@app.post("/rerank")
async def rerank_texts(request: RerankRequest):
    """Rerank documents based on their relevance to the query"""
    start_time = time.time()

    # Safety limits
    MAX_DOCS = 100
    if len(request.documents) > MAX_DOCS:
        raise HTTPException(
            status_code=400,
            detail=f"Exceeded maximum document limit ({MAX_DOCS}). Please reduce batch size."
        )

    # Prepare model inputs
    model_inputs = [[request.query, doc] for doc in request.documents]

    # Process in batches to avoid memory issues
    scores = []
    for i in range(0, len(model_inputs), request.batch_size):
        batch = model_inputs[i:i + request.batch_size]
        scores.extend(model.predict(batch))

    # Combine and sort results
    results = sorted(
        zip(request.documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Apply top_k filtering if specified
    if request.top_k is not None and request.top_k > 0:
        results = results[:request.top_k]

    processing_time = time.time() - start_time

    return {
        "model": MODEL_NAME,
        "device": "cpu",
        "processing_time_seconds": round(processing_time, 3),
        "documents_processed": len(request.documents),
        "results": [
            {"document": doc, "score": float(score), "rank": idx+1}
            for idx, (doc, score) in enumerate(results)
        ]
    }

@app.get("/model-info")
async def get_model_info():
    """Return model metadata and configuration"""
    return {
        "model_name": MODEL_NAME,  
        "max_sequence_length": 512,
        "recommended_batch_size": 16,
        "device": "cpu"
    }

@app.get("/health")
async def health_check():
    """Service health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

