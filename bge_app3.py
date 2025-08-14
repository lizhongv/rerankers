from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from typing import Optional
import os
import time

app = FastAPI(
    title="BGE-Reranker CPU Service",
    description="API for document reranking using BGE-Reranker models"
)

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load both models at startup
MODEL_PATHS = {
    "large": "./bge-reranker-large",
    "m3": "./bge-reranker-v2-m3"
}

models = {
    "large": CrossEncoder(MODEL_PATHS["large"], device='cpu'),
    "m3": CrossEncoder(MODEL_PATHS["m3"], device='cpu')
}

# Request/Response models
class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: Optional[int] = None  # Optional top-k filtering
    batch_size: int = 16  # Default batch size for processing

def perform_reranking(model, request: RerankRequest):
    """Common reranking logic for both endpoints"""
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
    original_indices = list(range(len(request.documents)))  # [0, 1, 2, ...]

    # Process in batches to avoid memory issues
    scores = []
    for i in range(0, len(model_inputs), request.batch_size):
        batch = model_inputs[i:i + request.batch_size]
        scores.extend(model.predict(batch))

    # Combine results with original indices
    indexed_results = list(zip(request.documents, scores, original_indices))

    # Sort by score in descending order
    sorted_results = sorted(
        indexed_results,
        key=lambda x: x[1],  
        reverse=True
    )

    # Apply top_k filtering if specified
    if request.top_k is not None and request.top_k > 0:
        sorted_results = sorted_results[:request.top_k]

    processing_time = time.time() - start_time

    return {
        "processing_time_seconds": round(processing_time, 3),
        "documents_processed": len(request.documents),
        "query": request.query,
        "results": [
            {
                "index": original_index,  # Original position in input list
                "document": doc, 
                "score": float(score), 
                "rank": idx+1  # New position after ranking
            }
            for idx, (doc, score, original_index) in enumerate(sorted_results)
        ]
    }

@app.post("/rerank/large")
async def rerank_large(request: RerankRequest):
    """Rerank documents using the large model"""
    result = perform_reranking(models["large"], request)
    return {"model": "bge-reranker-large", "device": "cpu", **result}

@app.post("/rerank/m3")
async def rerank_m3(request: RerankRequest):
    """Rerank documents using the m3 model"""
    result = perform_reranking(models["m3"], request)
    return {"model": "bge-reranker-v2-m3", "device": "cpu", **result}

@app.get("/model-info")
async def get_model_info():
    """Return model metadata and configuration"""
    return {
        "available_models": {
            "large": {
                "model_name": "bge-reranker-large",
                "max_sequence_length": 512,
                "recommended_batch_size": 16,
                "device": "cpu",
                "endpoint": "/rerank/large"
            },
            "m3": {
                "model_name": "bge-reranker-v2-m3",
                "max_sequence_length": 512,
                "recommended_batch_size": 16,
                "device": "cpu",
                "endpoint": "/rerank/m3"
            }
        }
    }

@app.get("/health")
async def health_check():
    """Service health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys())
    }