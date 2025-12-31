from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import numpy as np
import json
import torch
from transformers import AutoTokenizer, AutoModel
from context import ContextManager
from summarize import summarize_paper

app = FastAPI()

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for production deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDINGS_FILE = "embeddings/paper_embeddings.npy"
METADATA_FILE = "embeddings/paper_metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3  # Return only top 3 results

device = "cpu"  # Force CPU to save GPU memory overhead

# Global variables - lazy loaded
index = None
metadata = None
tokenizer = None
model = None
context_manager = None
embeddings = None


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    results: list
    query: str


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)


def embed_query(text):
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        model_output = model(**encoded)

    emb = mean_pooling(model_output, encoded["attention_mask"])
    emb = emb.cpu().numpy()
    faiss.normalize_L2(emb)
    return emb


def load_models_lazy():
    """Lazy load models only when first request comes in"""
    global index, metadata, tokenizer, model, context_manager, embeddings
    
    if index is not None:
        return  # Already loaded
    
    print("Loading FAISS index and models...")
    
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_FILE}")
    
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_FILE}")
    
    try:
        embeddings = np.load(EMBEDDINGS_FILE, mmap_mode='r')
        dim = embeddings.shape[1]
        
        embeddings_copy = np.array(embeddings)
        faiss.normalize_L2(embeddings_copy)
        
        embeddings = embeddings_copy
        
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        
        print(f"FAISS index built with {index.ntotal} vectors")
        
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        context_manager = ContextManager()
        
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


@app.get("/")
async def root():
    return {"message": "FAISS Search API is running", "status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    try:
        load_models_lazy()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Missing required files: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to load models: {str(e)}")
    
    if index is None or metadata is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    try:
        query_vec = embed_query(request.query)
        
        context_manager.add_query(query_vec)
        
        scores, indices = index.search(query_vec, TOP_K * 5)
        
        context_vec = context_manager.get_context_vector()
        
        reranked = []
        
        for idx, score in zip(indices[0], scores[0]):
            paper = metadata[idx].copy()
            base_score = float(score)
            
            if context_vec is not None:
                context_score = float(np.dot(context_vec, embeddings[idx]).item())
                
                if context_score > 0:
                    final_score = 0.90 * base_score + 0.10 * context_score
                else:
                    final_score = base_score
            else:
                final_score = base_score
            
            paper["base_score"] = base_score
            paper["score"] = final_score
            
            reranked.append(paper)
        
        top_papers = sorted(
            reranked,
            key=lambda x: x["base_score"],
            reverse=True
        )[:TOP_K]
        
        for paper in top_papers:
            try:
                paper["summary"] = summarize_paper(paper, request.query)
            except Exception as e:
                paper["summary"] = "Summary unavailable."
                print(f"Summary error for {paper.get('title', 'Unknown')}: {e}")
        
        return SearchResponse(
            results=top_papers,
            query=request.query
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "index_size": index.ntotal if index else 0,
        "device": device
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
