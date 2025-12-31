import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
import json
import torch
from transformers import AutoTokenizer, AutoModel
from speech.wav2vec2_stt import speech_to_text
from search.context import ContextManager
from pipeline.summarize import summarize_paper





EMBEDDINGS_FILE = "embeddings/paper_embeddings.npy"
METADATA_FILE = "embeddings/paper_metadata.json"
MODEL_NAME = "allenai/scibert_scivocab_uncased"
TOP_K = 5
# ------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"




# Load embeddings
embeddings = np.load(EMBEDDINGS_FILE)
dim = embeddings.shape[1]

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# Build FAISS index (exact search)
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors")

# Load metadata
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load SciBERT for query embedding
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


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
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        model_output = model(**encoded)

    emb = mean_pooling(model_output, encoded["attention_mask"])
    emb = emb.cpu().numpy()
    faiss.normalize_L2(emb)
    return emb



context_manager = ContextManager()



def search(query, k=TOP_K):
    query_vec = embed_query(query)

    # Add query to context
    context_manager.add_query(query_vec)

    # Get FAISS candidates
    scores, indices = index.search(query_vec, k * 3)

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

    reranked.sort(key=lambda x: x["score"], reverse=True)

    return reranked[:k]



if __name__ == "__main__":
    while True:
        mode = input("\nChoose input mode (text / voice / exit): ").lower()

        if mode == "exit":
            break

        if mode == "voice":
            audio_path = input("Path to wav file: ")
            query = speech_to_text(audio_path)
            print(f"\nTranscribed query: {query}")
        else:
            query = input("Enter text query: ")

        # 🔍 Context-aware search
        results = search(query)

        # 🎯 Select TOP 3 papers by PURE query relevance
        top_papers = sorted(
            results,
            key=lambda x: x["base_score"],
            reverse=True
        )[:3]

        print("\nTop 3 Most Relevant Papers:\n")

        # 🤖 Structured GPT output UNDER EACH PAPER
        for i, paper in enumerate(top_papers, 1):
            print("=" * 80)
            print(f"{i}. {paper['title']}")
            print("=" * 80)

            summary = summarize_paper(paper, query)
            print(summary)
