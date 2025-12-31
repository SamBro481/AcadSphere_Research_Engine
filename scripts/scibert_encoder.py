import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_FILE = "data/papers.jsonl"
OUTPUT_EMBEDDINGS = "embeddings/paper_embeddings.npy"
OUTPUT_META = "embeddings/paper_metadata.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


embeddings = []
metadata = []

with open(DATA_FILE, "r", encoding="utf-8") as f:
    papers = list(f)

for line in tqdm(papers, desc="Encoding papers"):
    paper = json.loads(line)

    text = paper["title"] + " " + paper["abstract"]

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
    embeddings.append(emb.cpu().numpy()[0])
    metadata.append(paper)

embeddings = np.array(embeddings)

np.save(OUTPUT_EMBEDDINGS, embeddings)

with open(OUTPUT_META, "w", encoding="utf-8") as f:
    json.dump(metadata, f)

print("Saved embeddings:", embeddings.shape)
