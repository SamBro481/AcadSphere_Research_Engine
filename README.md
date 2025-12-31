# AI Research Paper Search Engine

A modern, AI-powered research paper search interface built with Next.js and powered by FAISS semantic search.

## Features

- **Semantic Search**: Uses SciBERT embeddings for intelligent paper discovery
- **Context-Aware**: Remembers your search history to refine results
- **Clean Interface**: Modern, dark-themed UI optimized for research
- **Fast**: FAISS-powered vector search for instant results

## Setup Instructions

### 1. Install Dependencies

**Frontend (Next.js):**
```bash
npm install
```

**Backend (Python):**
```bash
cd scripts
pip install fastapi uvicorn faiss-cpu numpy torch transformers openai soundfile librosa tqdm pydantic
```

### 2. Prepare Your Data

Place your files in the project root:
- `embeddings/paper_embeddings.npy` - Pre-computed SciBERT embeddings
- `embeddings/paper_metadata.json` - Paper metadata (title, abstract, authors, etc.)

If you don't have embeddings yet, run:
```bash
python scripts/scibert_encoder.py
```

Make sure you have a `data/papers.jsonl` file with your papers in JSONL format.

### 3. Set Environment Variables

Create a `.env.local` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
FAISS_BACKEND_URL=http://localhost:8000
```

### 4. Start the Backend

In one terminal:
```bash
cd scripts
python search_api.py
```

The FastAPI backend will start on `http://localhost:8000`

### 5. Start the Frontend

In another terminal:
```bash
npm run dev
```

The Next.js app will start on `http://localhost:3000`

## Project Structure

```
/app
  /api/search/route.ts    # Next.js API route that calls FastAPI
  page.tsx                # Main search page
/components
  search-interface.tsx    # Search UI component
/scripts
  search_api.py          # FastAPI server for FAISS search
  faiss_search.py        # Core FAISS search logic
  scibert_encoder.py     # Generate embeddings from papers
  context.py             # Context-aware search manager
  summarize.py           # GPT-powered paper summaries
  wav2vec2_stt.py        # Voice search (optional)
/embeddings
  paper_embeddings.npy   # Your paper embeddings
  paper_metadata.json    # Your paper metadata
```

## Usage

1. Open http://localhost:3000
2. Enter your research query
3. View semantically relevant papers ranked by relevance
4. Click on papers to explore details

## API Endpoints

**Backend (FastAPI - Port 8000):**
- `POST /search` - Search for papers
  ```json
  {
    "query": "transformer attention mechanisms",
    "k": 10,
    "summarize": false
  }
  ```

**Frontend (Next.js - Port 3000):**
- `POST /api/search` - Proxies requests to FastAPI backend

## Technologies

- **Frontend**: Next.js 16, React 19, TailwindCSS, shadcn/ui
- **Backend**: FastAPI, FAISS, PyTorch, Transformers (SciBERT)
- **AI**: OpenAI GPT-4 for summaries, SciBERT for embeddings
