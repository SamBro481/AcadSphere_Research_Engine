# FAISS Backend Setup

This backend connects your existing FAISS search implementation to the Next.js frontend.

## Setup Instructions

### 1. Replace the search_engine.py with your implementation

The current `search_engine.py` is a template. You need to:

- Replace the `encode_query()` method with your actual text embedding logic
- Update the file paths in `__init__()` to point to your FAISS index and metadata
- Adjust the `search()` method if your metadata structure is different

### 2. Install Python dependencies

```bash
pip install faiss-cpu numpy sentence-transformers
# Or if you have GPU support:
# pip install faiss-gpu numpy sentence-transformers
```

### 3. Add your FAISS index and metadata

Place your files in the `backend/` directory:
- `faiss_index` - Your FAISS index file
- `papers_metadata.pkl` - Your papers metadata (pickle format)

Or update the paths in `search_engine.py` to point to your existing files.

### 4. Test the backend independently

```bash
python3 backend/search_engine.py "your test query"
```

This should output JSON with search results.

### 5. Required metadata format

Your papers metadata should be a list of dictionaries with these fields:
```python
{
    "title": str,
    "authors": List[str],
    "abstract": str,
    "year": int,
    "venue": str,
    "url": str (optional),
    "doi": str (optional),
}
```

## Troubleshooting

- Make sure Python 3 is installed and accessible via `python3` command
- Check that all dependencies are installed
- Verify your FAISS index and metadata files are in the correct location
- Check the terminal/console logs for `[v0]` debug messages
