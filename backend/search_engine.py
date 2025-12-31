import faiss
import numpy as np
import json
import sys
import pickle
from pathlib import Path

class ResearchPaperSearchEngine:
    """
    FAISS-based research paper search engine.
    Replace this with your actual implementation.
    """
    def __init__(self, index_path: str = "backend/faiss_index", metadata_path: str = "backend/papers_metadata.pkl"):
        """
        Initialize the search engine with your FAISS index and metadata.
        
        Args:
            index_path: Path to your FAISS index file
            metadata_path: Path to your papers metadata (pickle or json)
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index = None
        self.papers_metadata = []
        self.embedding_model = None
        
        # Load index and metadata if they exist
        if self.index_path.exists():
            self.load_index()
        if self.metadata_path.exists():
            self.load_metadata()
    
    def load_index(self):
        """Load the FAISS index from disk"""
        try:
            self.index = faiss.read_index(str(self.index_path))
            print(f"[v0] Loaded FAISS index with {self.index.ntotal} vectors", file=sys.stderr)
        except Exception as e:
            print(f"[v0] Error loading FAISS index: {e}", file=sys.stderr)
    
    def load_metadata(self):
        """Load paper metadata from disk"""
        try:
            with open(self.metadata_path, 'rb') as f:
                self.papers_metadata = pickle.load(f)
            print(f"[v0] Loaded metadata for {len(self.papers_metadata)} papers", file=sys.stderr)
        except Exception as e:
            print(f"[v0] Error loading metadata: {e}", file=sys.stderr)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode the query text into embeddings.
        Replace this with your actual encoding logic (e.g., using sentence-transformers).
        """
        # Example using sentence-transformers (uncomment and modify):
        # from sentence_transformers import SentenceTransformer
        # if self.embedding_model is None:
        #     self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # embedding = self.embedding_model.encode([query])
        # return np.array(embedding).astype('float32')
        
        # Placeholder: Random embedding (replace with your actual implementation)
        return np.random.rand(1, 384).astype('float32')
    
    def search(self, query: str, top_k: int = 10):
        """
        Search for papers matching the query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of paper results with metadata and relevance scores
        """
        if self.index is None or len(self.papers_metadata) == 0:
            print("[v0] Index or metadata not loaded, returning empty results", file=sys.stderr)
            return []
        
        try:
            # Encode query
            query_embedding = self.encode_query(query)
            
            # Search FAISS index
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for idx, (distance, paper_idx) in enumerate(zip(distances[0], indices[0])):
                if paper_idx < len(self.papers_metadata):
                    paper = self.papers_metadata[paper_idx]
                    
                    # Convert distance to similarity score (0-1 range)
                    # Adjust this based on your distance metric
                    relevance_score = 1 / (1 + distance)
                    
                    results.append({
                        "id": str(paper_idx),
                        "title": paper.get("title", "Unknown Title"),
                        "authors": paper.get("authors", []),
                        "abstract": paper.get("abstract", ""),
                        "year": paper.get("year", None),
                        "venue": paper.get("venue", ""),
                        "relevanceScore": float(relevance_score),
                        "url": paper.get("url", None),
                        "doi": paper.get("doi", None),
                    })
            
            return results
            
        except Exception as e:
            print(f"[v0] Search error: {e}", file=sys.stderr)
            return []

def main():
    """
    Main function to handle search requests from Node.js
    Expects query as first command line argument
    """
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No query provided"}))
        sys.exit(1)
    
    query = sys.argv[1]
    
    # Initialize search engine
    search_engine = ResearchPaperSearchEngine()
    
    # Perform search
    results = search_engine.search(query)
    
    # Output results as JSON
    print(json.dumps({"results": results}))

if __name__ == "__main__":
    main()
