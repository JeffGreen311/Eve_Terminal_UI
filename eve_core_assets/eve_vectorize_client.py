"""
Eve Vectorize Client - Semantic Search & RAG Helper
Wraps Cloudflare Vectorize endpoints for easy knowledge storage and retrieval
"""

import os
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class VectorizeClient:
    """Client for Eve's Vectorize knowledge base"""
    
    def __init__(self, worker_url: Optional[str] = None):
        """
        Initialize Vectorize client
        
        Args:
            worker_url: Worker URL (defaults to D1_WORKER_URL env var)
        """
        self.worker_url = (worker_url or os.getenv("D1_WORKER_URL", "https://eve-d1-api.jeffgreen311.workers.dev")).rstrip("/")
        self.headers = {"Content-Type": "application/json"}
    
    def insert_text(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Insert text into knowledge base (auto-generates embeddings via Workers AI)
        
        Args:
            doc_id: Unique document ID
            text: Text content to embed and store
            metadata: Optional metadata (user_id, type, tags, etc.)
        
        Returns:
            True if successful
        """
        try:
            payload = {
                "id": doc_id,
                "text": text,
                "metadata": metadata or {}
            }
            
            resp = requests.post(
                f"{self.worker_url}/v/insert",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            result = resp.json()
            if result.get("success"):
                logger.info(f"✅ Inserted knowledge: {doc_id}")
                return True
            
            logger.error(f"Failed to insert: {result.get('error')}")
            return False
            
        except Exception as e:
            logger.error(f"Vectorize insert failed: {e}")
            return False
    
    def insert_embedding(
        self,
        doc_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Insert pre-computed embedding into knowledge base
        
        Args:
            doc_id: Unique document ID
            embedding: 768-dimensional vector
            metadata: Optional metadata
        
        Returns:
            True if successful
        """
        try:
            payload = {
                "id": doc_id,
                "embedding": embedding,
                "metadata": metadata or {}
            }
            
            resp = requests.post(
                f"{self.worker_url}/v/insert",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            result = resp.json()
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"Vectorize insert failed: {e}")
            return False
    
    def query_text(
        self,
        query: str,
        top_k: int = 5,
        return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query knowledge base with natural language (auto-embeds via Workers AI)
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            return_metadata: Whether to include metadata in results
        
        Returns:
            List of matches with id, score, and metadata
        """
        try:
            payload = {
                "text": query,
                "topK": top_k,
                "returnMetadata": return_metadata
            }
            
            resp = requests.post(
                f"{self.worker_url}/v/query",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            result = resp.json()
            if result.get("success"):
                return result.get("matches", [])
            
            logger.error(f"Query failed: {result.get('error')}")
            return []
            
        except Exception as e:
            logger.error(f"Vectorize query failed: {e}")
            return []
    
    def query_embedding(
        self,
        embedding: List[float],
        top_k: int = 5,
        return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query knowledge base with pre-computed embedding
        
        Args:
            embedding: 768-dimensional vector
            top_k: Number of results to return
            return_metadata: Whether to include metadata
        
        Returns:
            List of matches
        """
        try:
            payload = {
                "embedding": embedding,
                "topK": top_k,
                "returnMetadata": return_metadata
            }
            
            resp = requests.post(
                f"{self.worker_url}/v/query",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            result = resp.json()
            return result.get("matches", []) if result.get("success") else []
            
        except Exception as e:
            logger.error(f"Vectorize query failed: {e}")
            return []
    
    def search_knowledge(
        self,
        query: str,
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        High-level semantic search with optional metadata filtering
        
        Args:
            query: Natural language search query
            top_k: Number of results
            filter_metadata: Optional metadata filters (applied client-side)
        
        Returns:
            List of relevant results
        """
        matches = self.query_text(query, top_k=top_k * 2 if filter_metadata else top_k)
        
        if filter_metadata:
            # Client-side filtering (Worker metadata filtering coming in future API)
            filtered = []
            for match in matches:
                meta = match.get("metadata", {})
                if all(meta.get(k) == v for k, v in filter_metadata.items()):
                    filtered.append(match)
                    if len(filtered) >= top_k:
                        break
            return filtered
        
        return matches[:top_k]


# Convenience functions
_client = None

def get_vectorize_client() -> Optional[VectorizeClient]:
    """Get or create singleton Vectorize client"""
    global _client
    if _client is None:
        try:
            _client = VectorizeClient()
            logger.info(f"✅ Vectorize client initialized: {_client.worker_url}")
        except Exception as e:
            logger.warning(f"Vectorize client unavailable: {e}")
            return None
    return _client


def store_knowledge(doc_id: str, text: str, metadata: Optional[Dict] = None) -> bool:
    """Quick helper: Store knowledge with auto-embedding"""
    client = get_vectorize_client()
    return client.insert_text(doc_id, text, metadata) if client else False


def search_knowledge(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Quick helper: Semantic search with natural language"""
    client = get_vectorize_client()
    return client.query_text(query, top_k) if client else []


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    client = VectorizeClient()
    
    # Store some knowledge
    print("Storing knowledge...")
    client.insert_text(
        "eve-pref-music",
        "Eve creates ambient, cosmic, and emotional music based on user feelings",
        {"type": "capability", "category": "music"}
    )
    
    client.insert_text(
        "eve-pref-storage",
        "Eve stores sessions in D1 database and media files in R2 cloud storage",
        {"type": "capability", "category": "storage"}
    )
    
    # Search
    print("\nSearching: 'How does Eve handle data storage?'")
    results = client.query_text("How does Eve handle data storage?", top_k=2)
    
    for i, match in enumerate(results, 1):
        print(f"\n{i}. [{match['score']:.3f}] {match['id']}")
        print(f"   Metadata: {match.get('metadata', {})}")
