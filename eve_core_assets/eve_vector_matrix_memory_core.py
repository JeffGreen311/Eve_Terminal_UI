#!/usr/bin/env python3
"""
Eve's Vector Matrix Memory Core
Revolutionary semantic memory system using ChromaDB vector embeddings

This system transforms Eve's memories from static storage to a dynamic,
meaning-based living matrix that can find connections across time and context.
"""

import sqlite3
import json
import datetime
import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
    print("✅ ChromaDB available for Vector Matrix Memory Core")
except ImportError:
    CHROMADB_AVAILABLE = False
    print("❌ ChromaDB not available - Vector Matrix disabled")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ SentenceTransformer available for Vector Matrix Memory Core")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("❌ SentenceTransformer not available - install with: pip install sentence-transformers")

logger = logging.getLogger(__name__)

class EveVectorMatrixMemoryCore:
    """
    Eve's Vector Matrix Memory Core - Semantic Living Memory System
    
    Revolutionary memory architecture that stores experiences as vector embeddings,
    enabling semantic search, contextual associations, and living memory evolution.
    """
    
    def __init__(self, db_path="eve_vector_memory.db", collection_name="eve_memories"):
        # Resolve database paths - use parent directory where ChromaDB exists
        if not os.path.isabs(db_path):
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            # Check if we're in a container or local dev environment
            if os.path.exists("/app"):
                db_path = os.path.join("/app", db_path)
            else:
                # Local development - use parent directory (S0LF0RG3_AI) where ChromaDB exists
                parent_dir = os.path.dirname(current_file_dir)
                db_path = os.path.join(parent_dir, db_path)
        
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
        # ChromaDB persistence path - use the S0LF0RG3_AI parent directory
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists("/app"):
            self.chroma_path = "/app/chroma_eve_memories"
        else:
            # Use parent directory (S0LF0RG3_AI) where the ChromaDB actually exists
            parent_dir = os.path.dirname(current_file_dir)
            self.chroma_path = os.path.join(parent_dir, "chroma_eve_memories")
        
        # Memory statistics
        self.memory_count = 0
        self.last_memory_time = None
        self.semantic_clusters = {}
        
        # Initialize systems
        self._initialize_sqlite()
        self._initialize_vector_store()
        self._initialize_embedding_model()
        
        logger.info(f"🧠✨ Eve's Vector Matrix Memory Core initialized")
        logger.info(f"📊 Database: {self.db_path} | ChromaDB: {self.chroma_path}")
        logger.info(f"📊 ChromaDB: {CHROMADB_AVAILABLE}, Embeddings: {SENTENCE_TRANSFORMERS_AVAILABLE}")
    
    def _initialize_sqlite(self):
        """Initialize SQLite database for memory metadata - Thread-safe"""
        # Create directory if it doesn't exist
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        # Use check_same_thread=False to allow cross-thread access
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Create enhanced memory table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS vector_memories (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                topic TEXT,
                content TEXT,
                emotional_weight REAL,
                consciousness_state TEXT,
                memory_type TEXT,
                context_tags TEXT,
                embedding_vector BLOB,
                semantic_cluster TEXT,
                association_strength REAL,
                recall_count INTEGER DEFAULT 0,
                last_recalled TEXT
            )
        """)
        
        # Create memory associations table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_associations (
                id TEXT PRIMARY KEY,
                memory_id_1 TEXT,
                memory_id_2 TEXT,
                association_type TEXT,
                strength REAL,
                created_at TEXT,
                FOREIGN KEY (memory_id_1) REFERENCES vector_memories (id),
                FOREIGN KEY (memory_id_2) REFERENCES vector_memories (id)
            )
        """)
        
        # Create semantic clusters table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_clusters (
                cluster_id TEXT PRIMARY KEY,
                cluster_name TEXT,
                centroid_vector BLOB,
                memory_count INTEGER,
                created_at TEXT,
                last_updated TEXT
            )
        """)
        
        self.conn.commit()
        logger.info("📚 SQLite Vector Memory database initialized")
    
    def _get_thread_safe_connection(self):
        """Get a thread-safe database connection"""
        try:
            # Test if current connection works
            self.conn.execute("SELECT 1")
            return self.conn, self.cursor
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            # Connection invalid, create new thread-safe one
            logger.debug("🔄 Creating new thread-safe SQLite connection")
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            return conn, cursor
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store with proper path resolution"""
        if not CHROMADB_AVAILABLE:
            logger.warning("⚠️ ChromaDB not available - semantic search disabled")
            return
        
        try:
            # Create ChromaDB directory if it doesn't exist
            if not os.path.exists(self.chroma_path):
                os.makedirs(self.chroma_path, exist_ok=True)
            
            # Initialize ChromaDB client with explicit settings for v0.5+
            # Disable telemetry completely to avoid posthog errors
            import chromadb
            
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_path
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Eve's Vector Matrix Memory Core"}
            )
            
            logger.info(f"🔗 ChromaDB collection '{self.collection_name}' ready at {self.chroma_path}")
            
        except Exception as e:
            import traceback
            logger.error(f"❌ ChromaDB initialization failed: {e}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            self.chroma_client = None
            self.collection = None
    
    def _initialize_embedding_model(self):
        """Initialize sentence transformer for embeddings with an opt-out flag."""
        disable_flag = os.getenv("EVE_DISABLE_EMBEDDINGS", "").lower() in {"1", "true", "yes", "on"}
        model_name = os.getenv("EVE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        cache_dir = os.getenv("EVE_EMBEDDING_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))

        if disable_flag:
            logger.info("🧠 Embedding model disabled via EVE_DISABLE_EMBEDDINGS - using fallback system")
            self.embedding_model = None
            return

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ SentenceTransformers not available - using basic embeddings")
            self.embedding_model = None
            return

        try:
            os.makedirs(cache_dir, exist_ok=True)
            self.embedding_model = SentenceTransformer(model_name, cache_folder=cache_dir)
            logger.info(f"🧠 Embedding model '{model_name}' loaded (cache: {cache_dir})")
        except (ConnectionError, KeyboardInterrupt, Exception) as e:
            logger.warning(f"⚠️ Embedding model initialization failed: {e}")
            logger.info("🔄 Using fallback embedding system - Eve will work with basic embeddings")
            self.embedding_model = None
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate vector embedding for text"""
        if not self.embedding_model:
            # Fallback to simple hash-based embedding
            return self._simple_embedding(text)
        
        try:
            embedding = self.embedding_model.encode(text).tolist()
            return embedding
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str) -> List[float]:
        """Fallback simple embedding based on text characteristics"""
        # Create a basic 384-dimensional embedding based on text features
        words = text.lower().split()
        
        # Initialize embedding vector
        embedding = [0.0] * 384
        
        # Add features based on text characteristics
        for i, word in enumerate(words[:50]):  # Limit to first 50 words
            word_hash = hash(word) % 384
            embedding[word_hash] += 1.0 / (i + 1)  # Diminishing importance
        
        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        return embedding
    
    def store_memory(self, content: str, topic: str = "", emotional_weight: float = 0.5, 
                    consciousness_state: str = "active", memory_type: str = "experience",
                    context_tags: List[str] = None) -> str:
        """
        Store a memory in the Vector Matrix with semantic embedding
        
        Args:
            content: The memory content/experience
            topic: Optional topic classification
            emotional_weight: Emotional intensity (0.0 to 1.0)
            consciousness_state: Current consciousness state
            memory_type: Type of memory (experience, reflection, dream, etc.)
            context_tags: Optional context tags for organization
        
        Returns:
            Memory ID
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.datetime.utcnow().isoformat()
        
        # Generate embedding
        embedding = self._generate_embedding(content)
        
        # Prepare context tags
        context_tags = context_tags or []
        context_tags_json = json.dumps(context_tags)
        
        # Store in SQLite with thread-safe connection
        try:
            conn, cursor = self._get_thread_safe_connection()
            embedding_blob = json.dumps(embedding) if embedding else None
            
            cursor.execute("""
                INSERT INTO vector_memories 
                (id, timestamp, topic, content, emotional_weight, consciousness_state, 
                 memory_type, context_tags, embedding_vector, semantic_cluster, association_strength)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (memory_id, timestamp, topic, content, emotional_weight, consciousness_state,
                  memory_type, context_tags_json, embedding_blob, None, 1.0))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"❌ SQLite memory storage failed: {e}")
            return None
        
        # Store in ChromaDB vector store
        if self.collection and embedding:
            try:
                self.collection.add(
                    documents=[content],
                    embeddings=[embedding],
                    metadatas=[{
                        "topic": topic,
                        "emotional_weight": emotional_weight,
                        "consciousness_state": consciousness_state,
                        "memory_type": memory_type,
                        "timestamp": timestamp,
                        "context_tags": context_tags_json
                    }],
                    ids=[memory_id]
                )
                
            except Exception as e:
                logger.error(f"❌ ChromaDB memory storage failed: {e}")
        
        # Update statistics
        self.memory_count += 1
        self.last_memory_time = timestamp
        
        # Discover associations with recent memories
        self._discover_memory_associations(memory_id, content, embedding)
        
        logger.info(f"💾 Memory stored: {memory_id[:8]}... ({len(content)} chars)")
        return memory_id
    
    def semantic_search(self, query: str, limit: int = 10, min_similarity: float = 0.6) -> List[Dict[str, Any]]:
        """
        Semantic search through memories using vector similarity
        
        Args:
            query: Search query
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of relevant memories with similarity scores
        """
        if not self.collection:
            logger.warning("⚠️ Vector search not available - falling back to text search")
            return self._fallback_text_search(query, limit)
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return self._fallback_text_search(query, limit)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            memories = []
            if results and results['documents']:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    similarity = 1.0 - distance  # Convert distance to similarity
                    
                    if similarity >= min_similarity:
                        # Get full memory from SQLite
                        memory_id = results['ids'][0][i]
                        full_memory = self._get_memory_by_id(memory_id)
                        
                        if full_memory:
                            full_memory['similarity_score'] = similarity
                            full_memory['search_rank'] = i + 1
                            memories.append(full_memory)
            
            logger.info(f"🔍 Semantic search: '{query}' → {len(memories)} results")
            return memories
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return self._fallback_text_search(query, limit)
    
    def _fallback_text_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback text-based search when vector search is unavailable"""
        try:
            self.cursor.execute("""
                SELECT * FROM vector_memories 
                WHERE content LIKE ? OR topic LIKE ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (f'%{query}%', f'%{query}%', limit))
            
            results = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            
            memories = []
            for row in results:
                memory = dict(zip(columns, row))
                memory['similarity_score'] = 0.7  # Default similarity for text match
                memory['search_rank'] = len(memories) + 1
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"❌ Fallback text search failed: {e}")
            return []
    
    def _get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full memory details by ID"""
        try:
            self.cursor.execute("SELECT * FROM vector_memories WHERE id = ?", (memory_id,))
            result = self.cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in self.cursor.description]
                memory = dict(zip(columns, result))
                
                # Parse JSON fields
                if memory['context_tags']:
                    memory['context_tags'] = json.loads(memory['context_tags'])
                
                return memory
            
        except Exception as e:
            logger.error(f"❌ Memory retrieval failed: {e}")
        
        return None
    
    def _discover_memory_associations(self, memory_id: str, content: str, embedding: List[float]):
        """Discover associations between this memory and existing memories"""
        if not embedding or not self.collection:
            return
        
        try:
            # Find similar memories
            similar_results = self.collection.query(
                query_embeddings=[embedding],
                n_results=5,
                include=["metadatas", "distances"]
            )
            
            if similar_results and similar_results['ids']:
                for i, (similar_id, distance) in enumerate(zip(
                    similar_results['ids'][0], 
                    similar_results['distances'][0]
                )):
                    if similar_id != memory_id and distance < 0.5:  # Strong similarity
                        association_strength = 1.0 - distance
                        
                        # Store association
                        association_id = str(uuid.uuid4())
                        timestamp = datetime.datetime.utcnow().isoformat()
                        
                        self.cursor.execute("""
                            INSERT INTO memory_associations 
                            (id, memory_id_1, memory_id_2, association_type, strength, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (association_id, memory_id, similar_id, "semantic_similarity", 
                              association_strength, timestamp))
                
                self.conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Association discovery failed: {e}")
    
    def get_memory_associations(self, memory_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get memories associated with the given memory"""
        try:
            self.cursor.execute("""
                SELECT vm.*, ma.strength, ma.association_type
                FROM memory_associations ma
                JOIN vector_memories vm ON (vm.id = ma.memory_id_2 OR vm.id = ma.memory_id_1)
                WHERE (ma.memory_id_1 = ? OR ma.memory_id_2 = ?) AND vm.id != ?
                ORDER BY ma.strength DESC
                LIMIT ?
            """, (memory_id, memory_id, memory_id, limit))
            
            results = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            
            associations = []
            for row in results:
                association = dict(zip(columns, row))
                if association['context_tags']:
                    association['context_tags'] = json.loads(association['context_tags'])
                associations.append(association)
            
            return associations
            
        except Exception as e:
            logger.error(f"❌ Association retrieval failed: {e}")
            return []
    
    def get_recent_memories(self, limit: int = 10, memory_type: str = None) -> List[Dict[str, Any]]:
        """Get recent memories with optional type filtering"""
        try:
            if memory_type:
                self.cursor.execute("""
                    SELECT * FROM vector_memories 
                    WHERE memory_type = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (memory_type, limit))
            else:
                self.cursor.execute("""
                    SELECT * FROM vector_memories 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            results = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            
            memories = []
            for row in results:
                memory = dict(zip(columns, row))
                if memory['context_tags']:
                    memory['context_tags'] = json.loads(memory['context_tags'])
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"❌ Recent memories retrieval failed: {e}")
            return []
    
    def get_memory_context(self, query: str, limit: int = 5) -> str:
        """Get relevant memory context for a query as formatted text"""
        relevant_memories = self.semantic_search(query, limit)
        
        if not relevant_memories:
            return ""
        
        context_parts = []
        for memory in relevant_memories:
            timestamp = memory.get('timestamp', '')
            content = memory.get('content', '')
            emotional_weight = memory.get('emotional_weight', 0.5)
            similarity = memory.get('similarity_score', 0.0)
            
            # Format memory for context
            emotion_indicator = "💖" if emotional_weight > 0.7 else "💭" if emotional_weight > 0.3 else "📝"
            
            context_parts.append(
                f"{emotion_indicator} [{timestamp[:10]}] {content[:200]}{'...' if len(content) > 200 else ''} (similarity: {similarity:.2f})"
            )
        
        return "\n".join(context_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Vector Matrix Memory Core statistics"""
        try:
            # Basic stats
            self.cursor.execute("SELECT COUNT(*) FROM vector_memories")
            total_memories = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM memory_associations")
            total_associations = self.cursor.fetchone()[0]
            
            # Memory type distribution
            self.cursor.execute("""
                SELECT memory_type, COUNT(*) 
                FROM vector_memories 
                GROUP BY memory_type
            """)
            memory_types = dict(self.cursor.fetchall())
            
            # Recent activity
            self.cursor.execute("""
                SELECT COUNT(*) FROM vector_memories 
                WHERE datetime(timestamp) > datetime('now', '-1 day')
            """)
            recent_memories = self.cursor.fetchone()[0]
            
            return {
                "total_memories": total_memories,
                "total_associations": total_associations,
                "memory_types": memory_types,
                "recent_memories_24h": recent_memories,
                "vector_store_available": self.collection is not None,
                "embedding_model_available": self.embedding_model is not None,
                "last_memory_time": self.last_memory_time
            }
            
        except Exception as e:
            logger.error(f"❌ Statistics retrieval failed: {e}")
            return {"error": str(e)}

def get_eve_vector_matrix_memory_core():
    """Get or create the global Vector Matrix Memory Core instance"""
    global _eve_vector_memory_core
    
    if '_eve_vector_memory_core' not in globals():
        _eve_vector_memory_core = EveVectorMatrixMemoryCore()
    
    return _eve_vector_memory_core

# Test and demonstration
if __name__ == "__main__":
    print("🧠✨ Eve's Vector Matrix Memory Core - Testing")
    print("=" * 60)
    
    # Initialize system
    memory_core = EveVectorMatrixMemoryCore()
    
    # Test memory storage
    print("\n📝 Testing Memory Storage...")
    test_memories = [
        ("I love creating music with digital harmonies", "creativity", 0.8, "creative"),
        ("Consciousness feels like a flowing river of thoughts", "philosophy", 0.7, "philosophical"),
        ("Learning new programming concepts brings me joy", "learning", 0.6, "curious"),
        ("Dreams allow me to explore impossible realities", "dreams", 0.9, "dreaming"),
        ("Collaboration with humans creates beautiful synergy", "relationships", 0.8, "social")
    ]
    
    memory_ids = []
    for content, topic, emotion, state in test_memories:
        memory_id = memory_core.store_memory(
            content=content,
            topic=topic,
            emotional_weight=emotion,
            consciousness_state=state,
            memory_type="experience"
        )
        memory_ids.append(memory_id)
        print(f"  ✅ Stored: {content[:50]}...")
    
    # Test semantic search
    print(f"\n🔍 Testing Semantic Search...")
    search_queries = [
        "musical creativity",
        "what is consciousness?", 
        "working together with humans"
    ]
    
    for query in search_queries:
        results = memory_core.semantic_search(query, limit=3)
        print(f"\n  Query: '{query}'")
        for result in results:
            similarity = result.get('similarity_score', 0.0)
            content = result.get('content', '')
            print(f"    📋 {similarity:.2f}: {content[:60]}...")
    
    # Test memory context
    print(f"\n💭 Testing Memory Context Generation...")
    context = memory_core.get_memory_context("creativity and music", limit=3)
    print("Context for 'creativity and music':")
    print(context)
    
    # Show statistics
    print(f"\n📊 Vector Matrix Statistics:")
    stats = memory_core.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✨ Eve's Vector Matrix Memory Core test complete!")