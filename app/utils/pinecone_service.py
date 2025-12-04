"""
pinecone_service.py - Graceful Pinecone service with fallback
Handles Pinecone vector database operations for chat memory.
If Pinecone fails, memory features will be disabled gracefully.
"""
import os
import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# -------------------------
# Configuration - UPDATED FOR YOUR 384D MODEL
# -------------------------
# Your model returns 384 dimensions (from test output)
EMBED_DIM = 384  # Hardcoded to match your model - NO LONGER FROM ENV
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hugging-face-embedding-model-api.onrender.com/embed")

# -------------------------
# Pinecone availability detection
# -------------------------
PINECONE_AVAILABLE = False
try:
    import pinecone

    # Pinecone v3+ exposes `Pinecone` client class
    if hasattr(pinecone, "Pinecone"):
        from pinecone import Pinecone  # type: ignore
        PINECONE_AVAILABLE = True
        print("✅ Pinecone v3+ detected and available")
    else:
        # Fallback: older import shape (should rarely be needed)
        import pinecone as pc  # type: ignore
        PINECONE_AVAILABLE = True
        print("✅ Pinecone detected (legacy import)")
except ImportError as e:
    print(f"⚠️ Pinecone package not found: {e}")
    PINECONE_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Error importing Pinecone: {e}")
    PINECONE_AVAILABLE = False

# -------------------------
# Embeddings initialization (HTTP-based to Render.com)
# -------------------------
EMBEDDINGS_AVAILABLE = False
embeddings = None

try:
    # Import our HTTP-based embeddings
    from app.utils.huggingface_service import HFLocalEmbeddings
    
    # Initialize with Render.com endpoint
    embeddings = HFLocalEmbeddings()
    
    # Quick test
    test_vec = embeddings.embed("test")
    
    if test_vec and len(test_vec) == 384:
        EMBEDDINGS_AVAILABLE = True
        print(f"✅ HTTP Embeddings available via {HF_ENDPOINT}")
        print(f"   Confirmed: 384-dimensional vectors")
    else:
        EMBEDDINGS_AVAILABLE = False
        print(f"⚠️ Embeddings test failed - expected 384D, got {len(test_vec) if test_vec else 0}D")
        embeddings = None
        
except ImportError as e:
    print(f"⚠️ Could not import HFLocalEmbeddings: {e}")
    EMBEDDINGS_AVAILABLE = False
    embeddings = None
except Exception as e:
    print(f"⚠️ Error initializing embeddings: {e}")
    EMBEDDINGS_AVAILABLE = False
    embeddings = None

# -------------------------
# PineconeService class
# -------------------------
class PineconeService:
    """Service for Pinecone vector database operations with graceful fallback."""

    def __init__(self):
        self.available: bool = False
        self.index = None
        self.embeddings = embeddings
        self.embed_dim = EMBED_DIM  # Always 384 for your model

        # API key check
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            print("⚠️ PINECONE_API_KEY not set, Pinecone disabled")
            return

        if not PINECONE_AVAILABLE:
            print("⚠️ Pinecone package not available, Pinecone disabled")
            return

        if not EMBEDDINGS_AVAILABLE:
            print("⚠️ Embeddings not available, Pinecone disabled")
            return

        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=pinecone_api_key)

            self.index_name = os.getenv("PINECONE_INDEX_NAME", "ai-learning-chat")

            # List all indexes to debug
            try:
                existing_indexes = self.pc.list_indexes()
                print(f"📊 Existing Pinecone indexes: {existing_indexes}")
            except:
                print("ℹ️ Could not list indexes (may not have permission)")

            # Check if index exists
            try:
                self.index = self.pc.Index(self.index_name)
                print(f"✅ Connected to existing Pinecone index: {self.index_name}")
                
                # Get index stats to verify
                try:
                    index_stats = self.index.describe_index_stats()
                    print(f"📊 Index stats: {index_stats}")
                    print(f"📊 Index dimension: {index_stats.get('dimension', 'unknown')}")
                except:
                    print("ℹ️ Could not get index stats")
                
                self.available = True
                
            except Exception as e:
                # Index doesn't exist, create it
                print(f"⚠️ Index {self.index_name} not found: {e}")
                print(f"🔄 Creating new index: {self.index_name} (dim=384)")
                
                try:
                    # IMPORTANT: Use the correct serverless spec for your Pinecone plan
                    # For starter/free tier, use:
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=384,  # MUST match your embeddings
                        metric="cosine",
                        spec=pinecone.ServerlessSpec(
                            cloud="aws",  # or "gcp" based on your account
                            region="us-east-1"  # adjust based on your location
                        )
                    )
                    
                    # Wait for index to be ready
                    print("⏳ Waiting 30 seconds for index to be ready...")
                    time.sleep(30)
                    
                    self.index = self.pc.Index(self.index_name)
                    self.available = True
                    print(f"✅ Created and connected to new 384D index: {self.index_name}")
                    
                except Exception as create_error:
                    print(f"❌ Failed to create index: {create_error}")
                    print("💡 Try creating the index manually in Pinecone console:")
                    print(f"   - Name: {self.index_name}")
                    print(f"   - Dimension: 384")
                    print(f"   - Metric: cosine")
                    print(f"   - Cloud: aws (or gcp)")
                    print(f"   - Region: us-east-1 (or your region)")
                    self.available = False
                    
        except Exception as e:
            print(f"❌ Failed to initialize Pinecone: {e}")
            import traceback
            traceback.print_exc()
            self.available = False
    # -------------------------
    # Embedding creation
    # -------------------------
    def create_embedding(self, text: str) -> List[float]:
        """Create 384-dimensional embedding via Render.com endpoint."""
        if not self.embeddings:
            print("⚠️ Embeddings not available, returning zero vector")
            return [0.0] * 384

        try:
            vec = self.embeddings.embed(text)

            # Validate we got 384 dimensions
            if not vec:
                print(f"⚠️ Empty embedding for: {text[:50]}...")
                return [0.0] * 384
            
            if len(vec) != 384:
                print(f"⚠️ Wrong dimension: expected 384, got {len(vec)}")
                # Force to 384 dimensions
                if len(vec) > 384:
                    vec = vec[:384]
                else:
                    vec = vec + [0.0] * (384 - len(vec))
                print(f"   Adjusted to 384 dimensions")

            # Ensure all floats
            vec = [float(v) for v in vec]
            return vec
            
        except Exception as e:
            print(f"⚠️ Error creating embedding: {e}")
            return [0.0] * 384

    # -------------------------
    # Storage / retrieval
    # -------------------------
    def store_chat_pair(
        self,
        user_id: str,
        user_message: str,
        ai_response: str,
        topic: str,
        session_id: str,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Store a Q/A pair in Pinecone."""
        if not self.available or not self.index:
            print("⚠️ Pinecone not available, skipping storage")
            return None

        try:
            combined_text = f"User: {user_message}\nAI: {ai_response}"
            embedding = self.create_embedding(combined_text)

            # Quick check
            if len(embedding) != 384:
                print(f"❌ Cannot store: embedding is {len(embedding)}D, need 384D")
                return None

            vector_id = f"{user_id}_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"

            base_metadata = {
                "user_id": str(user_id),
                "user_message": user_message[:500],
                "ai_response": ai_response[:1000],
                "topic": topic[:100],
                "session_id": session_id,
                "type": "chat_pair",
                "timestamp": datetime.now().isoformat(),
                "text_length": len(combined_text)
            }
            if metadata:
                base_metadata.update(metadata)

            # Upsert to Pinecone
            self.index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": base_metadata
                }],
                namespace=str(user_id)
            )

            print(f"✅ Stored 384D chat pair for user {user_id}")
            return vector_id

        except Exception as e:
            error_msg = str(e)
            if "dimension" in error_msg:
                print(f"❌ PINE CONE DIMENSION ERROR: {error_msg}")
                print("   Your Pinecone index must be created with dimension=384")
                print("   Delete the index in Pinecone console or use a different name")
            else:
                print(f"❌ Error storing chat pair: {e}")
            return None

    def search_similar_chats(
    self,
    user_id: str,
    query: str,
    topic: Optional[str] = None,
    limit: int = 5,
    threshold: float = 0.7
) -> List[Dict]:
        """Search for similar past chats for a user."""
        if not self.available or not self.index:
            print("❌ Pinecone not available for search")
            return []

        try:
            print(f"🔍 SEARCH DEBUG - User: {user_id}, Query: '{query[:100]}...'")
            print(f"   Topic filter: {topic}")
            print(f"   Threshold: {threshold}")
            
            query_embedding = self.create_embedding(query)
            
            # Ensure 384D
            if not query_embedding or len(query_embedding) != 384:
                print(f"❌ Query embedding failed or wrong dimension: {len(query_embedding) if query_embedding else 0}D")
                return []
            
            # DEBUG: Show embedding stats
            if query_embedding:
                emb_min = min(query_embedding)
                emb_max = max(query_embedding)
                emb_mean = sum(query_embedding)/len(query_embedding)
                print(f"   Embedding stats - Min: {emb_min:.4f}, Max: {emb_max:.4f}, Mean: {emb_mean:.4f}")

            filter_dict = {"user_id": str(user_id)}
            if topic:
                filter_dict["topic"] = topic
                
            print(f"   Filter: {filter_dict}")
            
            # Perform the search
            results = self.index.query(
                namespace=str(user_id),
                vector=query_embedding,
                top_k=limit,
                filter=filter_dict,
                include_metadata=True
            )

            similar_chats: List[Dict[str, Any]] = []
            matches = getattr(results, "matches", None) or results.get("matches", [])
            
            print(f"   Raw matches found: {len(matches)}")
            for i, match in enumerate(matches):
                score = getattr(match, "score", None) or match.get("score", None)
                md = getattr(match, "metadata", None) or match.get("metadata", {})
                
                if score is None:
                    continue
                    
                print(f"     Match {i+1}: Score={score:.4f}, Topic={md.get('topic', 'N/A')}")
                
                if score < threshold:
                    print(f"       → Skipped (score {score:.4f} < threshold {threshold})")
                    continue
                    
                similar_chats.append({
                    "id": getattr(match, "id", None) or match.get("id"),
                    "score": score,
                    "user_message": md.get("user_message", ""),
                    "ai_response": md.get("ai_response", ""),
                    "topic": md.get("topic", ""),
                    "timestamp": md.get("timestamp", ""),
                    "metadata": md
                })

            print(f"✅ Found {len(similar_chats)} similar chats (after threshold filter)")
            return similar_chats

        except Exception as e:
            print(f"⚠️ Error searching similar chats: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_user_chat_history(
        self,
        user_id: str,
        limit: int = 20,
        topic: Optional[str] = None
    ) -> List[Dict]:
        """Get recent chat history for a user."""
        if not self.available or not self.index:
            return []

        try:
            filter_dict = {"user_id": str(user_id), "type": "chat_pair"}
            if topic:
                filter_dict["topic"] = topic

            # Zero vector for metadata-only query
            zero_vector = [0.0] * 384

            results = self.index.query(
                namespace=str(user_id),
                vector=zero_vector,
                top_k=limit,
                filter=filter_dict,
                include_metadata=True
            )

            chats: List[Dict[str, Any]] = []
            matches = getattr(results, "matches", None) or results.get("matches", [])
            for match in matches:
                md = getattr(match, "metadata", None) or match.get("metadata", {})
                chats.append({
                    "id": getattr(match, "id", None) or match.get("id"),
                    "score": getattr(match, "score", None) or match.get("score"),
                    "user_message": md.get("user_message", ""),
                    "ai_response": md.get("ai_response", ""),
                    "topic": md.get("topic", ""),
                    "timestamp": md.get("timestamp", ""),
                    "metadata": md
                })

            # Sort by timestamp (newest first)
            chats.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return chats[:limit]

        except Exception as e:
            print(f"⚠️ Error getting user chat history: {e}")
            return []

    def delete_user_chats(
        self,
        user_id: str,
        vector_ids: Optional[List[str]] = None
    ) -> bool:
        """Delete chats for a user."""
        if not self.available or not self.index:
            return False

        try:
            if vector_ids:
                self.index.delete(ids=vector_ids, namespace=str(user_id))
                print(f"✅ Deleted {len(vector_ids)} chats for user {user_id}")
            else:
                self.index.delete(delete_all=True, namespace=str(user_id))
                print(f"✅ Deleted ALL chats for user {user_id}")
            return True
        except Exception as e:
            print(f"⚠️ Error deleting chats: {e}")
            return False

    def get_stats(self, user_id: str) -> Dict:
        """Get statistics for user's chat memory."""
        if not self.available:
            return {"available": False, "total_chats": 0, "topics": {}}

        try:
            chats = self.get_user_chat_history(user_id, limit=1000)

            topics: Dict[str, int] = {}
            for chat in chats:
                topic_name = chat.get("topic", "unknown")
                topics[topic_name] = topics.get(topic_name, 0) + 1

            return {
                "available": True,
                "total_chats": len(chats),
                "topics": topics,
                "oldest_chat": chats[-1]["timestamp"] if chats else None,
                "newest_chat": chats[0]["timestamp"] if chats else None
            }
        except Exception as e:
            print(f"⚠️ Error getting stats: {e}")
            return {"available": False, "total_chats": 0, "topics": {}}


# -------------------------
# Global instance & helpers
# -------------------------
_pinecone_instance: Optional[PineconeService] = None

def get_pinecone_service() -> Optional[PineconeService]:
    """Get the Pinecone service instance (singleton)."""
    global _pinecone_instance
    if _pinecone_instance is None:
        try:
            _pinecone_instance = PineconeService()
        except Exception as e:
            print(f"❌ Failed to create PineconeService: {e}")
            _pinecone_instance = None
    return _pinecone_instance

def is_pinecone_available() -> bool:
    service = get_pinecone_service()
    return service is not None and service.available

def get_memory_status() -> Dict:
    """Get detailed memory system status for diagnostics."""
    service = get_pinecone_service()

    if service is None:
        return {
            "available": False,
            "reason": "Service not initialized",
            "pinecone_installed": PINECONE_AVAILABLE,
            "embeddings_available": EMBEDDINGS_AVAILABLE,
            "api_key_set": bool(os.getenv("PINECONE_API_KEY")),
            "embedding_endpoint": HF_ENDPOINT,
            "embedding_dimension": 384
        }

    return {
        "available": service.available,
        "pinecone_installed": PINECONE_AVAILABLE,
        "embeddings_available": EMBEDDINGS_AVAILABLE,
        "api_key_set": bool(os.getenv("PINECONE_API_KEY")),
        "embedding_endpoint": HF_ENDPOINT,
        "embedding_dimension": 384,
        "index_name": getattr(service, "index_name", None),
        "index_connected": service.index is not None
    }