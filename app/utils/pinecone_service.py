"""
pinecone_service.py - Graceful Pinecone service with fallback
Handles Pinecone vector database operations for chat memory.
If Pinecone fails, memory features will be disabled gracefully.
"""
import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

# Flag to track if Pinecone is available
PINECONE_AVAILABLE = False

try:
    # Try to import Pinecone with multiple strategies
    import pinecone
    
    # Check if it's the new version
    if hasattr(pinecone, 'Pinecone'):
        from pinecone import Pinecone
        PINECONE_AVAILABLE = True
        print("✅ Pinecone v3+ detected and available")
    else:
        # Try to import the old way
        import pinecone as pc
        PINECONE_AVAILABLE = True
        print("✅ Pinecone v2 detected and available")
        
except ImportError as e:
    print(f"⚠️ Pinecone package not found: {e}")
    PINECONE_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Error importing Pinecone: {e}")
    PINECONE_AVAILABLE = False

# Initialize embeddings only if we have Google API key
EMBEDDINGS_AVAILABLE = False
embeddings = None

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        EMBEDDINGS_AVAILABLE = True
        print("✅ Google Embeddings available")
    else:
        print("⚠️ GOOGLE_API_KEY not set, embeddings disabled")
except ImportError as e:
    print(f"⚠️ Google Generative AI not available: {e}")
except Exception as e:
    print(f"⚠️ Error initializing embeddings: {e}")


class PineconeService:
    """Service for Pinecone vector database operations with graceful fallback."""
    
    def __init__(self):
        """Initialize Pinecone if available, otherwise use dummy mode."""
        self.available = False
        self.index = None
        self.embeddings = embeddings
        
        # Check if we have the required API keys
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
            # Initialize Pinecone
            self.pc = Pinecone(api_key=pinecone_api_key)
            
            # Set index name
            self.index_name = os.getenv("PINECONE_INDEX_NAME", "ai-learning-chat")
            
            # Try to connect to index
            try:
                self.index = self.pc.Index(self.index_name)
                print(f"✅ Connected to Pinecone index: {self.index_name}")
                self.available = True
            except Exception as e:
                print(f"⚠️ Could not connect to index {self.index_name}: {e}")
                # Try to create index
                try:
                    print(f"Attempting to create index: {self.index_name}")
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=768,
                        metric="cosine"
                    )
                    # Wait a bit for index to be ready
                    import time
                    time.sleep(2)
                    self.index = self.pc.Index(self.index_name)
                    self.available = True
                    print(f"✅ Created new index: {self.index_name}")
                except Exception as create_error:
                    print(f"❌ Failed to create index: {create_error}")
                    self.available = False
                    
        except Exception as e:
            print(f"❌ Failed to initialize Pinecone: {e}")
            self.available = False
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text or return zero vector."""
        if not self.embeddings:
            return [0.0] * 768
        
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"⚠️ Error creating embedding: {e}")
            return [0.0] * 768
    
    def store_chat_pair(
        self, 
        user_id: str,
        user_message: str,
        ai_response: str,
        topic: str,
        session_id: str,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Store a Q/A pair in Pinecone.
        Returns vector ID if successful, None otherwise.
        """
        if not self.available or not self.index:
            print("⚠️ Pinecone not available, skipping storage")
            return None
        
        try:
            # Create embedding
            combined_text = f"User: {user_message}\nAI: {ai_response}"
            embedding = self.create_embedding(combined_text)
            
            # Generate unique ID
            vector_id = f"{user_id}_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
            
            # Prepare metadata
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
            
            # Store in Pinecone
            self.index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": base_metadata
                }],
                namespace=str(user_id)
            )
            
            print(f"✅ Stored chat pair for user {user_id}")
            return vector_id
            
        except Exception as e:
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
        """
        Search for similar past chats for a user.
        """
        if not self.available or not self.index:
            return []
        
        try:
            # Create embedding for the query
            query_embedding = self.create_embedding(query)
            
            # Prepare filter
            filter_dict = {"user_id": str(user_id)}
            if topic:
                filter_dict["topic"] = topic
            
            # Query Pinecone
            results = self.index.query(
                namespace=str(user_id),
                vector=query_embedding,
                top_k=limit,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Process results
            similar_chats = []
            if hasattr(results, 'matches'):
                for match in results.matches:
                    if match.score >= threshold:
                        similar_chats.append({
                            "id": match.id,
                            "score": match.score,
                            "user_message": match.metadata.get("user_message", ""),
                            "ai_response": match.metadata.get("ai_response", ""),
                            "topic": match.metadata.get("topic", ""),
                            "timestamp": match.metadata.get("timestamp", ""),
                            "metadata": match.metadata
                        })
            
            return similar_chats
            
        except Exception as e:
            print(f"⚠️ Error searching similar chats: {e}")
            return []
    
    def get_user_chat_history(
        self,
        user_id: str,
        limit: int = 20,
        topic: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recent chat history for a user.
        """
        if not self.available or not self.index:
            return []
        
        try:
            # Use search with a zero vector
            filter_dict = {"user_id": str(user_id), "type": "chat_pair"}
            if topic:
                filter_dict["topic"] = topic
            
            zero_vector = [0.0] * 768
            
            results = self.index.query(
                namespace=str(user_id),
                vector=zero_vector,
                top_k=limit,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Process and sort
            chats = []
            if hasattr(results, 'matches'):
                for match in results.matches:
                    chats.append({
                        "id": match.id,
                        "score": match.score,
                        "user_message": match.metadata.get("user_message", ""),
                        "ai_response": match.metadata.get("ai_response", ""),
                        "topic": match.metadata.get("topic", ""),
                        "timestamp": match.metadata.get("timestamp", ""),
                        "metadata": match.metadata
                    })
            
            # Sort by timestamp
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
            else:
                self.index.delete(delete_all=True, namespace=str(user_id))
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
            
            topics = {}
            for chat in chats:
                topic = chat.get("topic", "unknown")
                topics[topic] = topics.get(topic, 0) + 1
            
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


# Global instance
_pinecone_instance = None

def get_pinecone_service() -> Optional[PineconeService]:
    """Get the Pinecone service instance."""
    global _pinecone_instance
    
    if _pinecone_instance is None:
        try:
            _pinecone_instance = PineconeService()
        except Exception as e:
            print(f"❌ Failed to create PineconeService: {e}")
            _pinecone_instance = None
    
    return _pinecone_instance

def is_pinecone_available() -> bool:
    """Check if Pinecone is available and working."""
    service = get_pinecone_service()
    return service is not None and service.available

def get_memory_status() -> Dict:
    """Get detailed memory system status."""
    service = get_pinecone_service()
    
    if service is None:
        return {
            "available": False,
            "reason": "Service not initialized",
            "pinecone_installed": PINECONE_AVAILABLE,
            "embeddings_available": EMBEDDINGS_AVAILABLE,
            "api_key_set": bool(os.getenv("PINECONE_API_KEY"))
        }
    
    return {
        "available": service.available,
        "pinecone_installed": PINECONE_AVAILABLE,
        "embeddings_available": EMBEDDINGS_AVAILABLE,
        "api_key_set": bool(os.getenv("PINECONE_API_KEY")),
        "index_name": getattr(service, 'index_name', None),
        "index_connected": service.index is not None
    }