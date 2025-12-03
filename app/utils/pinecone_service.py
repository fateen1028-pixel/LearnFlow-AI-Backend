"""
pinecone_service.py
Handles Pinecone vector database operations for chat memory.
"""
import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class PineconeService:
    """Service for Pinecone vector database operations."""
    
    def __init__(self):
        """Initialize Pinecone and embeddings."""
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "ai-learning-chat")
        
        if not self.api_key or not self.environment:
            raise ValueError("Pinecone API key and environment must be set")
        
        # Initialize Pinecone
        pinecone.init(
            api_key=self.api_key,
            environment=self.environment
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create or connect to index
        self.setup_index()
        
        # Get index reference
        self.index = pinecone.Index(self.index_name)
    
    def setup_index(self):
        """Create Pinecone index if it doesn't exist."""
        if self.index_name not in pinecone.list_indexes():
            # Create new index
            pinecone.create_index(
                name=self.index_name,
                dimension=768,  # embedding-001 dimension
                metric="cosine",
                metadata_config={
                    "indexed": ["user_id", "topic", "type", "session_id", "timestamp"]
                }
            )
            print(f"Created new Pinecone index: {self.index_name}")
        else:
            print(f"Using existing Pinecone index: {self.index_name}")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text."""
        return self.embeddings.embed_query(text)
    
    def store_chat_pair(
        self, 
        user_id: str,
        user_message: str,
        ai_response: str,
        topic: str,
        session_id: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a Q/A pair in Pinecone.
        Returns the vector ID.
        """
        try:
            # Create embedding from the combined Q/A for better retrieval
            combined_text = f"User: {user_message}\nAI: {ai_response}"
            embedding = self.create_embedding(combined_text)
            
            # Generate unique ID
            vector_id = f"{user_id}_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
            
            # Prepare metadata
            base_metadata = {
                "user_id": str(user_id),
                "user_message": user_message[:500],  # Limit length
                "ai_response": ai_response[:1000],   # Limit length
                "topic": topic[:100],
                "session_id": session_id,
                "type": "chat_pair",
                "timestamp": datetime.now().isoformat(),
                "text_length": len(combined_text)
            }
            
            # Add custom metadata if provided
            if metadata:
                base_metadata.update(metadata)
            
            # Store in Pinecone
            self.index.upsert(
                vectors=[(vector_id, embedding, base_metadata)],
                namespace=str(user_id)  # Namespace per user for isolation
            )
            
            print(f"Stored chat pair for user {user_id}, vector_id: {vector_id}")
            return vector_id
            
        except Exception as e:
            print(f"Error storing chat pair: {e}")
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
        try:
            # Create embedding for the query
            query_embedding = self.create_embedding(query)
            
            # Prepare filter
            filter_dict = {"user_id": str(user_id)}
            if topic:
                filter_dict["topic"] = topic
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=limit,
                filter=filter_dict,
                include_metadata=True,
                namespace=str(user_id)
            )
            
            # Process results
            similar_chats = []
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
            
            print(f"Found {len(similar_chats)} similar chats for user {user_id}")
            return similar_chats
            
        except Exception as e:
            print(f"Error searching similar chats: {e}")
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
        try:
            # We'll use fetch with a timestamp filter, but Pinecone doesn't support ordering by timestamp
            # For now, we'll use search with a zero vector to get all
            filter_dict = {"user_id": str(user_id), "type": "chat_pair"}
            if topic:
                filter_dict["topic"] = topic
            
            # Use a zero vector for fetch-all (not ideal, but works)
            zero_vector = [0.0] * 768
            
            results = self.index.query(
                vector=zero_vector,
                top_k=limit,
                filter=filter_dict,
                include_metadata=True,
                namespace=str(user_id)
            )
            
            # Sort by timestamp
            chats = []
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
            
            # Sort by timestamp descending
            chats.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return chats[:limit]
            
        except Exception as e:
            print(f"Error getting user chat history: {e}")
            return []
    
    def delete_user_chats(
        self,
        user_id: str,
        vector_ids: Optional[List[str]] = None
    ) -> bool:
        """
        Delete specific chats or all chats for a user.
        """
        try:
            if vector_ids:
                # Delete specific vectors
                self.index.delete(ids=vector_ids, namespace=str(user_id))
                print(f"Deleted {len(vector_ids)} chats for user {user_id}")
            else:
                # Delete all vectors for user (using namespace)
                self.index.delete(delete_all=True, namespace=str(user_id))
                print(f"Deleted all chats for user {user_id}")
            
            return True
            
        except Exception as e:
            print(f"Error deleting chats: {e}")
            return False
    
    def get_stats(self, user_id: str) -> Dict:
        """Get statistics for user's chat memory."""
        try:
            chats = self.get_user_chat_history(user_id, limit=1000)
            
            # Group by topic
            topics = {}
            for chat in chats:
                topic = chat.get("topic", "unknown")
                if topic not in topics:
                    topics[topic] = 0
                topics[topic] += 1
            
            return {
                "total_chats": len(chats),
                "topics": topics,
                "oldest_chat": chats[-1]["timestamp"] if chats else None,
                "newest_chat": chats[0]["timestamp"] if chats else None
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_chats": 0, "topics": {}}

# Singleton instance
pinecone_service = None

def get_pinecone_service():
    """Get or create Pinecone service instance."""
    global pinecone_service
    if pinecone_service is None:
        try:
            pinecone_service = PineconeService()
        except Exception as e:
            print(f"Failed to initialize Pinecone: {e}")
            pinecone_service = None
    return pinecone_service

def is_pinecone_available() -> bool:
    """Check if Pinecone is configured and available."""
    service = get_pinecone_service()
    return service is not None