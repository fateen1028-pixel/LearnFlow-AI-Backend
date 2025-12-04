"""
huggingface_service.py - HTTP-based embeddings for Render.com deployed model
"""
import os
import requests
from typing import List, Optional
import time

class HFLocalEmbeddings:
    """HTTP client for Render.com deployed embedding model"""
    
    def __init__(self):
        # Your Render.com endpoint
        self.endpoint = os.getenv(
            "HF_ENDPOINT", 
            "https://hugging-face-embedding-model-api.onrender.com/embed"
        )
        self.api_key = os.getenv("HF_API_KEY", "")
        self.session = requests.Session()
        self.timeout = 45  # Increased for Render.com free tier
        
        # Headers for authentication
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "AI-Learning-Chat/1.0"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        print(f"✅ HFLocalEmbeddings initialized with endpoint: {self.endpoint}")
        
        # Test connection on initialization
        self.test_connection()
    
    def test_connection(self):
        """Test connection to embedding endpoint"""
        try:
            test_response = self.session.post(
                self.endpoint,
                json={"texts": ["ping"]},
                headers=self.headers,
                timeout=10
            )
            if test_response.status_code == 200:
                print(f"✅ Embedding endpoint is responsive")
            else:
                print(f"⚠️ Embedding endpoint returned status: {test_response.status_code}")
        except Exception as e:
            print(f"⚠️ Could not connect to embedding endpoint: {e}")
    
    def embed(self, text: str) -> List[float]:
        """Create embedding for single text (compatibility with pinecone_service.py)"""
        try:
            # Clean and validate text
            if not text or not isinstance(text, str):
                print("⚠️ Invalid text for embedding")
                return []
            
            # Trim very long text
            if len(text) > 10000:
                text = text[:10000]
                print(f"⚠️ Text truncated to 10000 characters for embedding")
            
            response = self.session.post(
                self.endpoint,
                json={"texts": [text]},
                headers=self.headers,
                timeout=self.timeout
            )
            
            # Check response
            if response.status_code != 200:
                print(f"⚠️ Embedding API returned status {response.status_code}: {response.text[:200]}")
                return []
            
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list):  # [[embeddings]]
                    embedding = data[0]
                elif isinstance(data[0], (int, float)):  # [embeddings]
                    embedding = data
                else:
                    print(f"⚠️ Unexpected list format: {type(data[0])}")
                    return []
            elif isinstance(data, dict):
                # Try common response formats
                if "embeddings" in data and data["embeddings"] and len(data["embeddings"]) > 0:
                    embedding = data["embeddings"][0]
                elif "embedding" in data and data["embedding"]:
                    embedding = data["embedding"]
                elif "vectors" in data and data["vectors"] and len(data["vectors"]) > 0:
                    embedding = data["vectors"][0]
                elif "vector" in data and data["vector"]:
                    embedding = data["vector"]
                else:
                    print(f"⚠️ Unexpected dict format, keys: {data.keys()}")
                    return []
            else:
                print(f"⚠️ Unexpected response type: {type(data)}")
                return []
            
            # Convert to list of floats
            embedding = [float(v) for v in embedding]
            
            # Validate embedding
            if not embedding:
                print("⚠️ Empty embedding received")
                return []
            
            if len(embedding) < 10:
                print(f"⚠️ Embedding too short: {len(embedding)} dimensions")
                return embedding  # Still return it
            
            print(f"✅ Generated embedding: {len(embedding)} dimensions")
            return embedding
            
        except requests.exceptions.Timeout:
            print(f"⚠️ Embedding request timed out after {self.timeout}s")
            return []
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Embedding request failed: {e}")
            return []
        except Exception as e:
            print(f"⚠️ Error in embed(): {str(e)[:200]}")
            return []
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts (compatibility with test.py)"""
        if not texts:
            return []
        
        try:
            # Clean texts
            clean_texts = []
            for text in texts:
                if text and isinstance(text, str):
                    if len(text) > 10000:
                        text = text[:10000]
                    clean_texts.append(text)
            
            if not clean_texts:
                return []
            
            response = self.session.post(
                self.endpoint,
                json={"texts": clean_texts},
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                print(f"⚠️ Embedding API returned status {response.status_code}")
                return []
            
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], list):
                    embeddings = data  # Already in correct format
                elif len(data) > 0 and isinstance(data[0], (int, float)):
                    embeddings = [data]  # Single embedding, wrap in list
                else:
                    print(f"⚠️ Unexpected list format")
                    return []
            elif isinstance(data, dict):
                if "embeddings" in data:
                    embeddings = data["embeddings"]
                elif "vectors" in data:
                    embeddings = data["vectors"]
                else:
                    # Try to find any list in the dict
                    for key, value in data.items():
                        if isinstance(value, list) and value and isinstance(value[0], list):
                            embeddings = value
                            break
                    else:
                        print(f"⚠️ No embeddings found in dict, keys: {data.keys()}")
                        return []
            else:
                print(f"⚠️ Unexpected response type: {type(data)}")
                return []
            
            # Convert all values to floats
            result = []
            for emb in embeddings:
                result.append([float(v) for v in emb])
            
            print(f"✅ Generated {len(result)} embeddings")
            return result
            
        except requests.exceptions.Timeout:
            print(f"⚠️ Embedding request timed out after {self.timeout}s")
            return []
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Embedding request failed: {e}")
            return []
        except Exception as e:
            print(f"⚠️ Error in embed_texts(): {e}")
            return []