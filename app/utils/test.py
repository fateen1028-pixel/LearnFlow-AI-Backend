"""
test.py - Test the HTTP embedding service
"""
import sys
import os

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_service import HFLocalEmbeddings

def test_embeddings():
    """Test the HTTP embedding service"""
    print("🧪 Testing HTTP Embedding Service")
    print("=" * 50)
    
    # Initialize
    hf = HFLocalEmbeddings()
    
    # Test 1: Single embedding
    print("\n1. Testing single embedding...")
    text = "Hello world, this is a test for AI learning"
    embedding = hf.embed(text)
    
    if embedding:
        print(f"✅ Single embedding successful")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Last 5 values: {embedding[-5:]}")
    else:
        print("❌ Single embedding failed")
    
    # Test 2: Multiple embeddings
    print("\n2. Testing multiple embeddings...")
    texts = [
        "What is machine learning?",
        "Explain neural networks",
        "How to train a model"
    ]
    embeddings = hf.embed_texts(texts)
    
    if embeddings:
        print(f"✅ Multiple embeddings successful")
        print(f"   Number of embeddings: {len(embeddings)}")
        for i, emb in enumerate(embeddings):
            print(f"   Text {i+1}: {len(emb)} dimensions")
    else:
        print("❌ Multiple embeddings failed")
    
    # Test 3: Empty text
    print("\n3. Testing edge cases...")
    empty_embedding = hf.embed("")
    print(f"   Empty text: {'Failed' if empty_embedding else 'Handled correctly (empty list)'}")
    
    # Test 4: Very long text
    long_text = "A " * 1000
    long_embedding = hf.embed(long_text)
    print(f"   Long text: {'Success' if long_embedding else 'Failed'}")
    
    print("\n" + "=" * 50)
    print("✅ Test completed")
    
    # Return summary
    return {
        "single_embedding": len(embedding) if embedding else 0,
        "multiple_embeddings": len(embeddings) if embeddings else 0,
        "embedding_dimension": len(embedding) if embedding else 0
    }

if __name__ == "__main__":
    result = test_embeddings()
    print(f"\n📊 Summary:")
    print(f"   Embedding dimension: {result['embedding_dimension']}")
    print(f"   Note: Set EMBED_DIM={result['embedding_dimension']} in your .env file")