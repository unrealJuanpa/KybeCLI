import numpy as np
from agents import EmbeddingAgent

def is_normalized(vector: np.ndarray) -> bool:
    """
    Check if a vector is normalized (has unit length).
    
    Args:
        vector: The vector to check
        
    Returns:
        bool: True if the vector is normalized (length ~= 1.0), False otherwise
    """
    length = np.linalg.norm(vector)
    # Allow for small floating point errors
    return abs(length - 1.0) < 1e-6

def main():
    # Create an embedding agent with normalization turned OFF
    embedding_agent = EmbeddingAgent(normalize=False)
    
    print("Testing if the embedding model returns normalized vectors by default...")
    print("Type 'exit' to quit.")
    print("-" * 50)
    
    while True:
        text = input("\nEnter text to embed: ").strip()
        if text.lower() in ['exit', 'quit', 'q']:
            break
            
        try:
            # Get the embedding
            embedding = embedding_agent.embed(text)
            vector = np.array(embedding)
            
            # Calculate vector length
            length = np.linalg.norm(vector)
            
            # Check if normalized
            normalized = is_normalized(vector)
            
            print(f"\nVector length: {length:.6f}")
            print(f"Normalized: {'YES' if normalized else 'NO'}")
            print(f"Vector shape: {vector.shape}")
            print(f"First 5 dimensions: {vector[:5]}")
            
            if not normalized:
                print("\nNote: The vector is not normalized. To get normalized vectors, set normalize=True when creating the EmbeddingAgent.")
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
