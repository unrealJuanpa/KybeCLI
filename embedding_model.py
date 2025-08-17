import json
import os
import requests
import numpy as np
from typing import List, Optional

class EmbeddingModel:
    def __init__(self, model: str = 'nomic-embed-text', 
                 server_url: str = "http://localhost:11434", 
                 normalize: bool = True):
        """
        Initialize the EmbeddingModel with Ollama provider.
        
        Args:
            model (str): The model to use for embeddings
            server_url (str): The Ollama server URL (default: http://localhost:11434)
            normalize (bool): Whether to normalize the embeddings (default: True)
        """
        self.model = model
        self.normalize = normalize
        self.server_url = server_url.rstrip('/')
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize the embedding vector to unit length"""
        if not self.normalize:
            return embedding
            
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return (np.array(embedding) / norm).tolist()
    
    def calculate(self, text: str) -> List[float]:
        """
        Calculate embeddings for the input text using Ollama.
        
        Args:
            text (str): The input text to get embeddings for
            
        Returns:
            List[float]: The embedding vector
            
        Raises:
            Exception: If there's an error getting the embeddings
        """
        return self._calculate_ollama(text)
    
    def _calculate_ollama(self, text: str) -> List[float]:
        """Calculate embeddings using Ollama API"""
        url = f"{self.server_url}/api/embed"
        payload = {
            "model": self.model,
            "input": text
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            embedding = result.get('embedding', [])
            return self._normalize_embedding(embedding)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error getting embeddings from Ollama: {str(e)}")
    
