import json
import os
import requests
import numpy as np
from typing import List, Optional

class EmbeddingModel:
    def __init__(self, model: str = 'nomic-embed-text', 
                 server_ip: str = "http://localhost:11434", 
                 provider: str = "ollama",
                 normalize: bool = True):
        """
        Initialize the EmbeddingModel.
        
        Args:
            model (str): The model to use for embeddings (only used with ollama provider)
            server_ip (str): The server URL (default: http://localhost:11434 for ollama)
            provider (str): The provider to use ('ollama' or '0x255')
            normalize (bool): Whether to normalize the embeddings (default: True)
        """
        self.model = model
        self.provider = provider.lower()
        self.normalize = normalize
        
        if self.provider not in ['ollama', '0x255']:
            raise ValueError("Provider must be either 'ollama' or '0x255'")
            
        if self.provider == 'ollama':
            self.server_url = server_ip.rstrip('/')
        else:  # 0x255
            self.server_url = os.getenv('BACKEND_URL', '').rstrip('/')
            if not self.server_url:
                raise ValueError("BACKEND_URL environment variable is not set")
    
    def _load_token(self) -> Optional[str]:
        """Load the authentication token from account.json"""
        try:
            with open('account.json', 'r') as f:
                account_data = json.load(f)
                return account_data.get('token')
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None
    
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
        Calculate embeddings for the input text.
        
        Args:
            text (str): The input text to get embeddings for
            
        Returns:
            List[float]: The embedding vector
            
        Raises:
            Exception: If there's an error getting the embeddings
        """
        if self.provider == 'ollama':
            return self._calculate_ollama(text)
        else:  # 0x255
            return self._calculate_0x255(text)
    
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
    
    def _calculate_0x255(self, text: str) -> List[float]:
        """Calculate embeddings using 0x255 API"""
        token = self._load_token()
        if not token:
            raise Exception("No authentication token found. Please log in first.")
        
        url = f"{self.server_url}/api/embeddings"
        headers = {
            "Authorization": token,
            "Content-Type": "application/json"
        }
        payload = {
            "inputText": text
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 0x255 already returns normalized embeddings, but we'll respect the normalize flag
            embedding = result.get('embedding', [])
            return self._normalize_embedding(embedding) if self.normalize else embedding
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error getting embeddings from 0x255: {str(e)}"
            if response.status_code == 401:
                error_msg += " (Invalid or expired token)"
            raise Exception(error_msg)
