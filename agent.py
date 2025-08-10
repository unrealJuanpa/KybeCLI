import json
import requests
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from memory_manager import MemoryManager

class Agent:
    def __init__(self, name: str, system_prompt: str, llm_model: str, embedding_model: str, 
                 short_term_memory_interactions: int = 10, long_term_memory_recalls: int = 5,
                 server_url: str = "http://localhost:11434", 
                 additional_databases: List[Dict[str, Any]] = None):
        """
        Inicializa el agente con soporte para múltiples bases de datos RAG.
        
        Args:
            name: Nombre del agente
            system_prompt: Prompt del sistema
            llm_model: Modelo LLM de Ollama
            embedding_model: Modelo de embeddings
            short_term_memory_interactions: Interacciones en memoria corta
            long_term_memory_recalls: Recuerdos de BD principal
            server_url: URL del servidor Ollama
            additional_databases: Lista de BDs adicionales
                Formato: [{"name": "bd_name", "path": "bd_path", "limit": 3}, ...]
        """
        