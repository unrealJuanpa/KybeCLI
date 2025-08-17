import json
import requests
import sqlite3
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

class MemoryManager:
    def __init__(self, db_name: str, server_url: str = "http://localhost:11434"):
        """
        Inicializa el gestor de memoria con una base de datos SQLite.
        
        Args:
            db_name (str): Nombre de la base de datos (sin extensión .db)
            server_url (str): URL del servidor Ollama
        """
        self.db_path = f"{db_name}.db"
        self.server_url = server_url.rstrip('/')
        self.embeddings_endpoint = f"{self.server_url}/api/embeddings"
        
        # Crear tabla si no existe
        self._create_table()
    
    def _create_table(self):
        """
        Crea la tabla de memorias si no existe.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    interaction_uuid TEXT PRIMARY KEY,
                    interaction_content TEXT NOT NULL,
                    interaction_role TEXT NOT NULL,
                    interaction_embeddings TEXT NOT NULL,
                    interaction_level INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def get_embedding(self, text: str, embedding_model: str) -> List[float]:
        """
        Obtiene el embedding de un texto usando Ollama.
        
        Args:
            text (str): Texto para obtener embedding
            embedding_model (str): Modelo de embeddings a usar
            
        Returns:
            list: Vector de embedding normalizado
        """
        try:
            data = {
                "model": embedding_model,
                "prompt": text
            }
            
            response = requests.post(self.embeddings_endpoint, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get('embedding', [])
            
            if not embedding:
                raise ValueError("No se pudo obtener embedding de Ollama")
            
            # Normalizar embedding
            normalized_embedding = self._normalize_embedding(embedding)
            
            return normalized_embedding
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error obteniendo embedding de Ollama: {str(e)}")
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normaliza un embedding usando L2 norm.
        
        Args:
            embedding (list): Vector de embedding
            
        Returns:
            list: Vector normalizado
        """
        embedding_array = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding_array)
        
        if norm == 0:
            return embedding_array.tolist()
        
        normalized = embedding_array / norm
        return normalized.tolist()
    
    def save_memory(self, content: str, role: str, embedding_model: str, 
                   level: int = 0, interaction_uuid: Optional[str] = None) -> str:
        """
        Guarda una memoria en la base de datos.
        
        Args:
            content (str): Contenido de la interacción
            role (str): Rol ('user' o 'assistant')
            embedding_model (str): Modelo para calcular embeddings
            level (int): Nivel de la interacción
            interaction_uuid (str, optional): UUID específico o se genera automáticamente
            
        Returns:
            str: UUID de la memoria guardada
        """
        # Generar UUID si no se proporciona
        if interaction_uuid is None:
            interaction_uuid = str(uuid.uuid4())
        
        # Obtener embedding
        embedding = self.get_embedding(content, embedding_model)
        embedding_json = json.dumps(embedding)
        
        # Guardar en BD
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO memories 
                (interaction_uuid, interaction_content, interaction_role, 
                 interaction_embeddings, interaction_level, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction_uuid, content, role, embedding_json, level,
                datetime.now().isoformat(), datetime.now().isoformat()
            ))
            conn.commit()
        
        return interaction_uuid
    
    def update_memory(self, interaction_uuid: str, new_content: str, embedding_model: str):
        """
        Actualiza una memoria existente y recalcula su embedding.
        
        Args:
            interaction_uuid (str): UUID de la memoria a actualizar
            new_content (str): Nuevo contenido
            embedding_model (str): Modelo para recalcular embeddings
        """
        # Obtener nuevo embedding
        embedding = self.get_embedding(new_content, embedding_model)
        embedding_json = json.dumps(embedding)
        
        # Actualizar en BD
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE memories 
                SET interaction_content = ?, interaction_embeddings = ?, updated_at = ?
                WHERE interaction_uuid = ?
            """, (new_content, embedding_json, datetime.now().isoformat(), interaction_uuid))
            
            if cursor.rowcount == 0:
                raise ValueError(f"No se encontró memoria con UUID: {interaction_uuid}")
            
            conn.commit()
    
    def get_recent_interactions(self, limit: int) -> List[Dict[str, Any]]:
        """
        Obtiene las últimas N interacciones ordenadas por fecha con embeddings incluidos.
        
        Args:
            limit (int): Número de interacciones a obtener
            
        Returns:
            list: Lista de interacciones recientes con embeddings
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM memories 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit * 2,))  # * 2 porque cada interacción tiene user + assistant
            
            rows = cursor.fetchall()
        
        # Convertir a diccionarios y revertir orden (más antiguo primero)
        interactions = []
        for row in reversed(rows):
            interactions.append({
                'uuid': row['interaction_uuid'],
                'role': row['interaction_role'],
                'content': row['interaction_content'],
                'embeddings': json.loads(row['interaction_embeddings']),
                'level': row['interaction_level'],
                'created_at': row['created_at']
            })
        
        return interactions
    
    def search_similar_memories(self, query_embedding: List[float], limit: int) -> List[Dict[str, Any]]:
        """
        Busca memorias similares usando cosine similarity.
        
        Args:
            query_embedding (list): Embedding de la consulta (ya normalizado)
            limit (int): Número máximo de resultados
            
        Returns:
            list: Lista de memorias ordenadas por similitud (mayor a menor)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT interaction_uuid, interaction_content, interaction_role, 
                       interaction_embeddings, interaction_level, created_at
                FROM memories 
                ORDER BY created_at DESC
            """)
            
            rows = cursor.fetchall()
        
        # Calcular similitudes
        similarities = []
        query_array = np.array(query_embedding, dtype=np.float32)
        
        for row in rows:
            stored_embedding = json.loads(row['interaction_embeddings'])
            stored_array = np.array(stored_embedding, dtype=np.float32)
            
            # Cosine similarity (dot product de vectores normalizados)
            similarity = float(np.dot(query_array, stored_array))
            
            similarities.append({
                'uuid': row['interaction_uuid'],
                'content': row['interaction_content'],
                'role': row['interaction_role'],
                'level': row['interaction_level'],
                'created_at': row['created_at'],
                'similarity': similarity
            })
        
        # Ordenar por similitud (mayor a menor) y limitar
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]
    
    def search_memories_by_text(self, text: str, embedding_model: str, limit: int) -> List[Dict[str, Any]]:
        """
        Busca memorias similares a partir de un texto.
        
        Args:
            text (str): Texto para buscar memorias similares
            embedding_model (str): Modelo para calcular embedding del texto
            limit (int): Número máximo de resultados
            
        Returns:
            list: Lista de memorias similares ordenadas por similitud
        """
        # Obtener embedding del texto
        text_embedding = self.get_embedding(text, embedding_model)
        
        # Buscar memorias similares
        return self.search_similar_memories(text_embedding, limit)
    
    def get_total_memories_count(self) -> int:
        """
        Obtiene el número total de memorias en la BD.
        
        Returns:
            int: Número total de memorias
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM memories")
                return cursor.fetchone()[0]
        except sqlite3.Error:
            return 0