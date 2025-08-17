import os
import sqlite3
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from agents import EmbeddingAgent

# Initialize embedding agent with normalization
EMBEDDING_AGENT = EmbeddingAgent(normalize=True)

class CodebaseManager:
    def __init__(self, base_path: str):
        """
        Initialize the CodebaseManager with a base path.
        
        Args:
            base_path: The root directory to scan for code files
        """
        self.base_path = os.path.abspath(base_path)
        self.db_path = os.path.join(self.base_path, 'kybe_codebase.db')
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS codebase (
                    path TEXT PRIMARY KEY,
                    checksum TEXT NOT NULL,
                    description TEXT NOT NULL,
                    description_embedding BLOB NOT NULL
                )
            ''')
            conn.commit()
    
    def _get_checksum(self, filepath: str) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except (IOError, PermissionError):
            return ""
    
    def _is_binary_file(self, filepath: str) -> bool:
        """Check if a file is binary."""
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(1024)
                return b'\x00' in chunk or not chunk.decode('utf-8', errors='ignore').isprintable()
        except (UnicodeDecodeError, PermissionError):
            return True
    
    def _get_file_description(self, filepath: str, rel_path: str) -> str:
        """Generate a description for a file."""
        if self._is_binary_file(filepath):
            file_type = mimetypes.guess_type(filepath)[0] or "binary"
            return f"File {rel_path}. This is a {file_type} file."
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(8192)  # Read first 8KB for description
                
            # Simple heuristic for code files
            if any(ext in filepath.lower() for ext in ['.py', '.js', '.java', '.c', '.cpp', '.h', '.go', '.rs']):
                # Get first few lines that aren't comments/imports
                lines = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith(('#', '//', '/*', '*', 'import', 'package', 'from')):
                        lines.append(line)
                        if len(lines) >= 5:
                            break
                sample = '\n'.join(lines)
                return f"File {rel_path}. Code file with content:\n{sample}"
            
            # For text files, just take first few lines
            return f"File {rel_path}. Text content:\n{content[:500]}"
            
        except Exception as e:
            return f"File {rel_path}. Could not read file: {str(e)}"
    
    def _get_gitignore_patterns(self) -> List[str]:
        """Get patterns from .gitignore files."""
        patterns = []
        for root, _, files in os.walk(self.base_path):
            if '.gitignore' in files:
                with open(os.path.join(root, '.gitignore'), 'r') as f:
                    patterns.extend([
                        os.path.join(root, line.strip()) 
                        for line in f 
                        if line.strip() and not line.startswith('#')
                    ])
        return patterns
    
    def _should_ignore(self, path: str, ignore_patterns: List[str]) -> bool:
        """Check if a path should be ignored based on .gitignore patterns."""
        rel_path = os.path.relpath(path, self.base_path)
        
        # Always ignore the database file and git directory
        if path == self.db_path or '.git' in path.split(os.sep):
            return True
            
        # Check against ignore patterns
        for pattern in ignore_patterns:
            if pattern.endswith('/'):
                pattern = pattern[:-1]
            if rel_path == pattern or rel_path.startswith(pattern + os.sep):
                return True
        return False
    
    def scan_and_update(self) -> None:
        """Scan the codebase and update the database with file information."""
        ignore_patterns = self._get_gitignore_patterns()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get existing files from DB
            cursor.execute('SELECT path, checksum FROM codebase')
            existing_files = dict(cursor.fetchall())
            
            # Track processed files
            processed_paths = set()
            
            for root, _, files in os.walk(self.base_path):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    rel_path = os.path.relpath(filepath, self.base_path)
                    
                    # Skip ignored files
                    if self._should_ignore(filepath, ignore_patterns):
                        continue
                    
                    checksum = self._get_checksum(filepath)
                    if not checksum:  # Skip unreadable files
                        continue
                        
                    processed_paths.add(rel_path)
                    
                    # Check if file is new or modified
                    if rel_path in existing_files:
                        if existing_files[rel_path] == checksum:
                            continue  # File unchanged, skip
                        
                    # Generate description and embedding
                    description = self._get_file_description(filepath, rel_path)
                    embedding = EMBEDDING_AGENT.embed(description)
                    
                    # Convert embedding to binary for storage
                    embedding_blob = sqlite3.Binary(embedding)
                    
                    # Insert or update the record
                    cursor.execute('''
                        INSERT OR REPLACE INTO codebase 
                        (path, checksum, description, description_embedding)
                        VALUES (?, ?, ?, ?)
                    ''', (rel_path, checksum, description, embedding_blob))
            
            # Remove deleted files
            deleted_paths = set(existing_files.keys()) - processed_paths
            if deleted_paths:
                placeholders = ','.join(['?'] * len(deleted_paths))
                cursor.execute(f'DELETE FROM codebase WHERE path IN ({placeholders})', tuple(deleted_paths))
            
            conn.commit()
    
    def search_similar(self, query: str, top_n: int = 5) -> List[Dict]:
        """Search for files similar to the query."""
        query_embedding = EMBEDDING_AGENT.embed(query)
        query_embedding_blob = sqlite3.Binary(query_embedding)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.create_function("cosine_sim", 2, self._cosine_similarity)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT path, description, cosine_sim(description_embedding, ?) as similarity
                FROM codebase
                ORDER BY similarity DESC
                LIMIT ?
            ''', (query_embedding_blob, top_n))
            
            return [
                {"path": row[0], "description": row[1], "similarity": row[2]}
                for row in cursor.fetchall()
            ]
    
    @staticmethod
    def _cosine_similarity(blob1: bytes, blob2: bytes) -> float:
        """
        Calculate cosine similarity between two normalized embedding blobs.
        
        Note: This is optimized for normalized vectors (||a|| = ||b|| = 1),
        so we can use a simple dot product instead of the full cosine formula.
        """
        import numpy as np
        from ast import literal_eval
        
        def blob_to_array(blob):
            if isinstance(blob, bytes):
                return np.frombuffer(blob, dtype=np.float32)
            return np.array(literal_eval(blob), dtype=np.float32)
        
        a = blob_to_array(blob1)
        b = blob_to_array(blob2)
        # Since vectors are normalized, we can use dot product directly
        return float(np.dot(a, b))  # No need for division by norms
