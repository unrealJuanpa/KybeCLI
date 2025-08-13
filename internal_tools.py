import os
import sqlite3
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
import fnmatch
from agent import Agent

# Initialize the agent for file descriptions
description_agent = Agent(
    name="FileDescriber",
    system_prompt="""You are a helpful assistant that analyzes code files and provides 
    a clear, concise description of their purpose and functionality. Focus on the main 
    components, key functions, and overall architecture. Be specific and technical.""",
    provider="ollama",
    model="llama3"
)

def describe_file(content: str) -> str:
    """Generate a description of the file content using an LLM."""
    try:
        # Limit content length to avoid context window issues
        truncated_content = content[:8000]  # Adjust based on model context window
        
        response = description_agent.interact(
            f"Please analyze this code and provide a detailed description of its purpose and functionality. Be specific about what the code does, its main components, and any important functions or classes.\n\n{truncated_content}"
        )
        return response.get("content", "No description generated").strip()
    except Exception as e:
        print(f"Error generating file description: {str(e)}")
        return "Error generating description"

def get_gitignore_patterns(base_path: Path) -> List[str]:
    """Get all .gitignore patterns from the directory tree."""
    patterns = []
    for root, _, files in os.walk(base_path):
        if '.gitignore' in files:
            gitignore_path = Path(root) / '.gitignore'
            with open(gitignore_path, 'r') as f:
                # Read non-empty lines and remove comments
                patterns.extend([
                    line.strip()
                    for line in f.read().splitlines()
                    if line.strip() and not line.startswith('#')
                ])
    return patterns

def should_ignore(path: Path, gitignore_patterns: List[str]) -> bool:
    """Check if a path should be ignored based on .gitignore patterns."""
    path_str = str(path)
    for pattern in gitignore_patterns:
        # Convert pattern to match fnmatch format
        if pattern.startswith('/'):
            pattern = pattern[1:]  # Remove leading slash for matching
            if fnmatch.fnmatch(path.name, pattern):
                return True
        else:
            # Check if any parent directory matches the pattern
            parts = Path(path).parts
            for i in range(len(parts)):
                if fnmatch.fnmatch(parts[i], pattern):
                    return True
    return False

def calculate_checksum(file_path: Path) -> str:
    """Calculate MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error calculating checksum for {file_path}: {e}")
        return ""

def init_database(db_path: Path) -> sqlite3.Connection:
    """Initialize the SQLite database with required tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create interactions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        id TEXT PRIMARY KEY,
        uuid TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        layer INTEGER DEFAULT 0,
        embedding BLOB,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create codebase table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS codebase (
        filepath TEXT PRIMARY KEY,
        checksum TEXT NOT NULL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    return conn

def checkChanges(base_path: str):
    """
    Check for file changes in the directory and update the database.
    
    Args:
        base_path: Root directory to monitor for changes
    """
    base_path = Path(base_path).resolve()
    db_path = base_path / 'kybe.db'
    
    # Initialize database
    conn = init_database(db_path)
    cursor = conn.cursor()
    
    # Get existing files from database
    cursor.execute('SELECT filepath, checksum FROM codebase')
    db_files = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Get gitignore patterns
    gitignore_patterns = get_gitignore_patterns(base_path)
    
    # Track current files to detect deletions
    current_files = set()
    
    # Process all files in the directory
    for root, _, files in os.walk(base_path):
        root_path = Path(root)
        
        # Skip .git directory and database file
        if '.git' in root_path.parts or root_path == db_path.parent and db_path.name in files:
            continue
            
        for file in files:
            file_path = root_path / file
            rel_path = file_path.relative_to(base_path)
            
            # Skip binary files and database
            if file_path.suffix in ['.pyc', '.pyo', '.pyd', '.so', '.dll', '.db', '.sqlite', '.sqlite3']:
                continue
                
            # Skip files matching gitignore patterns
            if should_ignore(rel_path, gitignore_patterns):
                continue
                
            current_files.add(str(rel_path))
            
            # Calculate checksum
            checksum = calculate_checksum(file_path)
            if not checksum:
                continue
                
            # Check if file is new or modified
            if str(rel_path) not in db_files:
                # New file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                description = describe_file(content)
                
                # Log the creation
                log_interaction(
                    conn,
                    role="interpreter",
                    content=f"New file created: {rel_path}\n\n{description}",
                    layer=0
                )
                
                # Update codebase
                cursor.execute(
                    'INSERT OR REPLACE INTO codebase (filepath, checksum) VALUES (?, ?)',
                    (str(rel_path), checksum)
                )
                
            elif db_files[str(rel_path)] != checksum:
                # Modified file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                description = describe_file(content)
                
                # Log the change
                log_interaction(
                    conn,
                    role="interpreter",
                    content=f"File modified: {rel_path}\n\n{description}",
                    layer=0
                )
                
                # Update codebase
                cursor.execute(
                    'UPDATE codebase SET checksum = ?, last_updated = CURRENT_TIMESTAMP WHERE filepath = ?',
                    (checksum, str(rel_path))
                )
    
    # Check for deleted files
    for filepath in set(db_files.keys()) - current_files:
        # Log the deletion
        log_interaction(
            conn,
            role="interpreter",
            content=f"File deleted: {filepath}",
            layer=0
        )
        
        # Remove from codebase
        cursor.execute('DELETE FROM codebase WHERE filepath = ?', (filepath,))
    
    conn.commit()
    conn.close()

def log_interaction(conn: sqlite3.Connection, role: str, content: str, layer: int = 0, embedding: Optional[bytes] = None):
    """Log an interaction to the database."""
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO interactions (id, uuid, role, content, layer, embedding) VALUES (?, ?, ?, ?, ?, ?)',
        (str(uuid.uuid4()), str(uuid.uuid4()), role, content, layer, embedding)
    )
    conn.commit()

def interpreter(text: str) -> str:
    """
    Detect and execute function calls from agent_tools.py in natural language text.
    
    This function analyzes the input text to find patterns that match function calls
    to any function defined in agent_tools.py, executes them, and returns the results.
    
    Args:
        text: Input text that may contain function calls
        
    Returns:
        str: A string containing the results of all executed function calls in the format:
             "Resultado de ejecutar function_name(args, kwargs): result\n"
             If no functions are found, returns an empty string.
    """
    """
    """
    import re
    import importlib.util
    import inspect
    from typing import List, Dict, Any, Tuple, Optional
    
    # Dynamically import agent_tools to get its functions
    spec = importlib.util.spec_from_file_location("agent_tools", "agent_tools.py")
    agent_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_tools)
    
    # Get all callable functions from agent_tools
    functions = {}
    for name, obj in inspect.getmembers(agent_tools):
        if inspect.isfunction(obj) and not name.startswith('_'):
            sig = inspect.signature(obj)
            functions[name] = {
                'func': obj,
                'params': list(sig.parameters.keys()),
                'param_count': len(sig.parameters)
            }
    
    # Pattern to match function calls with arguments
    function_pattern = r'\b(' + '|'.join(re.escape(f) for f in functions.keys()) + r')\s*\(([^)]*)\)'
    
    # Find all potential function calls
    matches = re.finditer(function_pattern, text, re.IGNORECASE)
    
    calls = []
    for match in matches:
        func_name = match.group(1)
        args_str = match.group(2).strip()
        
        # Skip if the match is part of a larger word (e.g., 'somefunction' in 'somefunctioncall')
        if not re.match(rf'\b{re.escape(func_name)}\b', match.group(0), re.IGNORECASE):
            continue
            
        func_info = functions.get(func_name)
        if not func_info:
            continue
            
        # Try to parse the arguments
        try:
            # This is a simple parser that handles basic cases
            # For a production system, you'd want a more robust parser
            args = []
            kwargs = {}
            
            if args_str:
                # Split arguments, handling strings and nested structures
                current_arg = []
                in_quotes = False
                in_parens = 0
                for char in args_str:
                    if char == '"' or char == "'":
                        in_quotes = not in_quotes
                        current_arg.append(char)
                    elif char == '(' and not in_quotes:
                        in_parens += 1
                        current_arg.append(char)
                    elif char == ')' and not in_quotes:
                        in_parens -= 1
                        current_arg.append(char)
                    elif char == ',' and not in_quotes and in_parens == 0:
                        arg_str = ''.join(current_arg).strip()
                        if '=' in arg_str:
                            k, v = arg_str.split('=', 1)
                            kwargs[k.strip()] = v.strip()
                        else:
                            args.append(arg_str)
                        current_arg = []
                    else:
                        current_arg.append(char)
                
                # Add the last argument
                if current_arg:
                    arg_str = ''.join(current_arg).strip()
                    if arg_str and '=' in arg_str:
                        k, v = arg_str.split('=', 1)
                        kwargs[k.strip()] = v.strip()
                    elif arg_str:
                        args.append(arg_str)
            
            calls.append({
                'function': func_name,
                'args': [a.strip('"\'') for a in args],
                'kwargs': {k: v.strip('"\'') for k, v in kwargs.items()}
            })
            
        except Exception as e:
            print(f"Error parsing arguments for {func_name}: {e}")
            continue
    
    # Also look for natural language mentions of functions
    for func_name, func_info in functions.items():
        # Look for patterns like "function changeFile" or "call changeFile"
        pattern = r'\b(function|call|use|invoke|run|execute)\s+' + re.escape(func_name) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            # Check if we already have this function in our calls
            if not any(call['function'].lower() == func_name.lower() for call in calls):
                calls.append({
                    'function': func_name,
                    'args': [],
                    'kwargs': {}
                })
    
    # If no calls detected, return empty string
    if not calls:
        return ""
    
    results = []
    
    # Execute each detected function call
    for call in calls:
        func_name = call['function']
        args = call['args']
        kwargs = call['kwargs']
        
        try:
            # Get the function object
            func = getattr(agent_tools, func_name, None)
            if not func or not callable(func):
                results.append(f"Error: Function '{func_name}' not found or not callable")
                continue
            
            # Convert string representations of Python literals to actual Python objects
            def parse_arg(arg):
                try:
                    # Try to evaluate the argument as a Python literal
                    return eval(arg, {'__builtins__': None}, {})
                except:
                    # If evaluation fails, return the string as is
                    return arg
            
            # Parse arguments
            parsed_args = [parse_arg(arg) for arg in args]
            parsed_kwargs = {k: parse_arg(v) for k, v in kwargs.items()}
            
            # Call the function with the parsed arguments
            result = func(*parsed_args, **parsed_kwargs)
            
            # Format the function call for output
            args_str = ', '.join([repr(arg) for arg in args])
            if kwargs:
                if args_str:
                    args_str += ', '
                args_str += ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            
            # Add the result to our results list
            results.append(f"Resultado de ejecutar {func_name}({args_str}): {result}")
            
        except Exception as e:
            results.append(f"Error ejecutando {func_name}: {str(e)}")
    
    # Return all results joined by newlines
    return '\n'.join(results)

if __name__ == "__main__":
    # Example usage
    checkChanges(".")