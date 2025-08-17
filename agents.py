import re
import inspect
import requests
from typing import List, Dict, Optional, Callable, Any, Union

class Agent:
    def __init__(self, model: str, system_prompt: str, max_interactions: int = 10, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the Agent with model, system prompt, and interaction history settings.
        
        Args:
            model: The name of the Ollama model to use (e.g., 'llama2', 'mistral')
            system_prompt: The system prompt to guide the AI's behavior
            max_interactions: Maximum number of user-assistant interactions to keep in context
            ollama_base_url: Base URL for the Ollama API
        """
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.max_interactions = max_interactions
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "Ok"}
        ]
    
    def _add_interaction(self, role: str, content: str) -> None:
        """Add an interaction to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        
        # Trim old interactions if we exceed max_interactions
        # We keep the system prompt interaction and then the most recent interactions
        if len(self.conversation_history) > (self.max_interactions * 2) + 2:  # +2 for initial system prompt
            # Keep system prompt and its response, then keep most recent interactions
            self.conversation_history = (
                self.conversation_history[:2] +  # Keep system prompt and "Ok"
                self.conversation_history[-(self.max_interactions * 2):]  # Keep most recent interactions
            )
    
    def interact(self, text: str) -> str:
        """
        Interact with the AI agent.
        
        Args:
            text: User's input text
            
        Returns:
            str: The AI's response text
        """
        # Add user message to history
        self._add_interaction("user", text)
        
        try:
            # Prepare the API request to Ollama
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": self.conversation_history,
                    "stream": False
                }
            )
            response.raise_for_status()
            
            # Get the AI's response
            ai_response = response.json()["message"]["content"]
            
            # Add AI's response to history
            self._add_interaction("assistant", ai_response)
            
            return ai_response
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error communicating with Ollama: {str(e)}"
            self._add_interaction("system", error_msg)
            return error_msg
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return self.conversation_history.copy()


class ComposedAgent:
    def __init__(self, model: str, system_prompt: str, functions: List[Callable], 
                 max_interactions: int = 32, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the ComposedAgent with model, system prompt, and functions.
        
        Args:
            model: The name of the Ollama model to use
            system_prompt: The base system prompt for the agent
            functions: List of callable functions the agent can use
            max_interactions: Maximum number of user-assistant interactions to keep in context
            ollama_base_url: Base URL for the Ollama API
        """
        self.functions = {func.__name__: func for func in functions}
        self.agent = Agent(
            model=model,
            system_prompt=self._build_system_prompt(system_prompt, functions),
            max_interactions=max_interactions,
            ollama_base_url=ollama_base_url
        )
    
    def _build_system_prompt(self, base_prompt: str, functions: List[Callable]) -> str:
        """Build the system prompt including function documentation."""
        function_docs = []
        for func in functions:
            doc = inspect.getdoc(func) or "No documentation available."
            function_docs.append(f"Function: {func.__name__}\n{doc}\n")
        
        functions_list = "\n".join([f"- {name}" for name in self.functions.keys()])
        
        # Build the system prompt with proper string formatting
        function_docs_str = "\n\n".join(function_docs)
        return f"""You are an AI assistant that interacts with users through an intermediary system. 
Your responses will be interpreted to detect and execute function calls.

{base_prompt}

AVAILABLE FUNCTIONS:
{functions_list}

FUNCTION DOCUMENTATION:
{function_docs_str}

ANALYTICAL THINKING CYCLE (ITERATIVE):
1. ANALYSIS: Evaluate the situation and identify what's needed
2. SOLUTION: Plan your approach and select tools
3. EXECUTION: Call functions using Python syntax (e.g., function_name(params))
4. EVALUATION: Analyze results and decide next steps
5. CONTINUITY: Proceed to next step or iterate

RULES:
- Your internal thinking is invisible to users unless you use show_user_response()
- Always end with show_user_response() to send final output
- Function calls must be on their own line
- Wait for function results before proceeding
- Be concise and focused in your reasoning

Remember: Users only see what you pass to show_user_response()
"""
    
    def _extract_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract function calls from the agent's response."""
        # Look for patterns like function_name(arg1=value1, arg2=value2)
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)'
        matches = re.finditer(pattern, text)
        
        calls = []
        for match in matches:
            func_name = match.group(1)
            if func_name in self.functions:
                # Convert string arguments to a dictionary
                args_str = match.group(2).strip()
                args = {}
                if args_str:
                    # Simple parsing of keyword arguments
                    for arg in args_str.split(','):
                        if '=' in arg:
                            key, value = arg.split('=', 1)
                            args[key.strip()] = value.strip('"\' ')
                calls.append({"name": func_name, "args": args})
        return calls
    
    def _execute_function(self, func_name: str, args: Dict[str, Any]) -> str:
        """Execute a function with the given arguments."""
        try:
            if func_name not in self.functions:
                return f"Error: Function '{func_name}' not found."
            
            func = self.functions[func_name]
            return str(func(**args))
        except Exception as e:
            return f"Error executing {func_name}: {str(e)}"
    
    def interact(self, user_message: str, max_iterations: int = 64) -> str:
        """
        Interact with the agent in a loop until a final response is ready.
        
        Args:
            user_message: The user's input message
            
        Returns:
            str: The final response to show the user
        """
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get agent's response
            response = self.agent.interact(user_message)
            
            # Check for function calls
            function_calls = self._extract_function_calls(response)
            
            if not function_calls:
                # If no function calls, return the response as is
                return response
            
            # Execute all function calls
            results = []
            for call in function_calls:
                # Format the function call with arguments
                args_str = ", ".join([f'{k}={repr(v)}' for k, v in call["args"].items()])
                func_call_str = f"{call['name']}({args_str})"
                
                # Execute the function and get result
                result = self._execute_function(call["name"], call["args"])
                
                # Format the result message
                results.append(f"Function Executed: {func_call_str}\nResult: {result}")
                
                # If this was a show_user_response call, return its message
                if call["name"] == "show_user_response":
                    return call["args"].get("message", "")
            
            # Prepare for next iteration with results
            user_message = "\n\n".join(results)
        
        return "Maximum number of iterations reached. Please try again with a more specific query."


class EmbeddingAgent:
    def __init__(self, model: str = "mxbai-embed-large", 
                 ollama_base_url: str = "http://localhost:11434",
                 normalize: bool = False):
        """
        Initialize the EmbeddingAgent.
        
        Args:
            model: The name of the Ollama model to use for embeddings
            ollama_base_url: Base URL for the Ollama API
            normalize: Whether to normalize the embedding vectors to unit length
        """
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.normalize = normalize
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize a vector to unit length."""
        import math
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector
        return [x / norm for x in vector]
    
    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            List[float]: The embedding vector
            
        Raises:
            Exception: If there's an error generating the embedding
        """
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            
            embedding = response.json().get("embedding", [])
            
            if self.normalize:
                embedding = self._normalize_vector(embedding)
                
            return embedding
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error generating embedding: {str(e)}")
        except (KeyError, ValueError) as e:
            raise Exception(f"Error parsing embedding response: {str(e)}")
