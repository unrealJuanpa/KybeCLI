import requests
import json

class Agent:
    def __init__(self, name, system_prompt, model='deepseek-r1:latest', server_url="http://localhost:11434"):
        """
        Constructor de la clase Agent
        
        Args:
            name (str): Nombre del agente
            model (str): Nombre del modelo a usar
            system_prompt (str): Prompt del sistema para el agente
            server_ip (str): IP del servidor (default: http://localhost:11434)
        """
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.server_url = server_url.rstrip('/')  # Quitar "/" al final si existe
    
    def interact(self, conversation, max_tokens=None):
        """
        Interactúa con el modelo
        
        Args:
            conversation (list): Lista de diccionarios con keys 'role' y 'content'
            max_tokens (int, optional): Número máximo de tokens para la respuesta
        
        Returns:
            dict: Diccionario con keys 'role' y 'content' con la respuesta del agente
        """
        return self._interact_ollama(conversation, max_tokens)
    
    def _interact_ollama(self, conversation, max_tokens):
        """Interactúa con Ollama (código original)"""
        url = f"{self.server_url}/api/chat"
        
        messages = [{ "role": "user", "content": self.system_prompt}, {"role": "assistant", "content": "Ok."}] if self.system_prompt else []
        
        messages.extend(conversation)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        
        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            assistant_message = result.get("message", {})
            
            return {
                "role": "assistant",
                "content": assistant_message.get("content", "")
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "role": "assistant",
                "content": f"Error de conexión: {str(e)}"
            }
        except json.JSONDecodeError as e:
            return {
                "role": "assistant", 
                "content": f"Error al decodificar JSON: {str(e)}"
            }
        except Exception as e:
            return {
                "role": "assistant",
                "content": f"Error inesperado: {str(e)}"
            }