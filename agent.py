import requests
import json
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class Agent:
    def __init__(self, name, system_prompt, model='deepseek-r1:latest', server_ip="http://localhost:11434", provider="ollama"):
        """
        Constructor de la clase Agent
        
        Args:
            name (str): Nombre del agente
            model (str): Nombre del modelo a usar
            system_prompt (str): Prompt del sistema para el agente
            server_ip (str): IP del servidor (default: http://localhost:11434)
            provider (str): Proveedor del modelo (default: ollama)
        """
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.server_ip = server_ip.rstrip('/')  # Quitar "/" al final si existe
        self.provider = provider
    
    def _load_session_token(self):
        """Carga el token de sesión desde account.json"""
        try:
            with open('account.json', 'r') as f:
                account_data = json.load(f)
                return account_data.get('token', '')
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None
    
    def interact(self, text, conversation, max_tokens=None):
        """
        Interactúa con el modelo
        
        Args:
            text (str): Texto del usuario para enviar al modelo
            conversation (list): Lista de diccionarios con keys 'role' y 'content'
            max_tokens (int, optional): Número máximo de tokens para la respuesta
        
        Returns:
            dict: Diccionario con keys 'role' y 'content' con la respuesta del agente
        """
        if self.provider == "0x255":
            return self._interact_backend(text, conversation, max_tokens)
        else:
            return self._interact_ollama(text, conversation, max_tokens)
    
    def _interact_backend(self, text, conversation, max_tokens):
        """Interactúa con el backend personalizado"""
        backend_url = os.getenv('BACKEND_URL')
        if not backend_url:
            return {
                "role": "assistant",
                "content": "Error: BACKEND_URL no encontrada en .env"
            }
        
        token = self._load_session_token()
        if not token:
            return {
                "role": "assistant",
                "content": "Error: Token de sesión no encontrado en account.json"
            }
        
        # Preparar conversación completa incluyendo el nuevo mensaje
        full_conversation = conversation + [{"role": "user", "content": text}]
        
        url = f"{backend_url.rstrip('/')}/agent/interact"
        headers = {"authorization": token}
        payload = {
            "system_prompt": self.system_prompt,
            "conversation": full_conversation,
            "model": self.model
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            return result.get("response", {
                "role": "assistant",
                "content": "Error: Respuesta vacía del backend"
            })
            
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
    
    def _interact_ollama(self, text, conversation, max_tokens):
        """Interactúa con Ollama (código original)"""
        url = f"{self.server_ip}/api/chat"
        
        messages = []
        
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        messages.extend(conversation)
        messages.append({
            "role": "user",
            "content": text
        })
        
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