#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import threading
from typing import Optional, Dict, Any, Callable

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get environment variables
BACKEND_URL = os.getenv('BACKEND_URL')
APP_CLIENT = os.getenv('APP_CLIENT')

# Validate required environment variables
if not all([BACKEND_URL, APP_CLIENT]):
    print("Error: Missing required environment variables in .env file")
    print("Please make sure BACKEND_URL and APP_CLIENT are set")
    sys.exit(1)

# Constants
ACCOUNT_FILE = "account.json"

def load_account() -> Dict[str, Any]:
    """Load account data from account.json"""
    if not os.path.exists(ACCOUNT_FILE):
        return {}
    try:
        with open(ACCOUNT_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_account(data: Dict[str, Any]) -> None:
    """Save account data to account.json"""
    with open(ACCOUNT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def show_main_screen(provider: str, model: str, path: str) -> None:
    """Display the main working screen"""
    print(f"Working on {path} using {model} with {provider} as provider")

class LoadingIndicator:
    def __init__(self, message: str = "Processing"):
        self.message = message
        self.done = False
        self.thread = None

    def __animate(self) -> None:
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while not self.done:
            print(f"\r{chars[i % len(chars)]} {self.message}", end="", flush=True)
            time.sleep(0.1)
            i += 1
        # Clear the line when done
        print("\r" + " " * (len(self.message) + 2) + "\r", end="", flush=True)

    def __enter__(self):
        self.done = False
        self.thread = threading.Thread(target=self.__animate, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done = True
        if self.thread:
            self.thread.join()

def login() -> Optional[str]:
    """
    Handle user login by making an API call to the backend.
    Returns token if successful, None otherwise.
    """
    import requests
    from getpass import getpass
    
    print("\nLogin")
    print("-----")
    email = input("Email: ")
    password = getpass("Password: ")
    
    try:
        with LoadingIndicator("Authenticating..."):
            response = requests.post(
                f"{BACKEND_URL}/auth/sign_in",
                json={
                    "email": email,
                    "password": password,
                    "app_client": APP_CLIENT
                },
                headers={"Content-Type": "application/json"}
            )
        
        if response.status_code == 200:
            data = response.json()
            if 'token' in data:
                return data['token']
            print("Error: No token in response")
        else:
            error_msg = response.json().get('error', 'Unknown error occurred')
            print(f"Login failed: {error_msg}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
    except json.JSONDecodeError:
        print("Error: Invalid response from server")
    
    return None

def show_0x255_welcome() -> None:
    """Display welcome message for 0x255 provider"""
    print("\n" + "="*60)
    print(" 0x255 CLOUD PLATFORM ".center(60, "#"))
    print("="*60)
    print("\n Welcome to 0x255, a private cloud platform that provides")
    print(" computational resources and infrastructure to access various")
    print(" state-of-the-art language models (LLMs) and embedding models.")
    print(" For more information, visit: https://0x255.cc/kybe_cli")
    print("\n 🚀 Benefits of using 0x255 cloud infrastructure:")
    print(" • Zero local resource usage - models run on our cloud servers")
    print(" • No heavy GPU/CPU requirements on your machine")
    print(" • Access to powerful models through optimized cloud infrastructure")
    print(" • High-performance computing resources for faster inference")
    print(" • Scalable infrastructure that grows with your needs")
    print("\n 🔒 Your privacy is important to us:")
    print(" • We do not store any of your model interactions")
    print(" • We do not share your data with third parties")
    print(" • We only store information necessary for authentication")
    print(" • All processing happens securely in our cloud environment")
    print("\n Please log in to continue using KybeCLI with 0x255 resources.")
    print("\n" + "="*60 + "\n")

def handle_0x255_provider() -> bool:
    """
    Handle 0x255 provider authentication flow.
    Returns True if authenticated (either with existing or new token), False otherwise.
    """
    account = load_account()
    
    # Si ya existe un token, lo consideramos válido y continuamos
    if 'token' in account and account['token']:
        return True
        
    # Si llegamos aquí, no hay token o está vacío
    show_0x255_welcome()
    print("\n 1. Log in")
    print(" 2. Sign up (coming soon)")
    
    while True:
        choice = input("\n Select an option (1-2): ")
        
        if choice == '1':
            token = login()
            if token:
                account['token'] = token
                save_account(account)
                return True
            return False
        elif choice == '2':
            print("\n  Registration will be available soon.")
            # Show menu again
            show_0x255_welcome()
            print("\n 1. Log in")
            print(" 2. Sign up (coming soon)")
        else:
            print("  ❌ Invalid option. Please select 1 or 2.")
    
    return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Kybe CLI Tool')
    parser.add_argument('--provider', required=True, help='Provider (ollama or 0x255)')
    parser.add_argument('--model', required=True, help='Model to use')
    parser.add_argument('--path', required=True, help='Path to work on')
    
    args = parser.parse_args()
    
    # Validate provider
    if args.provider not in ['ollama', '0x255']:
        print("Error: Provider must be either 'ollama' or '0x255'")
        sys.exit(1)
    
    # Handle provider-specific logic
    if args.provider == '0x255':
        if not handle_0x255_provider():
            print("Authentication failed. Exiting...")
            sys.exit(1)
    
    # If we get here, either provider is ollama or 0x255 auth was successful
    show_main_screen(args.provider, args.model, args.path)

if __name__ == "__main__":
    main()
