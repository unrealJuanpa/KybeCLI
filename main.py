#!/usr/bin/env python3
"""
KybeCLI - A command-line interface for codebase interaction using AI agents.
"""
import os
import sys
import time
import argparse
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from agents import ComposedAgent, EmbeddingAgent
from utils import CodebaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='KybeCLI - Interact with your codebase using AI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='Base path to the codebase directory',
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default='llama3.2:latest',
        help='Name of the Ollama model to use for the agent',
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='mxbai-embed-large',
        help='Name of the Ollama model to use for embeddings',
    )
    
    # Interaction settings
    parser.add_argument(
        '--max-interactions',
        type=int,
        default=32,
        help='Maximum number of interactions to keep in context',
    )
    parser.add_argument(
        '--top-files',
        type=int,
        default=8,
        help='Number of top similar files to retrieve',
    )
    
    # Feature toggles
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='Normalize embeddings (recommended)',
    )
    parser.add_argument(
        '--no-normalize',
        dest='normalize',
        action='store_false',
        help='Disable embedding normalization',
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        default=True,
        help='Stream responses in real-time',
    )
    parser.add_argument(
        '--no-stream',
        dest='stream',
        action='store_false',
        help='Disable streaming responses',
    )
    
    return parser.parse_args()

def search_codebase(manager: CodebaseManager, query: str, top_n: int) -> str:
    """Search the codebase for relevant files."""
    results = manager.search_similar(query, top_n=top_n)
    if not results:
        return "No relevant files found."
    
    response = ["Found the following relevant files:"]
    for i, result in enumerate(results, 1):
        response.append(f"{i}. {result['path']} (similarity: {result['similarity']:.2f})")
        response.append(f"   {result['description'][:200]}...")
    
    return "\n".join(response)

def main():
    """Main entry point for KybeCLI."""
    args = parse_args()
    
    # Validate paths
    base_path = Path(args.path).expanduser().absolute()
    if not base_path.exists():
        logger.error(f"Path does not exist: {base_path}")
        return 1
    if not base_path.is_dir():
        logger.error(f"Path is not a directory: {base_path}")
        return 1
    
    logger.info(f"Initializing KybeCLI with model: {args.model}")
    logger.info(f"Codebase path: {base_path}")
    
    # Initialize components
    try:
        # Initialize embedding agent and codebase manager
        logger.info(f"Initializing embedding model: {args.embedding_model}")
        embedding_agent = EmbeddingAgent(
            model=args.embedding_model,
            normalize=args.normalize
        )
        
        logger.info("Initializing codebase manager...")
        codebase_manager = CodebaseManager(str(base_path))
        
        # Define available functions for the agent
        def search_files(query: str) -> str:
            """Search the codebase for files relevant to the query."""
            return search_codebase(codebase_manager, query, args.top_files)
        
        def show_user_response(message: str) -> str:
            """Show a message to the user and end the interaction."""
            return message
        
        # Initialize the agent with available functions
        system_prompt = """
        You are Kybe, a helpful AI assistant specialized in software development.
        You can search the codebase and provide information about the project.
        Always be concise and to the point in your responses.
        """
        
        agent = ComposedAgent(
            model=args.model,
            system_prompt=system_prompt,
            functions=[search_files, show_user_response],
            max_interactions=args.max_interactions
        )
        
        # Main interaction loop
        logger.info("Initialization complete. Type 'exit' to quit.")
        print("\n" + "=" * 50)
        print("KybeCLI - Type your query or 'exit' to quit")
        print("=" * 50 + "\n")
        
        while True:
            try:
                # Update codebase before each query
                logger.info("Scanning codebase for changes...")
                start_time = time.time()
                codebase_manager.scan_and_update()
                logger.info(f"Codebase scan completed in {time.time() - start_time:.2f}s")
                
                # Get user input
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ('exit', 'quit', 'q'):
                    break
                
                if not user_input:
                    continue
                
                # Process the query
                logger.info("Processing query...")
                response = agent.interact(user_input)
                
                # Stream the response if enabled
                if args.stream:
                    print("\nAssistant: ", end='', flush=True)
                    for char in response:
                        print(char, end='', flush=True)
                        time.sleep(0.01)  # Small delay for better readability
                    print("\n")
                else:
                    print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit or press Ctrl+C again to force exit.")
                try:
                    time.sleep(1)  # Small delay to prevent accidental double Ctrl+C
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                continue
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
