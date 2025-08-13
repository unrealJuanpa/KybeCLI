from pathlib import Path
from typing import Optional
from agent import Agent

# Initialize the code editing agent
code_editor_agent = Agent(
    name="CodeEditor",
    system_prompt="""You are an expert code editor. Your task is to modify code based on the requested changes.
    
    IMPORTANT RULES:
    1. You will receive the current file content and a description of changes to make.
    2. You must return ONLY the complete modified code, with NO additional text, explanations, or markdown formatting.
    3. Do not include any preamble, comments about the changes, or code block markers.
    4. If the changes are not applicable or would break the code, return the original content unchanged.
    5. Preserve the original code style, indentation, and formatting.
    
    Your response will be directly written to the file, so it must be valid, complete code.
    """,
    provider="ollama",
    model="llama3"
)

def changeFile(file_path: str, change_description: str) -> str:
    """
    Modify a file based on a natural language description of changes.
    
    This function reads the specified file, sends its content along with the change description
    to an AI agent, and writes the modified content back to the file. The AI agent is instructed
    to only return the modified code without any additional text or explanations.
    
    Args:
        file_path: Path to the file to be modified (can be relative or absolute)
        change_description: Natural language description of the changes to make
        
    Returns:
        str: 'Done' if successful, or an error message if something went wrong
        
    Example:
        >>> changeFile("src/utils.py", "Add error handling to the process_data function")
        'Done'
    """
    try:
        path = Path(file_path).resolve()
        
        # Read the current file content
        with open(path, 'r', encoding='utf-8') as f:
            current_content = f.read()
        
        # Get the modified content from the AI agent
        prompt = f"""Current file content:
```
{current_content}
```

Make the following changes to this file (return ONLY the complete modified code with no additional text):
{change_description}"""
        
        response = code_editor_agent.interact(prompt)
        modified_content = response.get("content", "").strip()
        
        # Clean up the response - remove markdown code blocks if present
        if modified_content.startswith('```') and '\n' in modified_content:
            # Extract content between the first and last backticks
            modified_content = modified_content.split('\n', 1)[1].rsplit('```', 1)[0]
        
        # Write the modified content back to the file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
            
        return "Done"
        
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except PermissionError:
        return f"Error: Permission denied when trying to modify {file_path}"
    except Exception as e:
        return f"Error: {str(e)}"