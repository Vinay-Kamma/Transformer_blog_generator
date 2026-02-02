import os
from pathlib import Path

def create_project_structure():
    """Create the project directory structure for transformer-blog-generator."""
    
    base_path = Path(__file__).parent
    
    # Define directory structure
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "training",
        "inference",
        "tests",
        "scripts",
        "checkpoints",
        "logs",
    ]
    
    # Create directories
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")
    
    # Create __init__.py files for Python packages
    for package in ["training", "inference", "tests", "scripts"]:
        init_file = base_path / package / "__init__.py"
        init_file.touch(exist_ok=True)
        print(f"Created: {init_file}")
    
    # Create .gitkeep files to track empty directories
    for directory in ["data/raw", "data/processed", "models", "checkpoints", "logs"]:
        gitkeep = base_path / directory / ".gitkeep"
        gitkeep.touch(exist_ok=True)
    
    print("\nProject structure created successfully!")

if __name__ == "__main__":
    create_project_structure()