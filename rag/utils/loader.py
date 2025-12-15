import os
import json
from pathlib import Path
from typing import List, Dict
import logging
import frontmatter  # For parsing markdown with YAML frontmatter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_markdown_files(docs_dir: str) -> List[Dict[str, str]]:
    """
    Read all markdown files from the docs directory
    """
    docs_path = Path(docs_dir)
    documents = []

    for md_file in docs_path.glob("*.md"):
        logger.info(f"Reading file: {md_file.name}")

        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse markdown with frontmatter
        post = frontmatter.loads(content)

        documents.append({
            "id": md_file.stem,
            "content": post.content,  # Content without frontmatter
            "metadata": post.metadata,  # Frontmatter metadata
            "source": str(md_file),
            "title": post.metadata.get('title', md_file.stem.replace('-', ' ').title())
        })

    logger.info(f"Loaded {len(documents)} documents")
    return documents

def read_json_file(file_path: str) -> List[Dict[str, str]]:
    """
    Read documents from a JSON file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def read_text_file(file_path: str) -> str:
    """
    Read content from a text file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

def scan_directory_for_docs(dir_path: str, extensions: List[str] = ['.md', '.txt']) -> List[str]:
    """
    Scan directory for files with specified extensions
    """
    dir_path = Path(dir_path)
    files = []

    for ext in extensions:
        files.extend(dir_path.rglob(f"*{ext}"))

    return [str(f) for f in files]

def load_processed_chunks(input_path: str = "./vector_store/processed_chunks.json") -> List[Dict[str, Any]]:
    """
    Load processed chunks from JSON file
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} processed chunks from {input_path}")
        return chunks
    except FileNotFoundError:
        logger.error(f"Processed chunks file not found at {input_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading processed chunks: {str(e)}")
        return []

if __name__ == "__main__":
    # Example usage
    if os.path.exists("../docusaurus/docs"):
        docs = read_markdown_files("../docusaurus/docs")
        logger.info(f"Found {len(docs)} documents in docs directory")
    else:
        logger.info("Docs directory not found, skipping example")