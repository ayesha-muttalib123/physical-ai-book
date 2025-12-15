import os
import json
from pathlib import Path
from typing import List, Dict
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

        documents.append({
            "id": md_file.stem,
            "content": content,
            "source": str(md_file),
            "title": md_file.stem.replace('-', ' ').title()
        })

    logger.info(f"Loaded {len(documents)} documents")
    return documents

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict[str, str]]:
    """
    Split text into overlapping chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If this is not the last chunk, include overlap
        if end < len(text):
            # Try to break at sentence boundary if possible
            chunk_end = end
            for i in range(end, min(end + 50, len(text))):
                if text[i] in '.!?':
                    chunk_end = i + 1
                    break
            chunk_text = text[start:chunk_end]
        else:
            chunk_text = text[start:]

        chunks.append({
            "text": chunk_text.strip(),
            "start_pos": start,
            "end_pos": min(start + chunk_size, len(text))
        })

        # Move start position forward, accounting for overlap
        start = end - overlap

        # Handle edge case where remaining text is shorter than chunk_size
        if len(text) - start < chunk_size:
            if start < len(text):
                # Add the remaining text as the last chunk
                chunks.append({
                    "text": text[start:].strip(),
                    "start_pos": start,
                    "end_pos": len(text)
                })
            break

    return chunks

def process_documents(docs_dir: str, output_dir: str):
    """
    Process documents: read, chunk, and save
    """
    # Read documents
    documents = read_markdown_files(docs_dir)

    # Process each document
    processed_chunks = []
    chunk_id_counter = 0

    for doc in documents:
        logger.info(f"Processing document: {doc['id']}")

        # Chunk the document
        chunks = chunk_text(doc['content'])

        for chunk in chunks:
            processed_chunk = {
                "id": f"{doc['id']}_chunk_{chunk_id_counter}",
                "text": chunk['text'],
                "source_document": doc['id'],
                "source_path": doc['source'],
                "title": doc['title'],
                "start_pos": chunk['start_pos'],
                "end_pos": chunk['end_pos']
            }
            processed_chunks.append(processed_chunk)
            chunk_id_counter += 1

    # Save processed chunks to JSON file
    output_path = Path(output_dir) / "processed_chunks.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(processed_chunks)} chunks to {output_path}")
    logger.info(f"Processing complete!")

    return processed_chunks

if __name__ == "__main__":
    # Define paths
    DOCS_DIR = "../docusaurus/docs"  # Relative to rag directory
    OUTPUT_DIR = "./vector_store"

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process documents
    logger.info("Starting document processing...")
    chunks = process_documents(DOCS_DIR, OUTPUT_DIR)

    logger.info(f"Processed {len(chunks)} chunks from textbook content")