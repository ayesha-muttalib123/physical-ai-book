import logging
from typing import List, Dict
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextChunker:
    """
    Utility class for chunking text with overlap
    """
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

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
                "end_pos": min(start + self.chunk_size, len(text))
            })

            # Move start position forward, accounting for overlap
            start = end - self.overlap

            # Handle edge case where remaining text is shorter than chunk_size
            if len(text) - start < self.chunk_size:
                if start < len(text):
                    # Add the remaining text as the last chunk
                    chunks.append({
                        "text": text[start:].strip(),
                        "start_pos": start,
                        "end_pos": len(text)
                    })
                break

        return chunks

def clean_text(text: str) -> str:
    """
    Clean and normalize text content
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters

    return text.strip()

def extract_title_from_content(content: str) -> str:
    """
    Extract title from markdown content (first H1 or first line)
    """
    lines = content.split('\n')

    # Look for markdown H1
    for line in lines:
        if line.startswith('# '):
            return line[2:].strip()

    # If no H1 found, use first non-empty line
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped[:100]  # Limit to first 100 chars

    return "Untitled Document"

if __name__ == "__main__":
    # Example usage
    sample_text = "This is a sample text. It has multiple sentences. " * 50  # Long text for testing
    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk_text(sample_text)

    logger.info(f"Split text into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Print first 3 chunks as example
        logger.info(f"Chunk {i+1}: {len(chunk['text'])} chars")