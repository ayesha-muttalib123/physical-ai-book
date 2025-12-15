import numpy as np
from typing import List
import logging
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generate embeddings for text chunks using a pre-trained model
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator with a pre-trained model
        Using a lightweight model suitable for the free tier
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts)

        # Convert to list of lists (required by Qdrant)
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        logger.info("Embeddings generated successfully")

        return embeddings_list

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        """
        return self.generate_embeddings([text])[0]

if __name__ == "__main__":
    # Example usage
    generator = EmbeddingGenerator()
    sample_texts = [
        "Physical AI combines artificial intelligence with physical systems",
        "Robotics requires understanding of sensors and actuators",
        "Humanoid robots need sophisticated locomotion algorithms"
    ]

    embeddings = generator.generate_embeddings(sample_texts)
    logger.info(f"Generated {len(embeddings)} embeddings")
    logger.info(f"Embedding dimension: {len(embeddings[0])}")