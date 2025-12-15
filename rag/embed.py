import numpy as np
from typing import List, Dict, Any
import logging
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
import uuid
import asyncio
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

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

class VectorStoreManager:
    """
    Manage interactions with Qdrant vector store
    """
    def __init__(self, collection_name: str = "physical_ai_book"):
        """
        Initialize connection to Qdrant
        """
        # Get Qdrant credentials from environment
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url:
            # Connect to cloud instance
            self.client = qdrant_client.QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
        else:
            # Connect to local instance (for development)
            self.client = qdrant_client.QdrantClient(host="localhost", port=6333)

        self.collection_name = collection_name
        self.embedding_generator = EmbeddingGenerator()

    def create_collection(self, recreate: bool = False):
        """
        Create the collection in Qdrant with appropriate configuration
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if collection_exists and not recreate:
                logger.info(f"Collection {self.collection_name} already exists")
                return

            if collection_exists and recreate:
                logger.info(f"Dropping existing collection {self.collection_name}")
                self.client.delete_collection(collection_name=self.collection_name)

            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_generator.embedding_dim,
                    distance=models.Distance.COSINE
                )
            )

            logger.info(f"Created collection {self.collection_name} with dimension {self.embedding_generator.embedding_dim}")

        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Add document chunks to the vector store
        """
        if not chunks:
            logger.warning("No chunks to add to vector store")
            return

        logger.info(f"Adding {len(chunks)} chunks to vector store")

        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)

        # Prepare points for insertion
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=i,  # Using sequential IDs
                vector=embedding,
                payload={
                    "id": chunk['id'],
                    "text": chunk['text'],
                    "source_document": chunk['source_document'],
                    "source_path": chunk['source_path'],
                    "title": chunk['title'],
                    "start_pos": chunk['start_pos'],
                    "end_pos": chunk['end_pos']
                }
            )
            points.append(point)

        # Insert points into collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Successfully added {len(points)} points to collection {self.collection_name}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents to the query
        """
        logger.info(f"Searching for: {query}")

        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]

        # Perform search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        # Format results
        results = []
        for hit in search_results:
            results.append({
                "id": hit.payload.get("id"),
                "content": hit.payload.get("text"),
                "score": hit.score,
                "source_document": hit.payload.get("source_document"),
                "source_path": hit.payload.get("source_path"),
                "title": hit.payload.get("title")
            })

        logger.info(f"Found {len(results)} results for query")
        return results

def load_and_process_documents():
    """
    Load processed chunks and add them to the vector store
    """
    import json

    # Load processed chunks
    with open("./vector_store/processed_chunks.json", 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from processed file")

    # Initialize vector store manager
    vsm = VectorStoreManager()

    # Create collection (recreate if needed)
    vsm.create_collection(recreate=False)

    # Add documents to vector store
    vsm.add_documents(chunks)

    logger.info("Documents successfully added to vector store")

if __name__ == "__main__":
    logger.info("Starting embedding and vector store process...")

    # This would typically be called after running ingest.py
    try:
        load_and_process_documents()
        logger.info("Embedding and vector store process completed successfully!")
    except FileNotFoundError:
        logger.error("Processed chunks file not found. Please run ingest.py first.")
    except Exception as e:
        logger.error(f"Error in embedding process: {str(e)}")