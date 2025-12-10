from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()

class QdrantSetup:
    def __init__(self):
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()

    def create_collection(self, collection_name: str):
        """Create a new collection in Qdrant"""
        from qdrant_client.http import models

        # Check if collection already exists
        try:
            self.client.get_collection(collection_name)
            print(f"Collection '{collection_name}' already exists")
            return
        except:
            # Create new collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # OpenAI embedding size
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created collection '{collection_name}'")

    def add_documents(self, collection_name: str, documents: list):
        """Add documents to the collection"""
        # Convert to Langchain documents if needed
        langchain_docs = []
        for doc in documents:
            if isinstance(doc, dict):
                langchain_docs.append(Document(
                    page_content=doc.get("content", ""),
                    metadata=doc.get("metadata", {})
                ))
            else:
                langchain_docs.append(doc)

        # Add documents to Qdrant
        Qdrant.from_documents(
            langchain_docs,
            self.embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=collection_name
        )

    def get_retriever(self, collection_name: str, k: int = 4):
        """Get a retriever for the collection"""
        qdrant = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=self.embeddings
        )
        return qdrant.as_retriever(search_kwargs={"k": k})