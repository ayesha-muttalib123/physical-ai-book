from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from qdrant_config import QdrantSetup

load_dotenv()

class RAGChatbot:
    def __init__(self):
        # Initialize OpenAI model
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500
        )

        # Initialize Qdrant setup
        self.qdrant_setup = QdrantSetup()

        # Get the retriever for the physical AI book collection
        self.retriever = self.qdrant_setup.get_retriever("physical_ai_book", k=4)

        # Create the retrieval chain
        self.retrieval_chain = self._create_retrieval_chain()

    def _create_retrieval_chain(self):
        """Create a retrieval chain for RAG"""
        template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer as concise as possible, unless the question specifically asks for detail.
        If the context is not relevant to the question, please provide a general response based on your knowledge
        but indicate that the information is not from the Physical AI & Humanoid Robotics book.

        Context: {context}

        Question: {question}

        Helpful Answer:"""

        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        return qa_chain

    def get_response(self, query: str) -> dict:
        """Get response from the RAG system"""
        try:
            result = self.retrieval_chain.invoke({"query": query})

            # Extract sources if available
            sources = []
            if hasattr(result, 'get') and "source_documents" in result:
                sources = [doc.metadata.get("source", "") for doc in result["source_documents"]]

            return {
                "response": result["result"],
                "sources": sources
            }
        except Exception as e:
            return {
                "response": f"Error processing your query: {str(e)}",
                "sources": []
            }