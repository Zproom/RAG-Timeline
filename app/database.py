"""
Module for storing methods and classes for interfacing with the vector database Qdrant
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, QueryResponse
from app.constants import VECTOR_LENGTH, COLLECTION_NAME, QDRANT_PORT
from app.exc import DbError
from app.log import app_logger

class Vector_DB:

    def __init__(self) -> None:
        app_logger.debug("Attempting to connect to Qdrant client...")

        try:
            self.client = QdrantClient(host="localhost", port=QDRANT_PORT)
        except:
            app_logger.critical(f"Unable to connect to Qdrant Client on port {QDRANT_PORT}")
            DbError("Connection FAILED! Unable to connect to Qdrant Client")
        
        app_logger.debug("Success!")

        if not self.client.collection_exists(COLLECTION_NAME):
            app_logger.debug(f"Collection does not exist! Creating new collection: {COLLECTION_NAME}")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_LENGTH, distance=Distance.COSINE)
            )
    
    @staticmethod
    def is_running() -> bool:
        """Checks that the qdrant client is running"""

        try: 
            QdrantClient(host="localhost", port=QDRANT_PORT)
        except:
            return False
        
        return True
            
