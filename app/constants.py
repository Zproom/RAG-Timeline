"""
Modue for storing constants that are not expected to change
"""

from datetime import datetime
from typing import TypedDict

from torch import Tensor

################################################################
######################## DATABASE PROPS ########################
################################################################

# TODO: update collection name to non testing
COLLECTION_NAME = "News_test"

# length of embeddings from using the model: all-MiniLM-L6-v2
VECTOR_LENGTH = 384

QDRANT_PORT = 6333
SENTENCES_PER_CHUNK = 10

################################################################
########################## MODEL NAME ##########################
################################################################
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
RERANKING_MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L-2-v2"

################################################################
################# GENERATION STEP CONFIG NAME ##################
################################################################

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
NUM_RESOURCES = 5
CHARS_PER_TOKEN = 1 / 4  # estimated and used in lieu of a misc / 4 where needed

################################################################
############## Typed dict to help with type hints ##############
################################################################


class ArticleDict(TypedDict):
    source: str
    date: datetime | None
    title: str
    authors: list[str]
    text: str
    url: str


class ChunkDict(TypedDict):
    text: str
    source: str
    date: str
    title: str
    authors: list[str]
    url: str
    chunk: int
    embeddings: Tensor | None


class GdeltDict(TypedDict):
    url: str
    title: str
    domain: str
    country: str


class QueryResDict(TypedDict):
    score: float
    vector: Tensor
    source: str
    date: datetime | None
    title: str
    authors: list[str]
    text: str
    url: str
    chunk: int
