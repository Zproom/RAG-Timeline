"""
Modue for storing constants that are not expected to change
"""
################################################################
######################## DATABASE PROPS ########################
################################################################

# TODO: update collection name to non testing
COLLECTION_NAME = "News_test"

# length of embeddings from using the model: all-MiniLM-L6-v2
VECTOR_LENGTH = 384

QDRANT_PORT = 6333

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
CHARS_PER_TOKEN = 1 / 4 # estimated and used in lieu of a misc / 4 where needed
