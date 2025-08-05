"""
Module for storing methods and classes for interfacing with the vector database Qdrant
"""

from __future__ import annotations

import re
from uuid import uuid4

import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models.models import QueryResponse
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchValue,
                                  PointStruct, VectorParams)
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English

from app.constants import (COLLECTION_NAME, QDRANT_PORT, SENTENCES_PER_CHUNK,
                           VECTOR_LENGTH, ArticleDict, ChunkDict, QueryResDict)
from app.exc import DbError
from app.log import app_logger
from app.utils import create_embedding_model, format_date


def split_list(input_list: list[str], max_item_count: int) -> list[list[str]]:
    """
    Splits a list of strings into a seperate lists with specified maximum number of items
    """
    return [
        input_list[i : i + max_item_count]
        for i in range(0, len(input_list), max_item_count)
    ]


class Vector_DB:

    def __init__(
        self,
        embedding_model: SentenceTransformer | None = None,
        max_chunks_size: int | None = None,
    ) -> None:
        app_logger.debug("Attempting to connect to Qdrant client...")

        self.embedding_model = embedding_model or create_embedding_model()
        self.max_chunks_size = (
            max(0, max_chunks_size)
            if max_chunks_size is not None
            else SENTENCES_PER_CHUNK
        )

        try:
            self.client = QdrantClient(host="localhost", port=QDRANT_PORT)
        except:
            app_logger.critical(
                f"Unable to connect to Qdrant Client on port {QDRANT_PORT}"
            )
            DbError("Connection FAILED! Unable to connect to Qdrant Client")

        app_logger.debug("Success!")

        if not self.client.collection_exists(COLLECTION_NAME):
            app_logger.debug(
                f"Collection does not exist! Creating new collection: {COLLECTION_NAME}"
            )
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_LENGTH, distance=Distance.COSINE
                ),
            )

    def query_db(self, query_str: str, num_retrieve: int = 5) -> list[QueryResDict]:
        """Returns entries with similar vectors as query_str"""
        app_logger.debug(f"Querying db for entries similar to: {query_str}")

        query_embedding = self.embedding_model.encode(query_str)  # type: ignore

        # TODO: SAME AS BELOW
        query_res = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,  # type: ignore
            limit=num_retrieve,
            with_payload=True,
            with_vectors=True,
        )

        return self._query_response_to_dict(query_res)

    def check_article_exist(
        self, article: ArticleDict, show_debug: bool = True
    ) -> bool:
        """Returns true if the meta data in the Article Dict matches an existing db item"""
        if show_debug:
            app_logger.debug(f"Checking for existing article: {article['title']}")

        query_filter = Filter(
            must=[
                FieldCondition(key="title", match=MatchValue(value=article["title"])),
                FieldCondition(
                    key="date", match=MatchValue(value=format_date(article["date"]))
                ),
            ]
        )

        records = self.client.scroll(
            collection_name=COLLECTION_NAME, scroll_filter=query_filter, limit=1
        )

        if not records[0]:
            # nothing returned so it doesn't exist
            return False

        return True

    def add_articles(
        self, article_list: list[ArticleDict], max_chunks_size: int | None
    ):
        """Chunks and adds articles to the database"""
        app_logger.debug(f"Adding chunks to db, collection: {COLLECTION_NAME}")

        article_list = self.filter_out_existing_articles(article_list)
        max_chunks_size = max_chunks_size or self.max_chunks_size
        chunk_list = self._convert_articles_to_chunks(article_list, max_chunks_size)

        # TODO: is vector as a Tensor valid? error is being raised but unsure
        # NOTE: all embeddings should be populated from _convert_articles_to_chunks
        points = [
            PointStruct(
                id=str(uuid4),
                vector=chunk["embeddings"],  # type: ignore
                payload={
                    "source": chunk["source"],
                    "date": chunk["date"],
                    "url": chunk["url"],
                    "title": chunk["title"],
                    "authors": chunk["authors"],
                    "text": chunk["text"],
                    "chunk": chunk["chunk"],
                },
            )
            for chunk in chunk_list
        ]

        self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def filter_out_existing_articles(
        self, article_list: list[ArticleDict]
    ) -> list[ArticleDict]:
        """Removes articles from the list that which exist already in the db"""
        app_logger.debug(
            "Filtering list of articles to remove those that already exist..."
        )

        result: list[ArticleDict] = []
        removed_count = 0
        for article in article_list:
            if self.check_article_exist(article):
                removed_count += 1
                continue

            result.append(article)

        app_logger.debug(f"Removed {removed_count} of {len(article_list)} articles")

        return result

    @staticmethod
    def is_running() -> bool:
        """Checks that the qdrant client is running"""

        try:
            QdrantClient(host="localhost", port=QDRANT_PORT)
        except:
            return False

        return True

    def _query_response_to_dict(self, query_res: QueryResponse) -> list[QueryResDict]:
        """Converts a query response into a list of dicts"""
        result: list[QueryResDict] = []
        for point in query_res.points:
            chunk = point.payload
            new_dict: QueryResDict = {  # type: ignore
                "source": chunk["source"],  # type: ignore
                "date": chunk["date"],  # type: ignore
                "url": chunk["url"],  # type: ignore
                "title": chunk["title"],  # type: ignore
                "authors": chunk["authors"],  # type: ignore
                "text": chunk["text"],  # type: ignore
                "chunk": chunk["chunk"],  # type: ignore
                "score": float(point.score),
                "vector": point.vector,  # type: ignore
            }
            result.append(new_dict)

        return result

    def _convert_articles_to_chunks(
        self, article_list: list[ArticleDict], max_chunks_size: int
    ):
        """Converts the retrieved articles into chunks"""

        nlp = English()
        _ = nlp.add_pipe("sentencizer")

        chunk_list: list[ChunkDict] = []
        for article in app_logger.tqdm(article_list, "Chunking articles..."):
            article: ArticleDict

            sent_list = self._text_to_sentence_list(article, nlp)
            chunk_texts = self._chunk_sentence_list(sent_list, max_chunks_size)
            new_chunks: list[ChunkDict] = self._create_chunk_list(article, chunk_texts)
            chunk_list += new_chunks

        self._add_chunk_embedding(chunk_list)

        return chunk_list

    def _add_chunk_embedding(self, chunks: list[ChunkDict]) -> list[ChunkDict]:
        """
        Creates a list of embeddings from a list of chunk sentences

        **Note**: this is an inplace operation
        """

        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings: torch.Tensor = self.embedding_model.encode(  # type: ignore
            chunk_texts,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=app_logger.is_debug(),
        )

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embeddings"] = embedding

        return chunks

    def _text_to_sentence_list(self, article: ArticleDict, nlp: English) -> list[str]:
        """Converts paragraph of text into a list of sentences"""
        article_text = article["text"]
        sent_list = list(nlp(article_text).sents)
        return [str(s) for s in sent_list]

    def _chunk_sentence_list(
        self, sent_list: list[str], max_sent_count: int
    ) -> list[str]:
        """Breaks list of sentences into sublists with specified max size"""

        chunk_sent_lists = split_list(sent_list, max_sent_count)

        chunks: list[str] = []
        for chunk in chunk_sent_lists:
            chunk_text = "".join(chunk).replace("  ", " ").strip()
            chunk_text = re.sub(r"\.([A-Z])", r". \1", chunk_text)
            chunks.append(chunk_text)

        return chunks

    def _create_chunk_list(
        self, article: ArticleDict, chunk_texts: list[str]
    ) -> list[ChunkDict]:
        """Merges chunk text and meta data from original article to a list of chunks"""
        chunks: list[ChunkDict] = []
        for i, text in enumerate(chunk_texts):
            new_dict: ChunkDict = {
                "chunk": i,
                "source": article["source"],
                "date": format_date(article["date"]),
                "title": article["title"],
                "authors": article["authors"],
                "url": article["url"],
                "text": text,
                "embeddings": None,
            }

            chunks.append(new_dict)

        return chunks
