"""
Module for initializing the RAG required utilities to allow querying

*NOTE*: The following tutorial was heavily utilized to generate the overall workflow and methods for this RAG implementation
- Local Retrieval Augmented Generation (RAG) from Scratch (step by step tutorial) by Daniel Bourke
link: https://www.youtube.com/watch?v=qN_2fnOPY-M
"""

from __future__ import annotations

import textwrap

import app.constants as const
from app.database import Vector_DB
from app.exc import RagError
from app.log import app_logger
from app.utils import create_llm, create_reranker_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import re
import json
import numpy as np

class Lda:

    def __init__(self, context: list[tuple[float, const.QueryResDict]], num_topics: int | None = None ) -> None:
        """Function for topic modeling over the context"""
        self.num_topics = num_topics or const.NUM_LDA_TOPICS

        self.documents = [resp["text"] for _, resp in context if "text" in resp]

        self.vectorizer = CountVectorizer(stop_words="english")
        self.doc_term_matrix = self.vectorizer.fit_transform(self.documents)

        self.lda_model = LatentDirichletAllocation(n_components=self.num_topics, random_state=42)
        self.lda_model.fit(self.doc_term_matrix)
    
    def get_topic_word_freqs(self, topic_idx: int):
        topic = self.lda_model.components_[topic_idx]
        feature_names = self.vectorizer.get_feature_names_out()
        top_indices = topic.argsort()[:-self.num_topics - 1:-1]
        return {feature_names[i]: topic[i] for i in top_indices}

    def get_aggregate_topic_frequencies(self) -> dict[str, float]:
        # Sum across all topics to get total word weight
        combined_topic = np.sum(self.lda_model.components_, axis=0)
        feature_names = self.vectorizer.get_feature_names_out()
        top_indices = combined_topic.argsort()[-self.num_topics:][::-1]
        
        return {feature_names[i]: combined_topic[i] for i in top_indices}

class Reranker:

    def __init__(self, model_name: str | None = None) -> None:
        """Class for storing and interfacing with the re-ranker model"""
        app_logger.debug("Initializating reranking model...")
        model_name = model_name or const.RERANKING_MODEL_NAME

        self.model = create_reranker_model(model_name)

    def rerank(
        self,
        query_str: str,
        query_resp_list: list[const.QueryResDict],
    ) -> list[tuple[float, const.QueryResDict]]:
        """Reranks the input articles against the query"""
        app_logger.debug(f"Reranking {len(query_resp_list)} articles...")

        # extract original chunk text
        text_pairs = [(query_str, resp["text"]) for resp in query_resp_list]

        new_scores = self.model.predict(
            text_pairs,
            batch_size=32,
            show_progress_bar=app_logger.is_debug(),
        )

        app_logger.debug(f"new scores: {new_scores}")

        ordered_results = sorted(
            zip(new_scores.astype(float), query_resp_list),
            key=lambda x: x[0],
            reverse=True,
        )

        return ordered_results


class RAG:

    NUM_ARTICLES = 5
    GENERIC_MULTIPLE = 1

    def __init__(
        self,
        database: Vector_DB,
        llm_model_name: str | None = None,
        reranker_model_name: str | None = None,
        articles_per_context: int | None = None,
    ) -> None:
        """Class for complete the RAG portion of the application"""

        self.db = database
        self.articles_per_context = articles_per_context or self.NUM_ARTICLES
        self.temp = const.TEMPERATURE
        self.max_new_tokens_summary = const.MAX_NEW_TOKENS_SUMMARY
        self.max_new_tokens_events = const.MAX_NEW_TOKENS_EVENTS
        
        self.reranker = Reranker(reranker_model_name)
        app_logger.debug("Initializating llm model...")

        llm_model_name = llm_model_name or const.LLM_MODEL_NAME

        self.tokenizer, self.model = create_llm(llm_model_name)

        self.query_str: str = ""
        self.context: list[const.QueryResDict] = []
        self.ranked_context: list[tuple[float, const.QueryResDict]] = []

    def load_context(self, query_str: str, query_resp_list: list[const.QueryResDict]):
        """Loads the specified query responses into the RAGs context"""
        self.query_str = query_str
        self.context = query_resp_list.copy()
        self.ranked_context = self.reranker.rerank(query_str, query_resp_list)

    def set_query(self, query_str: str, num_retrieve: int | None = None):
        """Sets query string and searches db to load context"""
        self.clear_context()
        app_logger.debug(f"Updating context to for query: {query_str}")

        self.articles_per_context = num_retrieve or self.NUM_ARTICLES

        # times 3 to get more information before passing to reranker
        query_results = self.db.query_db(
            query_str, self.articles_per_context * self.GENERIC_MULTIPLE
        )
        ranked_results = self.reranker.rerank(query_str, query_results)

        self.query_str = query_str
        self.context = query_results
        self.ranked_context = ranked_results

    def get_ranked_context(self) -> list[tuple[float, const.QueryResDict]]:
        """Returns the ranked context"""
        return self.ranked_context

    def clear_context(self):
        """Clears current query string and context"""
        app_logger.debug("Clearing context")
        self.query_str = ""
        self.context = []
        self.ranked_context = []

    def ask_summary(self) -> tuple[str, str]:
        """Querying the RAG for a summary of the given context"""
        if not self._context_exists():
            app_logger.debug("Requesting summary prior to context being populated")
            raise RagError("Requesting summary prior to context being populated")

        prompt, summary = self.generate(self._summary_format(), True, return_tokens=self.max_new_tokens_summary)

        return prompt, summary

    def ask_key_events(self) -> tuple[str, str]:
        """Querying the RAG for a summary of key events given context"""
        if not self._context_exists():
            app_logger.debug("Requesting events prior to context being populated")
            raise RagError("Requesting events prior to context being populated")

        prompt, summary = self.generate(self._event_format(), True, return_tokens=self.max_new_tokens_events)

        updated_summary = self._sanitize_events(summary)

        return prompt, updated_summary

    def generate(
        self, prompt_format: str, format_output: bool = True, return_tokens: int | None = None
    ) -> tuple[str, str]:
        """Generates the prompt to be returned from the RAG"""

        app_logger.debug("Creating prompt and tokens...")
        # augment data and compile into prompt
        prompt_str = self._format_prompt(self.query_str, prompt_format)
        template = [{"role": "user", "content": prompt_str}]
        prompt = self.tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)  # type: ignore
        tokens = self.tokenizer(prompt, return_tensors="pt").to("cuda")  # type: ignore


        return_tokens = return_tokens or self.max_new_tokens_summary

        app_logger.debug("Generating RAG response...")
        output_tokens = self.model.generate(  # type: ignore
            **tokens,
            temperature=self.temp,
            do_sample=True,
            max_new_tokens=return_tokens,
            pad_token_id=self.tokenizer.eos_token_id,  # just to get rid of a warning msg # type: ignore
        )
        app_logger.debug("Done!")
    
        output_text = self.tokenizer.decode(output_tokens[0])

        if format_output:
            output_text = output_text.replace(prompt, "").replace("<s> ", "").replace("</s>", "")  # type: ignore

        return prompt_str, output_text  # type: ignore

    def topic_model_context(self) -> dict[str, float]:
        """Topic models the ranked context"""
        lda = Lda(self.ranked_context[:self.articles_per_context])
        key_topics = lda.get_aggregate_topic_frequencies()

        return key_topics

    def _sanitize_events(self, event_results: str) -> str:
        """Formats event summary so it is valid json"""
        app_logger.debug("Sanitizing Events...")
        
        with open("raw_event_results.txt", "w+") as f:
            f.write(event_results)

        event_pattern = re.compile(r'\{[^{}]*"start_date"[^{}]*\{[^{}]*"year"[^{}]*\}[^{}]*"text"[^{}]*\{[^{}]*"headline"[^{}]*"text"[^{}]*\}[^{}]*\}', re.DOTALL)
        event_matches = event_pattern.findall(event_results)
        
        clean_events = []
        for event_str in event_matches:
            try:
                json_str = event_str.replace("'", '"')
                event = json.loads(json_str)
                clean_events.append(event)
            except:
                continue
        
        result = str({"events": clean_events})
        result = result.replace("'", '"')
        return result

    def _format_prompt(self, query_str: str, prompt_format: str) -> str:
        """Create prompt"""
        context_str = self._create_context_str(self.articles_per_context, True)
        prompt_str = prompt_format.format(context=context_str, query_str = query_str)

        return prompt_str

    def _create_context_str(self, num_articles: int, use_ranked: bool = True) -> str:
        """Uses current context and creates a string to be added to prompt"""

        if use_ranked:
            context_list = [item[1] for item in self.ranked_context[:num_articles]]
        else:
            context_list = self.context

        context_str = ""
        for i, chunk in enumerate(context_list):
            context_str += (
                f"article {i+1} - (published {chunk['date']}): {chunk['title']}\n"
            )
            context_str += f"text: {chunk['text']}\n"

        return context_str

    def _context_exists(self) -> bool:
        """Confirms the context has been set and is valid"""

        # 5 due to qdrant issues on larger models
        if self.query_str == "":
            return False

        if len(self.context) == 0:
            return False

        if len(self.ranked_context) == 0:
            return False

        return True

    def _event_format(self) -> str:
        """Format for event prompt"""
                # {{
                #     "title": "<event title>",
                #     "description": "<brief description (1-2 concise sentences)>",
                #     "date": "<date or date range>"
                # }},
                # ...
        return textwrap.dedent(
            """\
            You are a helpful assistant that extracts a list of key events from news article excerpts.

            Context:
            {context}

            User Question:
            {query_str}

            Instructions:
            - Based only on the provided context, extract key events relevant to the question.
            - For each event, provide a short title, a brief description, and an exact or approximate date or date range.
            - Return the results in a structured, easy-to-parse format as a list of dictionaries like this:
            
            {{
                "events": [
                    {{
                        "start_date": {{ "year": <year>, "month": <month>, "day": <day> }},
                        "text": {{
                            "headline": "<event title>",
                            "text": "<brief description (1-2 concise sentences)>"
                        }}
                    }},
                    ...
                ]
            }}

            - veritfy the output matches the above format and remove events that are incomplete
            - If no events are found, return an empty list: []
            - Do not fabricate events or use information not present in the context.
            - If start_date is unkown, use publish date
            
            Extracted Events:
        """
        )
    
    
            # - for the start_date, year is required.
            # - for the start_date, use 0 for <month> and/or <day> if those values are unknown

    def _summary_format(self) -> str:
        """Format for summary prompt"""
        return textwrap.dedent(
            """\
            You are a helpful assistant that summarizes current events based on news article excerpts.

            Context:
            {context}

            User Question:
            {query_str}

            Instructions:
            - Based only on the provided context, write a **concise summary** (2–5 sentences) of the events that directly relate to the user question.
            - Focus on key facts, dates, people, organizations, and consequences.
            - Do not include information that is not supported by the context.
            - Avoid repetition or speculation.

            Summary:
        """
        )
