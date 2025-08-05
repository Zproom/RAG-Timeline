"""
Module for initializing the application
"""

from __future__ import annotations

from datetime import datetime

import app.constants as const
from app.database import Vector_DB
from app.exc import InitError
from app.log import app_logger
from app.rag import RAG
from app.ui import View
from app.utils import connect_hugging_face, get_cuda_name, log_cuda_mem


class App:

    def __init__(self, log_level: int) -> None:
        """
        Main class for execution of application
        """
        app_logger.setLevel(log_level)

        app_logger.debug("Starting App...")

        if not connect_hugging_face():
            raise InitError("Unable to connect to hugging face.")

        self.check_cuda()
        self.db = Vector_DB()
        self.rag = RAG(self.db)
        self.view = View(fetch_data_func=self.fetch_data)
        app_logger.debug("Setup complete!")
        log_cuda_mem()

    def fetch_data(
        self,
        query_str: str,
        start_date: datetime,
        end_date: datetime,
        num_articles: int,
    ):
        """Fetches data to update ui"""
        self.rag.set_query(query_str, num_articles)

        summary_prompt, summary = self.rag.ask_summary()
        event_prompt, events = self.rag.ask_key_events()
        ranked_articles = self.rag.get_ranked_context()[:num_articles]

        payload: const.DataPayload = {
            "summary_prompt": summary_prompt,
            "summary": summary,
            "events": events,
            "event_prompt": event_prompt,
            "ranked_articles": ranked_articles,
        }

        return payload

    def check_cuda(self):
        """Helper method to log if there is a cuda device connected"""
        cuda_name = get_cuda_name()
        if cuda_name:
            app_logger.debug(f"Cuda device detected: {cuda_name}")
        else:
            app_logger.debug(f"Cuda not detected. Models will run on CPU (good luck).")

        log_cuda_mem()

    # TODO: this is where the streamlit viz logic will initiate to not premptively startup
    def execute(self):
        """Executes the UI and other components of the application"""
        self.view.run()

    # TODO
    @staticmethod
    def download_feed(log_level: int):
        """Method to pull down the latest feed via the RSS Feed library"""
        app_logger.setLevel(log_level)
