from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Callable

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_timeline import timeline

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import app.constants as const
import app.log as log
from typing import Any

JSON_EVENT_FILE = "last_event.json"

class Controller:

    def __init__(self, view: View, fetch_data_func: Callable | None) -> None:
        """Helper class to abstract actual code logic out of View"""
        self.view = view
        self.fetch_func = fetch_data_func or self._default_fetch

    def _default_fetch(self, *args):
        print(args)

    def search_news(self):
        """
        Updates the view according to the search and search results

        **Triggers** when submit button pressed on View
        """
        query_str: str = st.session_state["search_query"]
        start_date: datetime = st.session_state["start_date"]
        end_date: datetime = st.session_state["end_date"]
        article_count: int = st.session_state["article_count"]

        if self.fetch_func:
            payload = self.fetch_func(query_str, start_date, end_date, article_count)

            if payload is not None:
                self.update_with_payload(payload)
        
        st.session_state["query_str"] = query_str

        if query_str == "":
            st.session_state["show_tabs"] = False
            st.rerun()
            return

        st.session_state["show_tabs"] = True

        st.rerun()

    def update_with_payload(self, payload: const.DataPayload):
        """"""
        st.session_state["summary_result"] = payload["summary"]
        st.session_state["event_result"] = payload["events"]
        st.session_state["summary_prompt"] = payload["summary_prompt"]
        st.session_state["event_prompt"] = payload["event_prompt"]
        st.session_state["chunks"] = payload["ranked_articles"]
        st.session_state["topics"] = payload["topics"]

    def events_to_timeline(self):
        """Writes the events to the news.json file for use in the timeline"""

        events_str: str = st.session_state["event_result"]
        
        with open("events_log.txt", "a+") as file:
            file.write(f"\n\nQuery_time: {datetime.now()}")
            file.write(events_str)

        json_output = json.loads(events_str)
        with open(JSON_EVENT_FILE, "w+") as file:
            json.dump(json_output, file, indent=4)
            
class View:

    def __init__(
        self,
        window_title: str | None = None,
        window_icon: str | None = None,
        fetch_data_func: Callable | None = None,
    ):
        """Class for storing streamlit visuals / functionality"""
        log.app_logger.debug("Initializing view...")
        self._window_title = window_title or const.WINDOW_TITLE
        self._window_icon = window_icon or const.WINDOW_ICON
        self.name = ""
        self.number = 0
        self.controller = Controller(self, fetch_data_func)

    def run(self):
        log.app_logger.debug("starting view...")
        self._init_variables()
        self._init_page()
        self._init_sidebar()
        self._init_tabs()
        log.app_logger.debug("View is running...")

    def update_query_str(self, query_str: str):
        """Updates query string"""
        st.session_state["search_query"] = query_str

    def _init_page(self):
        """Initilize page configuration"""
        st.set_page_config(
            page_title=self._window_title, page_icon=self._window_icon, layout="wide"
        )

        st.title("News Timeline")
        st.text_input("Entered Query String:", disabled=True, value=st.session_state["query_str"])

    def _init_sidebar(self):
        """Initialize the sidebar"""
        st.sidebar.header(const.SIDEBAR_TITLE)

        with st.sidebar.form(key="query_form"):

            start_str = datetime(year=2025, month=5, day=1).strftime("%Y-%m-%d")
            end_str = datetime.now().strftime("%Y-%m-%d")

            st.date_input(
                "Start Date",
                min_value=start_str,
                max_value=end_str,
                key="start_date",
            )
            st.date_input(
                "End Date",
                min_value=start_str,
                max_value=end_str,
                key="end_date",
            )
            st.number_input(
                "Number of Results", min_value=1, max_value=10, key="article_count"
            )
            st.text_input(
                "Search Query", key="search_query"
            )

            submitted = st.form_submit_button("Submit Query")

            if submitted:
                self.controller.search_news()

    def _init_variables(self):
        """Helper function to init query_str in session state"""

        if "search_query" not in st.session_state:
            st.session_state["search_query"] = ""

        if "query_str" not in st.session_state:
            st.session_state["query_str"] = "N/A - enter a query from the sidebar"

        if "show_tabs" not in st.session_state:
            st.session_state["show_tabs"] = False

        if "start_date" not in st.session_state:
            st.session_state["start_date"] = (datetime.now() - timedelta(days=7))

        if "end_date" not in st.session_state:
            st.session_state["end_date"] = datetime.now()

        if "article_count" not in st.session_state:
            st.session_state["article_count"] = 3
        
        if "show_tabs" not in st.session_state:
            st.session_state["show_tabs"] = False
        
        if "summary_prompt" not in st.session_state:
            st.session_state["summary_prompt"] = ""

        if "event_prompt" not in st.session_state:
            st.session_state["event_prompt"] = ""

        if "summary_result" not in st.session_state:
            st.session_state["summary_result"] = ""

        if "event_result" not in st.session_state:
            st.session_state["event_result"] = ""

        if "chunks" not in st.session_state:
            st.session_state["chunks"] = []

        if "topics" not in st.session_state:
            st.session_state["topics"] = {}

    def _init_tabs(self):
        """Initialize the tabs"""

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Summary", "Related Chunks", "Wordcloud", "Prompts"]
        )

        with tab1:
            self._init_tab_summary()

        with tab2:
            self._init_tab_chunks()

        with tab3:
            self._init_tab_wordcloud()

        with tab4:
            self._init_tab_prompts()

    def _init_tab_summary(self):
        """Initialize the summary tab"""
        
        if not st.session_state["show_tabs"]:
            return

        st.text_area("Summary of Events:", key="summary_result", disabled=True, height=400)

        self.controller.events_to_timeline()

        with open(JSON_EVENT_FILE, "r") as f:
            timeline_data = json.load(f)

        timeline(timeline_data, height=600)

    def _init_tab_chunks(self):
        """Initialize the articles tab"""
        if not st.session_state["show_tabs"]:
            return
        
        for score, chunk in st.session_state["chunks"]:
            self.render_chunk(score, chunk)

    def render_chunk(self, score: float, chunk: const.QueryResDict):

        date_text = chunk.get("date", "")
        date_text = f"[published - {date_text}]" if date_text != "" else ""
        score_text = f"Score: {score:.6f},"
        main_text = " ".join([score_text, chunk["title"], date_text])
        with st.expander(main_text):
            st.markdown(f"**Source:** {chunk.get('source', 'N/A')}")
            st.markdown(f"**Authors:** {', '.join(chunk.get('authors', []))}")
            st.markdown(f"**URL:** [{chunk.get('url', 'N/A')}]({chunk.get('url', '#')})")
            st.markdown("**Text:**")
            st.write(chunk.get("text", "No text available."))

    def _init_tab_wordcloud(self):
        """Initialize the articles tab"""
        if not st.session_state["show_tabs"]:
            return
        
        word_freqs = st.session_state["topics"]        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freqs)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')

        st.pyplot(fig)

    def _init_tab_prompts(self):
        """Initializes the prompts tab"""
        if not st.session_state["show_tabs"]:
            return

        st.text_area("Prompt for the summary:", key="summary_prompt", disabled=True, height=600)

        st.text_area("Prompt for the events:", key="event_prompt", disabled=True, height=600)

        st.text_area("Events raw output:", key="event_result", disabled=True, height=1000)

    @st.dialog("Search for Articles?")
    def yes_no_popup(self):
        """IGNORE WON'T GET HERE"""
        st.write(
            "Insufficient articles were found for your query. Would you like to search for more articles?"
        )

        if st.button("Search for more articles..."):
            print("Searching for more articles")
            st.rerun()

        if st.button("Close"):
            print("Closing")
            st.rerun()
