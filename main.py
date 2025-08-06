"""
Main execution script for the `RAG Timeline` application
"""

import logging
import streamlit as st


@st.cache_resource
def get_app(log_level):
    from app import App

    return App(log_level)


def main(log_level: int):
    """Main execution function"""

    app = get_app(log_level)
    app.execute()

main(logging.DEBUG)

