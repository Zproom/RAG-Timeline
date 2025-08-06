"""
Script for ingesting articles from RSS feeds
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from app.constants import RSS_FEED_URLS
from app.database import Vector_DB
from app.log import app_logger
from app.news import RssScraper

if __name__ == "__main__":
    app_logger.info("Initializing execution of the RSS Feed Scraper...")

    db = Vector_DB()

    rss = RssScraper()

    for source, url in RSS_FEED_URLS:
        rss.scrape(source, url)

        if len(rss.results) > 0:
            db.add_articles(rss.results)

        rss.clear_results()
        