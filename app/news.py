"""
Module for storing news scrapers
"""

from __future__ import annotations

from datetime import _Date, datetime, timedelta #type: ignore / using _Date for type hints only
from enum import Enum
from typing import Any

import newspaper
import pandas as pd
from gdeltdoc import Filters, GdeltDoc
from app.exc import ScraperError
from app.log import app_logger
from gdelt_theme_set import GDELT_THEMES
from app.constants import GdeltDict, ArticleDict, ChunkDict

class GdeltSource(Enum):
    """Storafe Enum for source links in Gdelt"""

    # CBC = "cbc.ca"                # wasn't working for some reason
    CNN = "cnn.com"
    # AP = "apnews.com"             # wasn't working for some reason
    BBC = "bbc.co.uk"
    # Wired = "wired.com"           # wasn't working for some reason
    # Reuters = "reuters.com"       # wasn't working for some reason
    NyTimes = "nytimes.com"
    Guardian = "theguardian.com"


def format_date(date: _Date) -> str:
    """Helper method to format a date for scrapers"""
    return date.strftime("%Y-%m-%d")


def format_text(text: str) -> str:
    """Removes newline characters + leading/trailing white space from string"""
    return text.replace("\n", "").strip()


class ArticleFetcher:
    """Container class that wraps the newspaper4k module"""

    @classmethod
    def fecth_articles(cls, urls: list[str] | str) -> list[ArticleDict]:
        """Queries article urls and extracts text and metadata"""
        urls = urls if isinstance(urls, list) else [urls]

        results: list[ArticleDict] = []

        # track existing titles as it can be used to filter
        # articles already captured (but existing under
        # different site, e.g., cnn.com vs cnn.uk)
        # to reduce duplicate work
        existing_titles: set[str] = set()
        for url in app_logger.tqdm(urls, "Fetching articles text and metadata..."):
            url: str

            # news.articles is an expensive operation so verify
            # the current story hasn't already been retrieved
            possible_title = url.rsplit("/", 1)[-1]
            if possible_title in existing_titles:
                app_logger.debug(f"Removed duplicate story: {possible_title}")
                continue
            existing_titles.add(possible_title)

            try:
                article = newspaper.article(        # pyright: ignore[reportUnknownMemberType]
                    url
                )  
            except:
                app_logger.warning(f"Error encountered while fetching: {url}")
                continue

            results.append(cls._article_to_dict(article))

        return results

    @classmethod
    def _article_to_dict(cls, article: newspaper.Article) -> ArticleDict:
        """Convert the Article object into a friendly dict"""

        new_dict: ArticleDict =  {
            "text": format_text(article.text),
            "title": article.title,
            "authors": article.authors,
            "date": article.publish_date,
            "source": article.source_url,
            "url": article.original_url,
        }

        return new_dict
    
    @classmethod
    def _add_metadata(cls, article_dict: dict[str, str | None | datetime | list[str]]):
        """Adds meta data like length, expected tokens, sentences etc, """


class GdeltScrapper:

    GdeltSource = GdeltSource

    def __init__(self) -> None:
        """Class for managing scrapping GDelt and adding articles to the database"""
        self.results: list[dict[str, Any]] = []

    def clear_results(self):
        """Method to clear the results from the scraper"""
        self.results = []

    def scrape(
        self,
        keywords: str | None = None,
        theme: str | None = None,
        sources: list[GdeltSource] | None = None,
        start_date: _Date = datetime.now().date() - timedelta(days=7),
        end_date: _Date = datetime.now().date(),
        num_records: int = 5,
    ):
        """Scrapes gdelt for the specified inputs"""
        app_logger.debug("Attempting to scrape Gdelt...")

        sources = sources or [s for s in GdeltSource]
        sources_str = [s.value for s in sources]

        if theme:
            theme = theme.upper()

        if not self._validate_scraper_input(
            keywords, theme, sources, start_date, end_date
        ):
            app_logger.error("Gdelt scraper input validation has failed.")
            raise ScraperError("Gdelt scraper input validation has failed.")

        gdelt_filter = self._init_filter(
            keywords,
            theme,
            sources_str,
            format_date(start_date),
            format_date(end_date),
            num_records,
        )
        gdelt_results = self._search(gdelt_filter)

        story_urls = [story["url"] for story in gdelt_results]
        articles = ArticleFetcher.fecth_articles(story_urls)

        app_logger.debug(
            f"Scaping complete. {len(articles)} articles remain after filtering"
        )

        self.results = articles

    def _search(self, gdelt_filters: Filters) -> list[GdeltDict]:
        """Searches Gdelt for articles with specified filters"""
        app_logger.debug("Searching Gdelt...")
        gdelt = GdeltDoc()
        search_results = gdelt.article_search(gdelt_filters)
        formatted_results = self._gdelt_to_dict(search_results)

        app_logger.debug(f"Searching complete, {len(formatted_results)} articles found")

        return formatted_results

    def _gdelt_to_dict(self, articles: pd.DataFrame) -> list[GdeltDict]:
        """Converts the output of gdelt.article_search from a dataframe into a easier to work with dict"""
        results: list[GdeltDict] = []
        for row in articles.iterrows():
            new_dict: GdeltDict = {
                "url": row["url"],  # type: ignore
                "title": row["title"],  # type: ignore
                "domain": row["domain"],  # type: ignore
                "country": row["sourcecountry"],  # type: ignore
            }
            results.append(new_dict)  # pyright: ignore[reportUnknownMemberType]

        return results

    def _validate_scraper_input(
        self,
        keywords: str | None,
        theme: str | None,
        sources: list[GdeltSource],
        start_date: _Date,
        end_date: _Date,
    ) -> bool:
        """Validates that the scraper input is valid and avoids common Gdelt query errors"""
        is_valid = True

        if keywords and len(keywords) < 5:
            app_logger.error(
                "Keywords must be gte 5 characters or the Gdelt API errors."
            )
            is_valid = False

        if theme and theme not in GDELT_THEMES:
            app_logger.error(f"{theme} is not a valid GDelt theme.")
            is_valid = False

        if not theme and not keywords:
            app_logger.error("Must specify either theme or keywords.")
            is_valid = False

        if len(sources) < 2:
            app_logger.error("Number of sources must be gte 2 or the Gdelt API errors.")
            is_valid = False

        if start_date > datetime.now().date():
            app_logger.error("Start date after today. News cannot be in the future...")
            is_valid = False

        if end_date > datetime.now().date():
            app_logger.error("End date after today. News cannot be in the future...")
            is_valid = False

        if start_date > end_date:
            app_logger.error(
                "Start date after end date. How you gunna have news in the future?"
            )
            is_valid = False

        return is_valid

    def _init_filter(
        self,
        keywords: str | None,
        theme: str | None,
        sources: list[str],
        start_date_str: str,
        end_date_str: str,
        num_records: int,
    ) -> Filters:
        """Helper method to construct a GDelt filter with input parameters"""
        app_logger.debug("creating GDelt filter")

        config: dict[str, Any] = {
            "start_date": start_date_str,
            "end_date": end_date_str,
            "num_records": num_records,
            "domain": sources,
            "country": ["UK", "US"],
            "language": "eng",
        }

        if not keywords and not theme:
            app_logger.critical("Both keywords and theme cannot be empty!")
            raise ScraperError("Both keywords and theme cannot be empty!")

        if theme:
            config["theme"] = theme

        if keywords:
            config["keywords"] = keywords

        return Filters(**config)
