"""
Script for ingesting articles Gdelt with specific criteria
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from app.database import Vector_DB
from app.log import app_logger
from app.news import GdeltScrapper


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser"""

    parser = argparse.ArgumentParser(
        prog="Gdelt Scraper",  
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Scrapes the Gdelt repository for articles related to the provided key words"
    )

    parser.add_argument(
        "-k",
        "--key-words",
        help="Specifies key words to query GDelt with.",
        dest="key_words",
        required=True
    )

    parser.add_argument(
        "-n",
        "--num-record",
        help="Specifies the number of records to retrieve and add to the database.",
        dest="num_records",
        default=30,
        required=False,
        type=int
    )

    return parser


if __name__ == "__main__":
    app_logger.info("Initializing execution of the Gdelt Scraper...")

    args = get_parser().parse_args()

    db = Vector_DB()
    gdelt = GdeltScrapper()

    gdelt.scrape(keywords=args.key_words, num_records=args.num_records)
    db.add_articles(gdelt.results)

    app_logger.info("Gdelt ingestion - Complete!")
