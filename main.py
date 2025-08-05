"""
Main execution script for the `RAG Timeline` application
"""

import argparse
import logging

from app import App

VALID_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser"""

    parser = argparse.ArgumentParser(
        prog="RAG_Timeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # TODO: update default to info
    parser.add_argument(
        "--log-level",
        type=str,
        choices=VALID_LOG_LEVELS.keys(),
        default="debug",
        help="Set the log level for the application (default: info)",
        dest="log_level",
    )

    parser.add_argument(
        "-d",
        "--download-feed",
        help="Pull in the latest feeds via the RSS Feed library.",
        dest="update",
        action="store_true",
    )

    return parser


def main(log_level: int, update: bool):
    """Main execution function"""

    if update:
        App.download_feed(log_level)
    else:
        app = App(log_level)
        app.execute()


if __name__ == "__main__":
    print("Hi")

    parser = get_parser()
    args = parser.parse_args()

    main(log_level=VALID_LOG_LEVELS[args.log_level], update=args.update)
