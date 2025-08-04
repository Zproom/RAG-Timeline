"""
Module for initializing the application
"""

from __future__ import annotations

from app.exc import InitError
from app.log import app_logger
from app.utils import connect_hugging_face, get_cuda_name


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

        app_logger.debug("Setup complete!")

    def check_cuda(self):
        """Helper method to log if there is a cuda device connected"""
        cuda_name = get_cuda_name()
        if cuda_name:
            app_logger.debug(f"Cuda device detected: {cuda_name}")
        else:
            app_logger.debug(f"Cuda not detected. Models will run on CPU (good luck).")


    # TODO: this is where the streamlit viz logic will initiate to not premptively startup
    def execute(self):
        """Executes the UI and other components of the application"""
        

    # TODO
    @staticmethod
    def download_feed(log_level: int):
        """Method to pull down the latest feed via the RSS Feed library"""
        app_logger.setLevel(log_level)
