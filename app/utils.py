"""
General purpose module for storing misc functions and utilities
"""

from __future__ import annotations

import os

import torch
from huggingface_hub import HfFolder, whoami  # type: ignore

from app.log import app_logger


def connect_hugging_face() -> bool:
    """Attempts to connect to hugging face with HF_TOKEN stored in env variables"""
    app_logger.debug("Attempting to connect to hugging face...")

    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN is None:
        app_logger.debug(f"Failed! Could not find HF_TOKEN in evn variables")
        return False

    HfFolder.save_token(HF_TOKEN)  # type: ignore

    if app_logger.is_debug():
        user = whoami()  # type: ignore
        app_logger.debug(
            f"Success! Logged into hugging face as: {user['fullname']} - {user['name']}"
        )

    return True


def get_cuda_name() -> str | None:
    """Gets the name of the cuda device"""

    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(torch.cuda.current_device())
        return device

    return None
