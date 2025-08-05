"""
General purpose module for storing misc functions and utilities
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from huggingface_hub import HfFolder, whoami  # type: ignore
from sentence_transformers import CrossEncoder, SentenceTransformer
from torch import cuda, float16
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers.utils.quantization_config import BitsAndBytesConfig

import app.constants as const
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

    if cuda.is_available():
        device = cuda.get_device_name(cuda.current_device())
        return device

    return None


def log_cuda_mem():
    """Helper method to print the current memory used/available on cuda device"""
    if cuda.is_available():
        gpu_id = cuda.current_device()
        total = cuda.get_device_properties(gpu_id).total_memory  # type: ignore
        reserved = cuda.memory_reserved(gpu_id)
        allocated = cuda.memory_allocated(gpu_id)
        free = reserved - allocated

        app_logger.debug(f"Total memory:     {total / 1e6:.2f} MB")
        app_logger.debug(f"Reserved memory:  {reserved / 1e6:.2f} MB")
        app_logger.debug(f"Allocated memory: {allocated / 1e6:.2f} MB")
        app_logger.debug(f"Free within reserved: {free / 1e6:.2f} MB")
    else:
        app_logger.debug("No CUDA device available.")


def create_embedding_model(
    model_name: str = const.EMBEDDING_MODEL_NAME, send_to_cuda: bool = True
) -> SentenceTransformer:
    """Initializes a reranker model"""
    app_logger.debug(f"Attempting to initialize a embedding model: {model_name}...")

    model = SentenceTransformer(model_name)

    if send_to_cuda:
        if cuda.is_available():
            model.to("cuda")
            app_logger.debug(
                f"Embedding model sent to cuda device: {get_cuda_name()}..."
            )
        else:
            app_logger.warning(
                f"Cuda not available! Attempt to send embedding model to cuda failed."
            )

    app_logger.debug(f"Embedding model intialization complete.")

    return model


def create_reranker_model(
    model_name: str = const.RERANKING_MODEL_NAME, send_to_cuda: bool = True
) -> CrossEncoder:
    """Initializes a reranker model"""
    app_logger.debug(f"Attempting to initialize a reranker model: {model_name}")

    model = CrossEncoder(model_name)

    if send_to_cuda:
        if cuda.is_available():
            model.to("cuda")
            app_logger.debug(f"Reranker model sent to cuda device: {get_cuda_name()}")
        else:
            app_logger.warning(
                f"Cuda not available! Attempt to send reranker model to cuda failed."
            )

    app_logger.debug(f"Reranker model intialization complete.")

    return model


def create_llm(
    model_name: str = const.LLM_MODEL_NAME, send_to_cuda: bool = True
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Helper method to initialize an llm model and the associated auto tokenizer"""
    app_logger.debug(f"Attempting to initialize a LLM: {model_name}...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=float16,
        bnb_4bit_quant_type="nf4",
    )

    config: dict[str, Any] = {
        "pretrained_model_name_or_path": model_name,
        "torch_dtype": float16,
        "device_map": "auto",
        "quantization_config": quant_config,
    }

    if is_flash_attn_2_available() and (cuda.get_device_capability(0)[0] >= 8):
        app_logger.debug(f"Using flash_attention_2")
        config["attn_implementation"] = "flash_attention_2"

    app_logger.debug("Creating llm")
    llm: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(**config)  # type: ignore

    app_logger.debug("Creating autotokenizer")
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)  # type: ignore

    app_logger.debug(f"LLM and autotokenizer intialization complete.")

    return tokenizer, llm  # pyright: ignore[reportUnknownVariableType]


def format_date(date: datetime | None) -> str:
    """Converts date or datetime into a str"""
    if date is None:
        return ""

    return date.strftime("%Y-%m-%d")
