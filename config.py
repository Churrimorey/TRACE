from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_openai_client() -> OpenAI:
    api_key = get_env("OPENAI_API_KEY", required=True)
    base_url = os.getenv("OPENAI_BASE_URL")
    organization = os.getenv("OPENAI_ORG")
    project = os.getenv("OPENAI_PROJECT")

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if organization:
        kwargs["organization"] = organization
    if project:
        kwargs["project"] = project

    return OpenAI(api_key=api_key, **kwargs)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
