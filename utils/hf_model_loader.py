"""Helpers for resolving model weights from local disk or Hugging Face.

The local project keeps its existing on-disk model path behavior.
If a weight file is missing, deployment environments can fall back to
downloading the file from a Hugging Face repository into a local cache.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download


def _normalize_repo_id(repo_id: str) -> str:
    return repo_id.strip().rstrip("/")


def get_model_path(
    local_path: str | Path,
    *,
    hf_repo: str = "",
    filename: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> Optional[Path]:
    """Return a usable model path.

    Preference order:
    1. An existing local file.
    2. A Hugging Face download if `hf_repo` is configured.
    3. `None` if neither is available.
    """

    resolved_local_path = Path(local_path)
    if resolved_local_path.exists():
        return resolved_local_path

    repo_id = _normalize_repo_id(hf_repo or os.getenv("HF_MODEL_REPO", ""))
    if not repo_id:
        return None

    model_name = filename or resolved_local_path.name
    download_cache_dir = Path(
        cache_dir or os.getenv("MODEL_CACHE_DIR", "/tmp/models")
    )
    download_cache_dir.mkdir(parents=True, exist_ok=True)

    auth_token = token if token is not None else os.getenv("HF_TOKEN") or None
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_name,
        cache_dir=str(download_cache_dir),
        token=auth_token,
    )
    return Path(downloaded_path)

