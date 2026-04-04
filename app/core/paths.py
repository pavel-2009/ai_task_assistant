from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AVATAR_SEGMENTS_DIR = PROJECT_ROOT / "avatars" / "segments"


def get_avatar_segments_dir() -> Path:
    """Return a writable directory for segmentation artifacts."""
    configured_path = os.getenv("AVATAR_SEGMENTS_DIR")
    if configured_path:
        return Path(configured_path)
    return DEFAULT_AVATAR_SEGMENTS_DIR
