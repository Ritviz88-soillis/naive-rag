"""Helpers for parsing YouTube URLs (shared, stateless -> utils/)."""

import re
from urllib.parse import parse_qs, urlparse

# A YouTube video id is exactly 11 URL-safe characters.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """Extract the 11-character video id from a YouTube URL (or pass an id through).
    Args:
        url_or_id: A full YouTube URL or a bare video id.

    Returns:
        The 11-character video id.

    Raises:
        ValueError: If no valid video id can be found in the input.
    """

    text = url_or_id.strip()

    # Already a bare id.
    if _ID_PATTERN.match(text):
        return text

    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    # youtu.be/<id>
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
        if _ID_PATTERN.match(candidate):
            return candidate

    # youtube.com/... forms
    if host.endswith("youtube.com"):
        query = parse_qs(parsed.query)
        if "v" in query and _ID_PATTERN.match(query["v"][0]):
            return query["v"][0]

        parts = parsed.path.split("/")
        for index, part in enumerate(parts):
            if part in ("embed", "shorts", "v") and index + 1 < len(parts):
                candidate = parts[index + 1]
                if _ID_PATTERN.match(candidate):
                    return candidate

    raise ValueError(f"Could not extract a YouTube video id from: {url_or_id!r}")
