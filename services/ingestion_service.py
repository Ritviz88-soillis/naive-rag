"""Stage 1 — ingestion: fetch a YouTube transcript."""

from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi

import config


class IngestionService:
    """Fetches and flattens a video's transcript into plain text."""

    def __init__(self) -> None:
        """Instantiate the transcript API client once."""

        self._api = YouTubeTranscriptApi()

    def fetch_transcript(self, video_id: str, language: str = config.DEFAULT_LANGUAGE) -> str:
        """Fetch the transcript for a video and join it into one string.

        Args:
            video_id: The YouTube video id.
            language: The transcript language code (e.g. "en").

        Returns:
            The full transcript text.

        Raises:
            TranscriptsDisabled: If the video has no captions available.
        """

        fetched = self._api.fetch(video_id, languages=[language])
        return " ".join(snippet.text for snippet in fetched)
