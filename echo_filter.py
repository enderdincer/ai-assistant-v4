"""Echo filter for text-based echo detection.

This module provides text-based echo filtering to prevent the assistant
from responding to its own speech. It uses word overlap and fuzzy string
matching to distinguish between echo and genuine user input.
"""

import re
import time
from difflib import SequenceMatcher
from typing import List, Optional, Set
from ai_assistant.shared.logging import get_logger

logger = get_logger(__name__)


def _extract_words(text: str) -> Set[str]:
    """Extract lowercase words from text, removing punctuation."""
    # Remove punctuation and split into words
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    # Include all words (removed the length filter to catch more matches)
    return set(words)


class EchoFilter:
    """Filter echo by comparing transcriptions with stored TTS responses.

    This class tracks recent TTS outputs and compares incoming transcriptions
    against them using word overlap and fuzzy matching. It filters during
    a time window around TTS playback.

    The filter uses multiple strategies:
    1. Word overlap - if most words in transcription appear in stored response
    2. Substring matching - if transcription is part of stored response
    3. Fuzzy matching - for typos and transcription errors
    """

    def __init__(
        self,
        buffer_ms: float = 1500,  # Increased buffer
        similarity_threshold: float = 0.40,  # Lowered threshold for more aggressive filtering
        word_overlap_threshold: float = 0.30,  # 30% of words must match (lowered)
        max_stored_responses: int = 5,  # Increased storage
    ):
        """Initialize echo filter.

        Args:
            buffer_ms: Milliseconds to continue filtering after TTS ends
            similarity_threshold: Similarity ratio (0-1) for fuzzy matching
            word_overlap_threshold: Fraction of words that must overlap
            max_stored_responses: Max responses to keep for comparison
        """
        self._buffer_ms = buffer_ms
        self._similarity_threshold = similarity_threshold
        self._word_overlap_threshold = word_overlap_threshold
        self._max_stored_responses = max_stored_responses

        self._stored_responses: List[str] = []
        self._stored_words: List[Set[str]] = []  # Pre-computed word sets
        self._tts_end_time: Optional[float] = None
        self._tts_active = False

    def set_tts_response(self, text: str, duration_ms: float) -> None:
        """Store TTS response with timing information.

        Args:
            text: The text being spoken by TTS
            duration_ms: Estimated duration of TTS playback in milliseconds
        """
        # Add response to storage
        self._stored_responses.append(text)
        self._stored_words.append(_extract_words(text))

        # Trim to max size
        if len(self._stored_responses) > self._max_stored_responses:
            self._stored_responses.pop(0)
            self._stored_words.pop(0)

        # Calculate when filtering window ends
        total_window = (duration_ms + self._buffer_ms) / 1000  # Convert to seconds
        self._tts_end_time = time.time() + total_window
        self._tts_active = True

        logger.info(
            f"Echo filter: Stored response ({duration_ms:.0f}ms + {self._buffer_ms}ms buffer): "
            f"'{text[:60]}...'"
        )

    def is_echo(self, transcription: str) -> bool:
        """Check if transcription is likely echo of stored TTS response.

        Uses multiple matching strategies to determine if the transcription
        is echo or genuine user input.

        Args:
            transcription: The transcribed text to check

        Returns:
            True if likely echo, False if likely user input
        """
        # Check if we're outside the filtering window
        if self._tts_end_time is None:
            return False

        if time.time() > self._tts_end_time:
            if self._tts_active:
                logger.debug("Echo filter: Window expired, clearing filter")
                self._tts_active = False
            return False

        if not self._stored_responses:
            return False

        # Normalize transcription for comparison
        trans_clean = transcription.lower().strip()

        if not trans_clean or len(trans_clean) < 2:
            return False

        trans_words = _extract_words(transcription)

        logger.debug(
            f"Echo check: '{transcription}' vs {len(self._stored_responses)} stored responses"
        )

        # Check against each stored response
        for i, response in enumerate(self._stored_responses):
            resp_clean = response.lower().strip()
            resp_words = self._stored_words[i]

            logger.debug(f"  Comparing to response {i}: '{response[:50]}...'")

            # Strategy 1: Word overlap
            if trans_words and resp_words:
                overlap = trans_words & resp_words
                overlap_ratio = len(overlap) / len(trans_words) if trans_words else 0

                logger.debug(
                    f"    Word overlap: {len(overlap)}/{len(trans_words)} = {overlap_ratio:.0%} (threshold: {self._word_overlap_threshold:.0%})"
                )

                if overlap_ratio >= self._word_overlap_threshold:
                    logger.info(
                        f"Echo filtered (word overlap {overlap_ratio:.0%}): '{transcription}'"
                    )
                    return True

            # Strategy 2: Substring matching
            if trans_clean in resp_clean:
                logger.info(f"Echo filtered (substring): '{transcription}'")
                return True

            # Strategy 3: Check if any significant portion matches
            # Split response into chunks and check if transcription matches any
            resp_chunks = [
                resp_clean[i : i + len(trans_clean)]
                for i in range(0, len(resp_clean) - len(trans_clean) + 1, 10)
            ]
            for chunk in resp_chunks:
                chunk_sim = SequenceMatcher(None, trans_clean, chunk).ratio()
                if chunk_sim > 0.7:
                    logger.info(f"Echo filtered (chunk match {chunk_sim:.0%}): '{transcription}'")
                    return True

            # Strategy 4: Fuzzy matching for whole strings
            similarity = SequenceMatcher(None, trans_clean, resp_clean).ratio()
            if similarity > self._similarity_threshold:
                logger.info(f"Echo filtered (fuzzy {similarity:.0%}): '{transcription}'")
                return True

        logger.debug(f"Echo filter: Not echo - '{transcription}'")
        return False

    def clear(self) -> None:
        """Clear all stored responses and reset timing."""
        self._stored_responses.clear()
        self._stored_words.clear()
        self._tts_end_time = None
        self._tts_active = False
        logger.debug("Echo filter cleared")

    def get_stored_responses(self) -> List[str]:
        """Get list of currently stored responses.

        Returns:
            List of stored TTS response texts
        """
        return self._stored_responses.copy()
