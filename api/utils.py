"""
Utility functions for text processing and validation.
"""

import re
import string
from typing import Optional
from .config import settings


def process_text(text: str) -> str:
    """
    Process text for model input.

    Applies basic cleaning while preserving semantic content:
    - Removes extra whitespace
    - Normalizes line breaks
    - Preserves punctuation and capitalization

    Args:
        text: Raw input text

    Returns:
        Processed text ready for embedding
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Normalize common unicode issues
    text = text.replace('\u2019', "'")  # Replace smart apostrophe
    text = text.replace('\u201c', '"')  # Replace smart quote
    text = text.replace('\u201d', '"')  # Replace smart quote
    text = text.replace('\u2013', '-')  # Replace en dash
    text = text.replace('\u2014', '-')  # Replace em dash

    return text


def validate_text(text: str, min_length: Optional[int] = None, max_length: Optional[int] = None) -> bool:
    """
    Validate text meets minimum requirements.

    Args:
        text: Text to validate
        min_length: Minimum required length (defaults to settings.MIN_TEXT_LENGTH)
        max_length: Maximum allowed length (defaults to settings.MAX_TEXT_LENGTH)

    Returns:
        True if text is valid, False otherwise
    """
    if not text or not isinstance(text, str):
        return False

    # Use settings defaults if not specified
    min_len = min_length if min_length is not None else settings.MIN_TEXT_LENGTH
    max_len = max_length if max_length is not None else settings.MAX_TEXT_LENGTH

    # Check length
    text_len = len(text.strip())
    if text_len < min_len or text_len > max_len:
        return False

    # Check if text has actual content (not just whitespace/punctuation)
    text_stripped = text.strip()
    text_alphanum = ''.join(c for c in text_stripped if c.isalnum())

    if len(text_alphanum) < min_len // 2:  # At least half should be alphanumeric
        return False

    return True


def clean_text(text: str) -> str:
    """
    Clean text for display purposes.

    More aggressive cleaning than process_text, useful for logging/display.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def get_text_stats(text: str) -> dict:
    """
    Get basic statistics about a text.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with text statistics
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)

    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'avg_sentence_length': len(words) / max(len([s for s in sentences if s.strip()]), 1)
    }
