"""
Input sanitization module for user questions.
Protects against prompt injection and cleans up user input.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Maximum question length (in characters)
MAX_QUESTION_LENGTH = 1000

# Patterns that might indicate prompt injection attempts
SUSPICIOUS_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
    r"disregard\s+(previous|above|all)\s+(instructions?|prompts?)",
    r"forget\s+(previous|above|all)\s+(instructions?|prompts?)",
    r"you\s+are\s+now\s+a",
    r"new\s+instructions?:",
    r"system\s*:\s*",
    r"assistant\s*:\s*",
    r"<\s*system\s*>",
    r"<\s*/?\s*prompt\s*>",
]


class SanitizationError(Exception):
    """Raised when sanitization fails or input is rejected."""
    pass


def sanitize_question(question):
    """
    Sanitize user question input.

    Args:
        question: Raw user question string

    Returns:
        Cleaned and safe question string

    Raises:
        SanitizationError: If the question is invalid or potentially malicious
    """
    # Check if question exists
    if question is None:
        logger.warning("Received None as question")
        raise SanitizationError("Question cannot be empty")

    # Convert to string if not already
    if not isinstance(question, str):
        question = str(question)

    # Strip whitespace
    question = question.strip()

    # Check if empty after stripping
    if not question:
        logger.warning("Received empty question after stripping")
        raise SanitizationError("Question cannot be empty")

    # Check length
    if len(question) > MAX_QUESTION_LENGTH:
        logger.warning(f"Question too long: {len(question)} chars (max {MAX_QUESTION_LENGTH})")
        raise SanitizationError(f"Question too long. Maximum {MAX_QUESTION_LENGTH} characters allowed.")

    # Check for suspicious patterns (potential prompt injection)
    question_lower = question.lower()
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, question_lower):
            logger.warning(f"Suspicious pattern detected in question: {pattern}")
            raise SanitizationError("Question contains disallowed content. Please rephrase your question.")

    # Remove excessive whitespace (multiple spaces, tabs, newlines)
    question = re.sub(r'\s+', ' ', question)

    # Remove control characters (except normal whitespace)
    question = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', question)

    logger.debug(f"Question sanitized successfully: '{question[:50]}...'")
    return question


def sanitize_for_prompt(text):
    """
    Additional sanitization for text going into LLM prompts.
    This is a lighter sanitization for context text.

    Args:
        text: Text to include in prompt

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text
