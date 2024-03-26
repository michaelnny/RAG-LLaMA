"""Code for chunking text"""

from typing import List
import re


def split_text_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Define a regular expression pattern for splitting sentences
    sentence_enders = re.compile(
        r"""
        # Split sentences on whitespace between them.
        (?:               # Group for two positive lookbehinds.
          (?<=[.!?])      # Either an end of sentence punct,
        | (?<=[.!?]['"])  # or end of sentence punct and quote.
        )                 # End group of two positive lookbehinds.
        (?<!\d\.)         # Don't split if the previous character is a digit followed by a period
        (?<!\.\s)         # Don't split if the previous character is a period followed by whitespace
        (?<!  Mr\.   )    # Don't end sentence on "Mr."
        (?<!  Mrs\.  )    # Don't end sentence on "Mrs."
        (?<!  Jr\.   )    # Don't end sentence on "Jr."
        (?<!  Dr\.   )    # Don't end sentence on "Dr."
        (?<!  Prof\. )    # Don't end sentence on "Prof."
        (?<!  Sr\.   )    # Don't end sentence on "Sr."
        \s+               # Split on whitespace between sentences.
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    sentences = sentence_enders.split(text)
    return sentences


def estimate_word_count(text: str) -> int:
    """Estimates the number of words in a string by splitting on whitespace."""
    if text is None:
        return 0
    words = text.split()
    return len(words)


def split_text_into_chunks(text: str, max_words: int) -> List[str]:
    """Split text into chunks based on maximum number of characters."""
    assert max_words >= 50, max_words

    sentences = split_text_into_sentences(text)
    chunks = []
    current_chunk = ''

    for sentence in sentences:
        if estimate_word_count(current_chunk) + estimate_word_count(sentence) <= max_words:
            if current_chunk:
                current_chunk += ' ' + sentence
            else:
                current_chunk = sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
