"""
Text processing utilities for TTS
"""

import gc
import re
from typing import List, Optional

import torch

from app.config import Config


def split_text_into_chunks(text: str, max_length: int = None) -> list:
    """Split text into manageable chunks for TTS processing"""
    if max_length is None:
        max_length = Config.MAX_CHUNK_LENGTH

    if len(text) <= max_length:
        return [text]

    # Try to split at sentence boundaries first
    sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
    chunks = []
    current_chunk = ""

    # Split into sentences
    sentences = []
    temp_text = text

    while temp_text:
        best_split = len(temp_text)
        best_ending = ""

        for ending in sentence_endings:
            pos = temp_text.find(ending)
            if pos != -1 and pos < best_split:
                best_split = pos + len(ending)
                best_ending = ending

        if best_split == len(temp_text):
            # No sentence ending found, take the rest
            sentences.append(temp_text)
            break
        else:
            sentences.append(temp_text[:best_split])
            temp_text = temp_text[best_split:]

    # Group sentences into chunks
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If single sentence is too long, split it further
            if len(sentence) > max_length:
                # Split at commas, semicolons, etc.
                sub_delimiters = [", ", "; ", " - ", " — "]
                sub_chunks = [sentence]

                for delimiter in sub_delimiters:
                    new_sub_chunks = []
                    for chunk in sub_chunks:
                        if len(chunk) <= max_length:
                            new_sub_chunks.append(chunk)
                        else:
                            parts = chunk.split(delimiter)
                            current_part = ""
                            for part in parts:
                                if (
                                    len(current_part) + len(delimiter) + len(part)
                                    <= max_length
                                ):
                                    current_part += (
                                        delimiter if current_part else ""
                                    ) + part
                                else:
                                    if current_part:
                                        new_sub_chunks.append(current_part)
                                    current_part = part
                            if current_part:
                                new_sub_chunks.append(current_part)
                    sub_chunks = new_sub_chunks

                # Add sub-chunks
                for sub_chunk in sub_chunks:
                    if len(sub_chunk) <= max_length:
                        chunks.append(sub_chunk.strip())
                    else:
                        # Last resort: split by words
                        words = sub_chunk.split()
                        current_word_chunk = ""
                        for word in words:
                            if len(current_word_chunk) + len(word) + 1 <= max_length:
                                current_word_chunk += (
                                    " " if current_word_chunk else ""
                                ) + word
                            else:
                                if current_word_chunk:
                                    chunks.append(current_word_chunk)
                                current_word_chunk = word
                        if current_word_chunk:
                            chunks.append(current_word_chunk)
                current_chunk = ""
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]

    return chunks


def split_text_for_streaming(
    text: str,
    chunk_size: Optional[int] = None,
    strategy: Optional[str] = None,
    quality: Optional[str] = None,
) -> List[str]:
    """
    Split text into chunks optimized for streaming with different strategies.

    Args:
        text: Input text to split
        chunk_size: Target chunk size (characters)
        strategy: Splitting strategy ('sentence', 'paragraph', 'fixed', 'word')
        quality: Quality preset ('fast', 'balanced', 'high')

    Returns:
        List of text chunks optimized for streaming
    """
    # Apply quality presets
    if quality:
        if quality == "fast":
            chunk_size = chunk_size or 100
            strategy = strategy or "word"
        elif quality == "balanced":
            chunk_size = chunk_size or 200
            strategy = strategy or "sentence"
        elif quality == "high":
            chunk_size = chunk_size or 300
            strategy = strategy or "paragraph"

    # Set defaults
    chunk_size = chunk_size or 200
    strategy = strategy or "sentence"

    # Apply strategy-specific splitting
    if strategy == "paragraph":
        return _split_by_paragraphs(text, chunk_size)
    elif strategy == "sentence":
        return _split_by_sentences(text, chunk_size)
    elif strategy == "word":
        return _split_by_words(text, chunk_size)
    elif strategy == "fixed":
        return _split_by_fixed_size(text, chunk_size)
    else:
        # Default to sentence splitting
        return _split_by_sentences(text, chunk_size)


def _split_by_paragraphs(text: str, max_length: int) -> List[str]:
    """Split text by paragraph breaks, respecting max length"""
    # Split by double newlines (paragraph breaks)
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # If paragraph fits with current chunk
        if (
            len(current_chunk) + len(paragraph) + 2 <= max_length
        ):  # +2 for paragraph break
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If paragraph is too long, split it by sentences
            if len(paragraph) > max_length:
                sentence_chunks = _split_by_sentences(paragraph, max_length)
                chunks.extend(sentence_chunks)
                current_chunk = ""
            else:
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if chunk.strip()]


def _split_by_sentences(text: str, max_length: int) -> List[str]:
    """Split text by sentence boundaries, respecting max length"""
    # Enhanced sentence splitting regex
    sentence_pattern = r"(?<=[.!?])\s+"
    sentences = re.split(sentence_pattern, text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If sentence fits with current chunk
        if len(current_chunk) + len(sentence) + 1 <= max_length:  # +1 for space
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If sentence is too long, split it further
            if len(sentence) > max_length:
                sub_chunks = _split_long_sentence(sentence, max_length)
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if chunk.strip()]


def _split_by_words(text: str, max_length: int) -> List[str]:
    """Split text by word boundaries, respecting max length"""
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        # If word fits with current chunk
        if len(current_chunk) + len(word) + 1 <= max_length:  # +1 for space
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word
        else:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If single word is too long, force it into its own chunk
            if len(word) > max_length:
                # Split very long words at character boundaries
                for i in range(0, len(word), max_length):
                    chunks.append(word[i : i + max_length])
                current_chunk = ""
            else:
                current_chunk = word

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if chunk.strip()]


def _split_by_fixed_size(text: str, chunk_size: int) -> List[str]:
    """Split text into fixed-size chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _split_long_sentence(sentence: str, max_length: int) -> List[str]:
    """Split a long sentence at natural break points"""
    # Try to split at commas, semicolons, etc.
    delimiters = [", ", "; ", " - ", " — ", ": ", " and ", " or ", " but "]

    chunks = [sentence]

    for delimiter in delimiters:
        new_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                new_chunks.append(chunk)
            else:
                parts = chunk.split(delimiter)
                current_part = ""
                for part in parts:
                    if len(current_part) + len(delimiter) + len(part) <= max_length:
                        current_part += (delimiter if current_part else "") + part
                    else:
                        if current_part:
                            new_chunks.append(current_part)
                        current_part = part
                if current_part:
                    new_chunks.append(current_part)
        chunks = new_chunks

    # Final fallback: split by words
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            word_chunks = _split_by_words(chunk, max_length)
            final_chunks.extend(word_chunks)

    return [chunk.strip() for chunk in final_chunks if chunk.strip()]


def get_streaming_settings(
    streaming_chunk_size: Optional[int],
    streaming_strategy: Optional[str],
    streaming_quality: Optional[str],
) -> dict:
    """
    Get optimized streaming settings based on parameters.

    Returns a dictionary with optimized settings for streaming.
    """
    settings = {
        "chunk_size": streaming_chunk_size or 200,
        "strategy": streaming_strategy or "sentence",
        "quality": streaming_quality or "balanced",
    }

    # Apply quality presets if not explicitly overridden
    if streaming_quality and not streaming_chunk_size:
        if streaming_quality == "fast":
            settings["chunk_size"] = 100
        elif streaming_quality == "high":
            settings["chunk_size"] = 300

    if streaming_quality and not streaming_strategy:
        if streaming_quality == "fast":
            settings["strategy"] = "word"
        elif streaming_quality == "high":
            settings["strategy"] = "paragraph"

    return settings


def concatenate_audio_chunks(audio_chunks: list, sample_rate: int) -> torch.Tensor:
    """Concatenate multiple audio tensors with proper memory management"""
    if len(audio_chunks) == 1:
        return audio_chunks[0]

    # Add small silence between chunks (0.1 seconds)
    silence_samples = int(0.1 * sample_rate)

    # Create silence tensor on the same device as audio chunks
    device = audio_chunks[0].device if hasattr(audio_chunks[0], "device") else "cpu"
    silence = torch.zeros(1, silence_samples, device=device)

    # Use torch.no_grad() to prevent gradient tracking
    with torch.no_grad():
        concatenated = audio_chunks[0]

        for i, chunk in enumerate(audio_chunks[1:], 1):
            # Concatenate current result with silence and next chunk
            concatenated = torch.cat([concatenated, silence, chunk], dim=1)

            # Optional: cleanup intermediate tensors for very long sequences
            if i % 10 == 0:  # Every 10 chunks
                gc.collect()

    # Clean up silence tensor
    del silence

    return concatenated
