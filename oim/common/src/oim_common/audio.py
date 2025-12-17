from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def pcm16_to_float32(payload: bytes) -> np.ndarray:
    """
    Decode signed PCM16 bytes into normalized float32 mono samples.
    """
    if not payload:
        raise ValueError("audio payload is empty")
    samples = np.frombuffer(payload, dtype=np.int16)
    if samples.size == 0:
        raise ValueError("audio payload contained no samples")
    return samples.astype(np.float32) / 32768.0


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """
    Naive resampling using linear interpolation; suitable for small payloads.
    """
    if audio.size == 0 or source_rate <= 0 or target_rate <= 0:
        return audio
    if source_rate == target_rate:
        return audio
    duration_seconds = float(audio.shape[0]) / float(source_rate)
    target_length = max(1, int(math.ceil(duration_seconds * float(target_rate))))
    source_indices = np.linspace(0.0, float(audio.shape[0] - 1), num=audio.shape[0])
    target_indices = np.linspace(0.0, float(audio.shape[0] - 1), num=target_length)
    return np.interp(target_indices, source_indices, audio).astype(np.float32)


def trim_audio_with_vad(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int,
    start_threshold: float,
    stop_threshold: float,
) -> Tuple[np.ndarray, float]:
    """
    Placeholder VAD trimming that currently passes audio through.
    """
    _ = (sample_rate, frame_ms, start_threshold, stop_threshold)
    if audio.size == 0:
        raise ValueError("audio payload is empty")
    return audio, 0.0


def pipeline_vad_probabilities(
    history: Iterable[Tuple[float, float]], window: int = 5
) -> List[float]:
    """
    Surface recent VAD probabilities collected during streaming.
    """
    tail = list(history)[-window:]
    return [probability for _, probability in tail]


def estimate_speaker_segments(
    audio: np.ndarray, sample_rate: int, max_speakers: int
) -> List[Tuple[float, float]]:
    """
    Produce simple, evenly split speaker segments for diarization.
    """
    if audio.size == 0 or sample_rate <= 0 or max_speakers <= 0:
        return []
    duration_seconds = float(audio.shape[0]) / float(sample_rate)
    segment_length = duration_seconds / float(max_speakers)
    segments: List[Tuple[float, float]] = []
    start = 0.0
    for index in range(max_speakers):
        end = duration_seconds if index == max_speakers - 1 else start + segment_length
        segments.append((start, end))
        start = end
    return segments


def assign_speakers(
    offsets: Sequence[dict[str, float]],
    segments: Sequence[Tuple[float, float]],
) -> dict[int, int]:
    """
    Map word offsets to speaker identifiers based on simple segment boundaries.
    """
    assignments: dict[int, int] = {}
    if not offsets or not segments:
        return assignments
    for index, entry in enumerate(offsets):
        start = float(entry.get("start", 0.0))
        end = float(entry.get("end", start))
        midpoint = (start + end) / 2.0
        for speaker_index, (segment_start, segment_end) in enumerate(segments):
            if segment_start <= midpoint <= segment_end:
                assignments[index] = speaker_index + 1
                break
        if index not in assignments:
            assignments[index] = (index % len(segments)) + 1
    return assignments
