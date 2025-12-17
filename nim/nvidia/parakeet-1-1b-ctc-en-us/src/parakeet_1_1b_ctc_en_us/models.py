from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass(slots=True)
class EndpointingOptions:
    """
    Tunable VAD/endpointing thresholds for streaming and offline recognition.
    """

    start_history_seconds: float
    start_threshold: float
    stop_history_seconds: float
    stop_threshold: float
    eou_history_seconds: float
    eou_threshold: float

    @property
    def max_history_seconds(self) -> float:
        """
        Maximum lookback horizon used by any endpointing window.

        Returns:
            Longest window size in seconds across start, stop, and EOU horizons.
        """
        return max(
            self.start_history_seconds,
            self.stop_history_seconds,
            self.eou_history_seconds,
        )


@dataclass(slots=True)
class RecognitionOptions:
    """
    Normalized recognition options derived from Riva RecognitionConfig inputs.
    """

    sample_rate: int
    language_code: str
    include_word_time_offsets: bool
    enable_punctuation: bool
    enable_diarization: bool
    interim_results: bool
    channel_count: int
    max_speaker_count: int
    endpointing: EndpointingOptions


@dataclass
class StreamingState:
    """
    Mutable state maintained during streaming recognition sessions.
    """

    options: RecognitionOptions
    audio_chunks: List[np.ndarray] = field(default_factory=list)
    logits: List[np.ndarray] = field(default_factory=list)
    total_audio_seconds: float = 0.0
    last_emit_time: float = 0.0
    last_transcript: Optional[str] = None
    vad_history: Deque[Tuple[float, float]] = field(default_factory=deque)
    speech_started: bool = False
    endpoint_triggered: bool = False
    last_silence_time: Optional[float] = None
    final_emitted: bool = False

    def append(
        self, audio_chunk: np.ndarray, new_logits: np.ndarray, duration_seconds: float
    ) -> None:
        """
        Append the latest audio chunk and logits to the streaming buffers.

        Args:
            audio_chunk: Raw PCM audio normalized to float32.
            new_logits: Model logits for the provided audio chunk.
            duration_seconds: Duration of the chunk in seconds at the target sample rate.
        """
        self.audio_chunks.append(audio_chunk)
        self.logits.append(new_logits)
        self.total_audio_seconds += duration_seconds

    def merged_logits(self) -> Optional[np.ndarray]:
        """
        Concatenate buffered logits into a single array.

        Returns:
            Concatenated logits or ``None`` when no logits have been collected.
        """
        if not self.logits:
            return None
        return np.concatenate(self.logits, axis=0)

    def merged_audio(self) -> Optional[np.ndarray]:
        """
        Concatenate buffered audio into a single array.

        Returns:
            Concatenated audio or ``None`` when no audio has been collected.
        """
        if not self.audio_chunks:
            return None
        return np.concatenate(self.audio_chunks)


@dataclass(slots=True)
class TritonModelConfig:
    """
    Minimal Triton model metadata surfaced over gRPC and HTTP for compatibility.
    """

    name: str
    version: str
    max_batch_size: int
    sample_rate: int
    language_code: str
