"""
Content classifier for guitar transcription.

Analyzes audio to classify sections as melodic (solos, melodies),
chordal (strumming, arpeggios), or hybrid (mixed content).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
import logging

import librosa
import numpy as np

_LOG = logging.getLogger(__name__)

# Classification thresholds (empirically calibrated, can be tuned)
_PITCH_DISPERSION_MELODIC_THRESHOLD = 4.0  # semitones std - high = melodic
_PITCH_DISPERSION_CHORDAL_THRESHOLD = 2.0  # semitones std - low = chordal
_ONSET_DENSITY_CHORDAL_THRESHOLD = 6.0  # onsets/sec - high simultaneous = chordal
_ONSET_DENSITY_MELODIC_THRESHOLD = 3.0  # onsets/sec - moderate = melodic
_PERIODICITY_CHORDAL_THRESHOLD = 0.4  # autocorr peak - high = chordal
_HARMONIC_RATIO_MELODIC_THRESHOLD = 0.6  # harmonic/total - high = melodic


class ContentType(str, Enum):
    """Type of musical content in a segment."""
    MELODIC = "melodic"
    CHORDAL = "chordal"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class ContentSegment:
    """A classified segment of musical content."""
    start_time_s: float
    end_time_s: float
    content_type: Literal["melodic", "chordal", "hybrid"]
    confidence: float
    metrics: dict = field(default_factory=dict)


def _compute_onset_density(y: np.ndarray, sr: int) -> float:
    """Compute onset density (onsets per second)."""
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        duration = len(y) / sr
        if duration <= 0:
            return 0.0
        return len(onsets) / duration
    except Exception:
        return 0.0


def _compute_pitch_dispersion(y: np.ndarray, sr: int) -> float:
    """
    Compute pitch dispersion (std of detected pitches in semitones).
    High dispersion suggests melodic content with varied pitches.
    Low dispersion suggests chordal content with repeated patterns.
    """
    try:
        # Use pyin for pitch tracking
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('E2'),
            fmax=librosa.note_to_hz('E6'),
            sr=sr,
        )
        # Filter to voiced frames only
        voiced_pitches = f0[voiced_flag]
        if len(voiced_pitches) < 2:
            return 0.0
        # Convert to MIDI and compute std
        midi_pitches = librosa.hz_to_midi(voiced_pitches[voiced_pitches > 0])
        if len(midi_pitches) < 2:
            return 0.0
        return float(np.std(midi_pitches))
    except Exception:
        return 0.0


def _compute_periodicity(y: np.ndarray, sr: int) -> float:
    """
    Compute rhythmic periodicity via autocorrelation.
    High periodicity suggests repetitive strumming patterns.
    """
    try:
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if len(onset_env) < 10:
            return 0.0
        # Normalize
        onset_env = onset_env - np.mean(onset_env)
        norm = np.linalg.norm(onset_env)
        if norm < 1e-6:
            return 0.0
        onset_env = onset_env / norm
        # Autocorrelation
        autocorr = np.correlate(onset_env, onset_env, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        # Find peak in reasonable tempo range (60-200 BPM)
        hop_length = 512
        min_lag = int(sr * 60 / (200 * hop_length))  # 200 BPM
        max_lag = int(sr * 60 / (60 * hop_length))   # 60 BPM
        min_lag = max(1, min(min_lag, len(autocorr) - 1))
        max_lag = max(min_lag + 1, min(max_lag, len(autocorr)))
        if max_lag <= min_lag:
            return 0.0
        peak = np.max(autocorr[min_lag:max_lag])
        return float(max(0.0, min(1.0, peak)))
    except Exception:
        return 0.0


def _compute_harmonic_ratio(y: np.ndarray, sr: int) -> float:
    """
    Compute ratio of harmonic to percussive energy.
    High ratio suggests sustained melodic content.
    Low ratio suggests percussive strumming.
    """
    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_energy = np.sum(y_harm ** 2)
        perc_energy = np.sum(y_perc ** 2)
        total = harm_energy + perc_energy
        if total < 1e-9:
            return 0.5
        return float(harm_energy / total)
    except Exception:
        return 0.5


def _classify_segment(
    pitch_dispersion: float,
    onset_density: float,
    periodicity: float,
    harmonic_ratio: float,
) -> tuple[ContentType, float]:
    """
    Classify a segment based on computed metrics.
    Returns (content_type, confidence).
    """
    # Score for each type (higher = more likely)
    melodic_score = 0.0
    chordal_score = 0.0

    # Pitch dispersion: high = melodic, low = chordal
    if pitch_dispersion >= _PITCH_DISPERSION_MELODIC_THRESHOLD:
        melodic_score += 2.0
    elif pitch_dispersion <= _PITCH_DISPERSION_CHORDAL_THRESHOLD:
        chordal_score += 2.0
    else:
        melodic_score += 0.5
        chordal_score += 0.5

    # Onset density: moderate = melodic (single notes), high = chordal (strums)
    if onset_density >= _ONSET_DENSITY_CHORDAL_THRESHOLD:
        chordal_score += 1.5
    elif onset_density <= _ONSET_DENSITY_MELODIC_THRESHOLD:
        melodic_score += 1.0
    else:
        melodic_score += 0.5
        chordal_score += 0.5

    # Periodicity: high = chordal (repetitive patterns)
    if periodicity >= _PERIODICITY_CHORDAL_THRESHOLD:
        chordal_score += 1.5
    else:
        melodic_score += 0.5

    # Harmonic ratio: high = melodic (sustained notes)
    if harmonic_ratio >= _HARMONIC_RATIO_MELODIC_THRESHOLD:
        melodic_score += 1.0
    else:
        chordal_score += 0.5

    # Determine type based on scores
    total_score = melodic_score + chordal_score
    if total_score < 1e-6:
        return ContentType.HYBRID, 0.5

    score_diff = abs(melodic_score - chordal_score)
    confidence = min(1.0, score_diff / total_score + 0.3)

    if melodic_score > chordal_score * 1.3:
        return ContentType.MELODIC, confidence
    elif chordal_score > melodic_score * 1.3:
        return ContentType.CHORDAL, confidence
    else:
        return ContentType.HYBRID, max(0.3, confidence - 0.2)


def analyze_musical_content(
    y: np.ndarray,
    sr: int,
    *,
    window_sec: float = 3.0,
    hop_sec: float = 1.5,
    min_segment_sec: float = 1.0,
) -> list[ContentSegment]:
    """
    Analyze audio to classify content as melodic, chordal, or hybrid.

    Args:
        y: Audio signal (mono)
        sr: Sample rate
        window_sec: Analysis window size in seconds
        hop_sec: Hop between windows in seconds
        min_segment_sec: Minimum segment duration to output

    Returns:
        List of ContentSegment with classification and metrics
    """
    duration = len(y) / sr
    if duration < min_segment_sec:
        # Too short - classify entire segment
        pitch_disp = _compute_pitch_dispersion(y, sr)
        onset_dens = _compute_onset_density(y, sr)
        periodicity = _compute_periodicity(y, sr)
        harm_ratio = _compute_harmonic_ratio(y, sr)
        content_type, confidence = _classify_segment(
            pitch_disp, onset_dens, periodicity, harm_ratio
        )
        return [
            ContentSegment(
                start_time_s=0.0,
                end_time_s=duration,
                content_type=content_type.value,
                confidence=confidence,
                metrics={
                    "pitch_dispersion": pitch_disp,
                    "onset_density": onset_dens,
                    "periodicity": periodicity,
                    "harmonic_ratio": harm_ratio,
                },
            )
        ]

    # Analyze windows
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    raw_segments: list[tuple[float, float, ContentType, float, dict]] = []

    pos = 0
    while pos < len(y):
        end_pos = min(pos + window_samples, len(y))
        if end_pos - pos < sr * 0.5:  # Skip very short trailing windows
            break

        window = y[pos:end_pos]
        start_time = pos / sr
        end_time = end_pos / sr

        # Compute metrics for this window
        pitch_disp = _compute_pitch_dispersion(window, sr)
        onset_dens = _compute_onset_density(window, sr)
        periodicity = _compute_periodicity(window, sr)
        harm_ratio = _compute_harmonic_ratio(window, sr)

        content_type, confidence = _classify_segment(
            pitch_disp, onset_dens, periodicity, harm_ratio
        )

        raw_segments.append((
            start_time,
            end_time,
            content_type,
            confidence,
            {
                "pitch_dispersion": pitch_disp,
                "onset_density": onset_dens,
                "periodicity": periodicity,
                "harmonic_ratio": harm_ratio,
            },
        ))

        pos += hop_samples

    if not raw_segments:
        return [
            ContentSegment(
                start_time_s=0.0,
                end_time_s=duration,
                content_type=ContentType.HYBRID.value,
                confidence=0.5,
                metrics={},
            )
        ]

    # Merge consecutive segments of same type
    merged: list[ContentSegment] = []
    current_start = raw_segments[0][0]
    current_end = raw_segments[0][1]
    current_type = raw_segments[0][2]
    current_conf_sum = raw_segments[0][3]
    current_metrics_list = [raw_segments[0][4]]
    segment_count = 1

    for start, end, ctype, conf, metrics in raw_segments[1:]:
        if ctype == current_type:
            # Extend current segment
            current_end = end
            current_conf_sum += conf
            current_metrics_list.append(metrics)
            segment_count += 1
        else:
            # Finalize current segment
            avg_metrics = {}
            for key in current_metrics_list[0]:
                avg_metrics[key] = float(np.mean([m[key] for m in current_metrics_list]))
            merged.append(
                ContentSegment(
                    start_time_s=current_start,
                    end_time_s=current_end,
                    content_type=current_type.value,
                    confidence=current_conf_sum / segment_count,
                    metrics=avg_metrics,
                )
            )
            # Start new segment
            current_start = start
            current_end = end
            current_type = ctype
            current_conf_sum = conf
            current_metrics_list = [metrics]
            segment_count = 1

    # Finalize last segment
    avg_metrics = {}
    for key in current_metrics_list[0]:
        avg_metrics[key] = float(np.mean([m[key] for m in current_metrics_list]))
    merged.append(
        ContentSegment(
            start_time_s=current_start,
            end_time_s=current_end,
            content_type=current_type.value,
            confidence=current_conf_sum / segment_count,
            metrics=avg_metrics,
        )
    )

    # Filter out very short segments by merging with neighbors
    final: list[ContentSegment] = []
    for seg in merged:
        if seg.end_time_s - seg.start_time_s < min_segment_sec and final:
            # Merge with previous segment
            prev = final[-1]
            avg_conf = (prev.confidence + seg.confidence) / 2
            # Keep the type of the longer segment
            prev_dur = prev.end_time_s - prev.start_time_s
            seg_dur = seg.end_time_s - seg.start_time_s
            keep_type = prev.content_type if prev_dur >= seg_dur else seg.content_type
            final[-1] = ContentSegment(
                start_time_s=prev.start_time_s,
                end_time_s=seg.end_time_s,
                content_type=keep_type,
                confidence=avg_conf,
                metrics=prev.metrics,  # Keep previous metrics
            )
        else:
            final.append(seg)

    _LOG.info(
        "Content analysis: %d segments (melodic=%d, chordal=%d, hybrid=%d)",
        len(final),
        sum(1 for s in final if s.content_type == "melodic"),
        sum(1 for s in final if s.content_type == "chordal"),
        sum(1 for s in final if s.content_type == "hybrid"),
    )

    return final
