import numpy as np
import librosa
import noisereduce as nr

# ============================================
# TOGGLE EACH STEP ON / OFF
# ============================================

ENABLE_NOISE_REDUCTION = True
ENABLE_SILENCE_TRIM    = True
ENABLE_NORMALIZATION   = True
ENABLE_VAD             = True

# ============================================
# INDIVIDUAL STEPS
# ============================================

def reduce_noise(array: np.ndarray, sr: int) -> np.ndarray:
    """Suppress background noise, hold music, and static."""
    return nr.reduce_noise(y=array, sr=sr)

def trim_silence(array: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Cut dead air from the start and end of the clip."""
    trimmed, _ = librosa.effects.trim(array, top_db=top_db)
    return trimmed

def normalize_volume(array: np.ndarray) -> np.ndarray:
    """Bring all speakers to the same loudness using RMS normalization."""
    rms = np.sqrt(np.mean(array ** 2))
    if rms > 0:
        return array / rms * 0.1
    return array

def apply_vad(array: np.ndarray, sr: int, frame_ms: int = 30, energy_threshold: float = 0.01) -> np.ndarray:
    """
    Voice Activity Detection — keeps only frames where speech is present.
    Drops silent gaps and non-speech segments in the middle of the audio.
    Uses energy-based detection: frames below the threshold are discarded.
    """
    frame_len = int(sr * frame_ms / 1000)
    frames = [
        array[i: i + frame_len]
        for i in range(0, len(array) - frame_len, frame_len)
    ]
    voiced = [f for f in frames if np.sqrt(np.mean(f ** 2)) >= energy_threshold]
    if not voiced:
        return array  # nothing passed VAD — return original to avoid empty audio
    return np.concatenate(voiced)

# ============================================
# MASTER FUNCTION
# ============================================

def preprocess(array: np.ndarray, sr: int) -> np.ndarray:
    """
    Run all enabled preprocessing steps on 8kHz audio.
    Toggle steps at the top of this file.
    """
    if ENABLE_NOISE_REDUCTION:
        array = reduce_noise(array, sr)

    if ENABLE_SILENCE_TRIM:
        array = trim_silence(array)

    if ENABLE_NORMALIZATION:
        array = normalize_volume(array)

    if ENABLE_VAD:
        array = apply_vad(array, sr)

    return array
