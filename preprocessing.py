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
USE_SILERO_VAD         = True   # True = Silero neural VAD | False = basic energy-threshold VAD

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
        return np.clip(array / rms * 0.1, -1.0, 1.0)
    return array

def apply_vad(array: np.ndarray, sr: int, frame_ms: int = 30, energy_threshold: float = 0.01) -> np.ndarray:
    """
    Basic energy-based VAD. Keeps frames whose amplitude exceeds energy_threshold.
    Simple but fooled by loud noise and quiet speech.
    Kept for comparison — use apply_silero_vad() for production.
    """
    frame_len = int(sr * frame_ms / 1000)
    frames = [
        array[i: i + frame_len]
        for i in range(0, len(array), frame_len)
        if len(array[i: i + frame_len]) == frame_len
    ]
    voiced = [f for f in frames if np.sqrt(np.mean(f ** 2)) >= energy_threshold]
    if not voiced:
        return array
    return np.concatenate(voiced)

def apply_silero_vad(array: np.ndarray, sr: int) -> np.ndarray:
    """
    Neural VAD using Silero VAD (snakers4/silero-vad).
    Detects speech by understanding what voice sounds like, not just volume.
    Handles quiet speakers and noisy backgrounds far better than energy-based VAD.
    Requires sr == 8000 or sr == 16000.
    """
    import torch

    supported = {8000, 16000}
    orig_sr = sr
    if sr not in supported:
        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
        sr = 16000

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        verbose=False,
    )
    get_speech_timestamps, _, _, _, collect_chunks = utils

    tensor = torch.from_numpy(array)
    timestamps = get_speech_timestamps(tensor, model, sampling_rate=sr)

    if not timestamps:
        return array
    speech = collect_chunks(timestamps, tensor).numpy()

    # resample back to the original rate so the caller's sample rate stays valid
    if sr != orig_sr:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=orig_sr)
    return speech

# ============================================
# MASTER FUNCTION
# ============================================

def preprocess(array: np.ndarray, sr: int) -> np.ndarray:
    """
    Run all enabled preprocessing steps.
    Order: silence trim → noise reduction → VAD → normalization.
    Toggle steps and VAD type at the top of this file.
    """
    if ENABLE_SILENCE_TRIM:
        array = trim_silence(array)

    if ENABLE_NOISE_REDUCTION:
        array = reduce_noise(array, sr)

    if ENABLE_VAD:
        if USE_SILERO_VAD:
            array = apply_silero_vad(array, sr)
        else:
            array = apply_vad(array, sr)

    if ENABLE_NORMALIZATION:
        array = normalize_volume(array)

    return array
