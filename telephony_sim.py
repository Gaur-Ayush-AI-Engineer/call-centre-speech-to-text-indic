import numpy as np
import librosa

TELEPHONY_SAMPLE_RATE = 8000
MU = 255          # G.711 mu-law constant
PACKET_LOSS_RATE = 0.02   # 2% of 20ms chunks get dropped (realistic VoIP)
PACKET_MS = 20            # standard VoIP packet size

# ============================================
# G.711 MU-LAW CODEC SIMULATION
# ============================================

def apply_g711_codec(array: np.ndarray) -> np.ndarray:
    """
    Encode audio to G.711 mu-law 8-bit and decode back.
    This introduces the same quantization distortion a real phone call has —
    the codec compresses dynamic range and loses fine detail.
    """
    # encode: compress dynamic range using mu-law curve
    compressed = np.sign(array) * np.log1p(MU * np.abs(array)) / np.log1p(MU)
    # quantize to 8-bit (256 levels) — this is the actual damage
    quantized = np.round(compressed * 127) / 127
    # decode: expand back (distortion is now baked in)
    decoded = np.sign(quantized) * (np.expm1(np.abs(quantized) * np.log1p(MU)) / MU)
    return decoded.astype(np.float32)

# ============================================
# VoIP PACKET LOSS SIMULATION
# ============================================

def apply_packet_loss(array: np.ndarray, sr: int) -> np.ndarray:
    """
    Randomly silence 20ms chunks at PACKET_LOSS_RATE probability.
    Simulates VoIP packets dropped due to network congestion or jitter.
    """
    result = array.copy()
    packet_len = int(sr * PACKET_MS / 1000)
    for i in range(0, len(result) - packet_len, packet_len):
        if np.random.random() < PACKET_LOSS_RATE:
            result[i: i + packet_len] = 0.0
    return result

# ============================================
# FULL TELEPHONY SIMULATION PIPELINE
# ============================================

def simulate_telephony(array: np.ndarray, orig_sr: int) -> tuple[np.ndarray, int]:
    """
    Full telephony simulation:
      1. Resample to 8kHz (narrow-band telephone audio)
      2. Apply G.711 mu-law codec distortion
      3. Simulate VoIP packet loss
    Returns (processed_array, 8000).
    """
    audio = librosa.resample(array.astype(np.float32), orig_sr=orig_sr, target_sr=TELEPHONY_SAMPLE_RATE)
    audio = apply_g711_codec(audio)
    audio = apply_packet_loss(audio, TELEPHONY_SAMPLE_RATE)
    return audio, TELEPHONY_SAMPLE_RATE
