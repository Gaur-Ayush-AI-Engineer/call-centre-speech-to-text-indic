import os
import json
import tempfile
import requests
import numpy as np
import soundfile as sf
import librosa
from openai import OpenAI
from jiwer import wer
from dotenv import load_dotenv
from preprocessing import preprocess

load_dotenv()

SAMPLES_DIR = "samples"
METADATA_PATH = os.path.join(SAMPLES_DIR, "metadata.json")
RESULTS_PATH = "results.json"

# ============================================
# CONFIG
# ============================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

TELEPHONY_SAMPLE_RATE = 8000   # 8kHz to simulate telephone/VoIP audio

LANGUAGES = {
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
}

groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ============================================
# TELEPHONY SIMULATION
# ============================================

def simulate_telephony(array: np.ndarray, orig_sr: int) -> tuple:
    resampled = librosa.resample(array.astype(np.float32), orig_sr=orig_sr, target_sr=TELEPHONY_SAMPLE_RATE)
    return resampled, TELEPHONY_SAMPLE_RATE

# ============================================
# STT: WHISPER VIA GROQ
# ============================================

def transcribe_whisper(array: np.ndarray, sample_rate: int, lang_code: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    try:
        sf.write(temp_path, array, sample_rate)
        with open(temp_path, "rb") as audio_file:
            result = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language=lang_code
            )
        return result.text.strip()
    finally:
        os.unlink(temp_path)

# ============================================
# STT: SARVAM AI
# ============================================

def transcribe_sarvam(array: np.ndarray, sample_rate: int) -> str:
    if SARVAM_API_KEY == "YOUR_SARVAM_API_KEY_HERE":
        return "[SARVAM PLACEHOLDER — API key not set]"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    try:
        sf.write(temp_path, array, sample_rate)
        with open(temp_path, "rb") as audio_file:
            response = requests.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": ("audio.wav", audio_file, "audio/wav")},
                data={"model": "saaras:v3", "mode": "transcribe"}
            )
        response.raise_for_status()
        return response.json().get("transcript", "").strip()
    finally:
        os.unlink(temp_path)

# ============================================
# EVALUATION LOOP — runs for a given audio mode
# ============================================

def run_evaluation(metadata: dict, use_telephony: bool) -> dict:
    mode_label = "telephony_8kHz" if use_telephony else "original_48kHz"
    results = {}

    for lang_code, lang_name in LANGUAGES.items():

        print("\n" + "=" * 65)
        print(f"LANGUAGE : {lang_name} ({lang_code})")
        print(f"MODE     : {'Telephony simulation (8kHz)' if use_telephony else 'Original quality (48kHz)'}")
        print("=" * 65)

        results[lang_code] = {"whisper": [], "sarvam": []}

        samples = metadata.get(lang_code, [])
        if not samples:
            print(f"  [!] No local samples for {lang_name}. Run download_samples.py first.")
            continue

        for i, sample in enumerate(samples):

            print(f"\n  Sample {i + 1}/{len(samples)}")
            print("  " + "-" * 50)

            ground_truth = sample["ground_truth"]
            print(f"  GROUND TRUTH : {ground_truth}")

            array, orig_sr = sf.read(sample["file"])
            array = array.astype(np.float32)

            if use_telephony:
                audio, sr = simulate_telephony(array, orig_sr)
            else:
                audio, sr = array, orig_sr

            audio = preprocess(audio, sr)

            # Whisper
            try:
                whisper_text = transcribe_whisper(audio, sr, lang_code)
                whisper_wer = round(wer(ground_truth, whisper_text), 4)
            except Exception as e:
                whisper_text = f"[ERROR: {e}]"
                whisper_wer = None

            print(f"  WHISPER : {whisper_text}  (WER: {whisper_wer})")
            if whisper_wer is not None:
                results[lang_code]["whisper"].append(whisper_wer)

            # Sarvam
            try:
                sarvam_text = transcribe_sarvam(audio, sr)
                sarvam_wer = round(wer(ground_truth, sarvam_text), 4) if "PLACEHOLDER" not in sarvam_text else None
            except Exception as e:
                sarvam_text = f"[ERROR: {e}]"
                sarvam_wer = None

            print(f"  SARVAM  : {sarvam_text}" + (f"  (WER: {sarvam_wer})" if sarvam_wer is not None else ""))
            if sarvam_wer is not None:
                results[lang_code]["sarvam"].append(sarvam_wer)

    return {mode_label: results}

# ============================================
# MAIN
# ============================================

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(
        "Samples not found. Run `python download_samples.py` first."
    )

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("\n" + "#" * 65)
print("PASS 1 — TELEPHONY SIMULATION (8kHz)")
print("#" * 65)
telephony_results = run_evaluation(metadata, use_telephony=True)

print("\n" + "#" * 65)
print("PASS 2 — ORIGINAL AUDIO QUALITY (48kHz)")
print("#" * 65)
original_results = run_evaluation(metadata, use_telephony=False)

# ============================================
# SAVE RESULTS
# ============================================

all_results = {**telephony_results, **original_results}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to {RESULTS_PATH}")

# ============================================
# SUMMARY TABLE
# ============================================

def avg(scores):
    return f"{sum(scores)/len(scores):.2f}" if scores else "N/A"

tel = telephony_results["telephony_8kHz"]
ori = original_results["original_48kHz"]

print("\n\n" + "=" * 85)
print("RESULTS SUMMARY — Average WER per Language  (lower is better)")
print("=" * 85)
print(f"{'Language':<12} {'Whisper 8kHz':>14} {'Sarvam 8kHz':>13} {'Whisper 48kHz':>15} {'Sarvam 48kHz':>14}")
print("-" * 75)

for lang_code, lang_name in LANGUAGES.items():
    w_tel = avg(tel.get(lang_code, {}).get("whisper", []))
    s_tel = avg(tel.get(lang_code, {}).get("sarvam", []))
    w_ori = avg(ori.get(lang_code, {}).get("whisper", []))
    s_ori = avg(ori.get(lang_code, {}).get("sarvam", []))
    print(f"{lang_name:<12} {w_tel:>14} {s_tel:>13} {w_ori:>15} {s_ori:>14}")

print("=" * 85)
print("\nNOTE: 8kHz = telephony simulation | 48kHz = original recording quality")
print("Code-switching (Tanglish/Tenglish etc.) not covered — known gap for call centers.")
