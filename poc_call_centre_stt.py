import os
import json
import tempfile
import requests
import numpy as np
import soundfile as sf
from openai import OpenAI
from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation, Strip
from dotenv import load_dotenv
from pipeline.preprocessing import preprocess
from pipeline.telephony_sim import simulate_telephony
import pipeline.indic_conformer as indic_conformer

load_dotenv()

SAMPLES_DIR = "samples"
METADATA_PATH = os.path.join(SAMPLES_DIR, "metadata.json")
RESULTS_PATH = "results.json"

# ============================================
# CONFIG
# ============================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

TELEPHONY_SAMPLE_RATE = 8000

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
# TEXT NORMALIZATION FOR FAIR SCORING
# ============================================

_text_transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])

def compute_scores(reference: str, hypothesis: str) -> tuple[float, float]:
    """Return (WER, CER) after normalizing both strings."""
    ref = _text_transform(reference)
    hyp = _text_transform(hypothesis)
    return round(wer(ref, hyp), 4), round(cer(ref, hyp), 4)

# ============================================
# STT: WHISPER VIA GROQ API
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
    if not SARVAM_API_KEY:
        return "[SARVAM — API key not set]"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    try:
        sf.write(temp_path, array, sample_rate)
        with open(temp_path, "rb") as audio_file:
            response = requests.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": ("audio.wav", audio_file, "audio/wav")},
                data={"model": "saaras:v3", "mode": "transcribe"},
                timeout=30,
            )
        response.raise_for_status()
        return response.json().get("transcript", "").strip()
    finally:
        os.unlink(temp_path)

# ============================================
# EVALUATION LOOP
# ============================================

def run_evaluation(metadata: dict, use_telephony: bool) -> dict:
    mode_label = "telephony_8kHz" if use_telephony else "original_48kHz"
    results = {}

    for lang_code, lang_name in LANGUAGES.items():

        print("\n" + "=" * 70)
        print(f"LANGUAGE : {lang_name} ({lang_code})")
        print(f"MODE     : {'Telephony simulation (8kHz)' if use_telephony else 'Original quality (48kHz)'}")
        print("=" * 70)

        results[lang_code] = {
            "whisper":         {"wer": [], "cer": []},
            "sarvam":          {"wer": [], "cer": []},
            "indic_conformer": {"wer": [], "cer": []},
        }

        samples = metadata.get(lang_code, [])
        if not samples:
            print(f"  [!] No local samples for {lang_name}. Run download_samples.py first.")
            continue

        for i, sample in enumerate(samples):

            print(f"\n  Sample {i + 1}/{len(samples)}")
            print("  " + "-" * 55)

            ground_truth = sample["ground_truth"]
            print(f"  GROUND TRUTH : {ground_truth}")

            array, orig_sr = sf.read(sample["file"])
            array = array.astype(np.float32)

            if use_telephony:
                audio, sr = simulate_telephony(array, orig_sr)
            else:
                audio, sr = array, orig_sr

            audio = preprocess(audio, sr)

            # --- Whisper ---
            try:
                whisper_text = transcribe_whisper(audio, sr, lang_code)
                w_wer, w_cer = compute_scores(ground_truth, whisper_text)
            except Exception as e:
                whisper_text = f"[ERROR: {e}]"
                w_wer = w_cer = None

            print(f"  WHISPER          : {whisper_text}")
            if w_wer is not None:
                print(f"                     WER={w_wer}  CER={w_cer}")
                results[lang_code]["whisper"]["wer"].append(w_wer)
                results[lang_code]["whisper"]["cer"].append(w_cer)

            # --- Sarvam ---
            try:
                sarvam_text = transcribe_sarvam(audio, sr)
                s_wer, s_cer = compute_scores(ground_truth, sarvam_text) if "API key" not in sarvam_text else (None, None)
            except Exception as e:
                sarvam_text = f"[ERROR: {e}]"
                s_wer = s_cer = None

            print(f"  SARVAM           : {sarvam_text}")
            if s_wer is not None:
                print(f"                     WER={s_wer}  CER={s_cer}")
                results[lang_code]["sarvam"]["wer"].append(s_wer)
                results[lang_code]["sarvam"]["cer"].append(s_cer)

            # --- IndicConformer ---
            try:
                indic_text = indic_conformer.transcribe(audio, sr, lang_code)
                i_wer, i_cer = compute_scores(ground_truth, indic_text)
            except Exception as e:
                indic_text = f"[ERROR: {e}]"
                i_wer = i_cer = None

            print(f"  INDIC CONFORMER  : {indic_text}")
            if i_wer is not None:
                print(f"                     WER={i_wer}  CER={i_cer}")
                results[lang_code]["indic_conformer"]["wer"].append(i_wer)
                results[lang_code]["indic_conformer"]["cer"].append(i_cer)

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

print("\n" + "#" * 70)
print("PASS 1 — TELEPHONY SIMULATION (8kHz)")
print("#" * 70)
telephony_results = run_evaluation(metadata, use_telephony=True)

print("\n" + "#" * 70)
print("PASS 2 — ORIGINAL AUDIO QUALITY (48kHz)")
print("#" * 70)
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

MODELS = [
    ("whisper",         "Whisper"),
    ("sarvam",          "Sarvam"),
    ("indic_conformer", "IndicConformer"),
]

for mode_label, mode_data in [("TELEPHONY 8kHz", tel), ("ORIGINAL 48kHz", ori)]:
    print(f"\n\n{'=' * 90}")
    print(f"RESULTS — {mode_label}   (WER / CER, lower is better)")
    print(f"{'=' * 90}")
    print(f"{'Language':<12}", end="")
    for _, label in MODELS:
        print(f"  {label + ' WER':>18} {label + ' CER':>18}", end="")
    print()
    print("-" * 90)

    for lang_code, lang_name in LANGUAGES.items():
        print(f"{lang_name:<12}", end="")
        for key, _ in MODELS:
            w = avg(mode_data.get(lang_code, {}).get(key, {}).get("wer", []))
            c = avg(mode_data.get(lang_code, {}).get(key, {}).get("cer", []))
            print(f"  {w:>18} {c:>18}", end="")
        print()

    print("=" * 90)

print("\nNOTE: 8kHz = telephony simulation | 48kHz = original recording quality")
print("WER > 1.0 means the model hallucinated more words than were in the original audio.")
