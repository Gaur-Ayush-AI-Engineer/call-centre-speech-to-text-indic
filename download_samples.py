import os
import json
import soundfile as sf
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

LANGUAGES = {
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
}

SAMPLES_PER_LANG = 20
SAVE_DIR = "samples"

os.makedirs(SAVE_DIR, exist_ok=True)
metadata = {}

for lang_code, lang_name in LANGUAGES.items():

    lang_dir = os.path.join(SAVE_DIR, lang_code)
    os.makedirs(lang_dir, exist_ok=True)
    metadata[lang_code] = []

    print(f"\nDownloading {SAMPLES_PER_LANG} samples for {lang_name}...")

    dataset = load_dataset(
        "ai4bharat/indicvoices_r",
        lang_name,
        split="train",
        streaming=True,
        token=HF_TOKEN
    )

    for i, sample in enumerate(dataset):
        if i >= SAMPLES_PER_LANG:
            break

        audio = sample["audio"]
        wav_path = os.path.join(lang_dir, f"sample_{i}.wav")
        sf.write(wav_path, audio["array"], audio["sampling_rate"])

        metadata[lang_code].append({
            "file": wav_path,
            "sampling_rate": audio["sampling_rate"],
            "ground_truth": sample.get("normalized") or sample.get("text", ""),
            "duration_sec": round(len(audio["array"]) / audio["sampling_rate"], 2),
            "scenario": sample.get("scenario", ""),
            "task_name": sample.get("task_name", ""),
        })

        print(f"  [{i+1}/{SAMPLES_PER_LANG}] saved {wav_path} ({metadata[lang_code][-1]['duration_sec']}s)")

with open(os.path.join(SAVE_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\nDone. All samples saved to ./{SAVE_DIR}/")
print(f"Metadata saved to ./{SAVE_DIR}/metadata.json")
