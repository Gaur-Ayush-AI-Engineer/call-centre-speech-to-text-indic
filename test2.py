import os
import tempfile
import soundfile as sf
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================
# CONFIG
# ============================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

LANGUAGES = {
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam"
}

SAMPLES_PER_LANG = 3

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ============================================
# PROCESS EACH LANGUAGE
# ============================================

for lang_code, lang_name in LANGUAGES.items():

    print("\n" + "=" * 60)
    print(f"LANGUAGE: {lang_name}")
    print("=" * 60)

    dataset = load_dataset(
        "fixie-ai/common_voice_17_0",
        lang_code,
        split=f"test[:{SAMPLES_PER_LANG}]",
        streaming=False,
        token=HF_TOKEN
    )

    for i, sample in enumerate(dataset):

        print("\n" + "-" * 40)

        print("\nGROUND TRUTH:")
        print(sample["sentence"])

        audio = sample["audio"]
        array = audio["array"]
        sampling_rate = audio["sampling_rate"]

        # ----------------------------------------
        # Save temp wav
        # ----------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_audio:
            temp_path = temp_audio.name

        try:
            sf.write(temp_path, array, sampling_rate)

            # ------------------------------------
            # Whisper API
            # ------------------------------------

            with open(temp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3"
                )

            print("\nPREDICTED:")
            print(transcription.text)

        finally:
            os.unlink(temp_path)