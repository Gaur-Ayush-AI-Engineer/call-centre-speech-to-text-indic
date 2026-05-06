import tempfile
import soundfile as sf
from datasets import load_dataset
from openai import OpenAI

# ============================================
# CONFIG
# ============================================

GROQ_API_KEY = ""

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ============================================
# LOAD DATASET
# ============================================

dataset = load_dataset(
    "fixie-ai/common_voice_17_0",
    "en",
    split="test",
    streaming=True
)

# ============================================
# PROCESS SAMPLES
# ============================================

for i, sample in enumerate(dataset):

    if i >= 3:
        break

    print("\n" + "=" * 50)

    print("\nGROUND TRUTH:")
    print(sample["sentence"])

    # ----------------------------------------
    # Decode audio
    # ----------------------------------------

    audio = sample["audio"]

    array = audio.get_all_samples().data
    sampling_rate = audio.metadata.sample_rate

    # ----------------------------------------
    # Save temp wav
    # ----------------------------------------

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=True
    ) as temp_audio:

        sf.write(
            temp_audio.name,
            array,
            sampling_rate
        )

        # ------------------------------------
        # Send to Whisper
        # ------------------------------------

        with open(temp_audio.name, "rb") as audio_file:

            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3"
            )

        print("\nPREDICTED:")
        print(transcription.text)