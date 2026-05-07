import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

dataset = load_dataset(
    "ai4bharat/indicvoices_r",
    "Tamil",                # full name required, not language code
    split="train",
    streaming=True,
    token=HF_TOKEN
)

sample = next(iter(dataset))
print("COLUMNS:", list(sample.keys()))
print("\nFIRST SAMPLE KEYS & VALUES:")
for key, value in sample.items():
    if key == "audio":
        print(f"  audio -> array shape: {value['array'].shape}, sampling_rate: {value['sampling_rate']}")
    else:
        print(f"  {key} -> {value}")
