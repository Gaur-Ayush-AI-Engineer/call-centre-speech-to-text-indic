from datasets import load_dataset

dataset = load_dataset(
    "fixie-ai/common_voice_17_0",
    "en",
    split="test",
    streaming=True
)

for i, sample in enumerate(dataset):

    print(sample.keys())

    print("\nTranscript:")
    print(sample["sentence"])

    print("\nAudio:")
    print(sample["audio"])

    if i == 2:
        break