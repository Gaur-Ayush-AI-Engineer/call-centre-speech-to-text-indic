For diarization test audio, here are your options ranked by effort:

Easiest — Resemblyzer demo file (no account, one command):


git clone https://github.com/resemble-ai/Resemblyzer
# audio is at: Resemblyzer/audio_data/X2zqiX6yL3I.mp3
Real two-person interview, clean audio, two very distinct voices. This is what most open-source diarization tools use as their demo file.

Quick smoke test — Kaggle Mini Speaker Diarization:

https://www.kaggle.com/datasets/wiradkp/mini-speech-diarization
Free Kaggle account required, small dataset made specifically for diarization testing
Most rigorous — VoxConverse (ground truth labels included):


wget https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip
Real YouTube conversational clips, CC BY 4.0, comes with RTTM annotation files so you can actually verify if your diarization got the speaker boundaries right. ~1.5GB.

Or via HuggingFace (you already have HF token):


from datasets import load_dataset
ds = load_dataset("diarizers-community/voxconverse", split="dev")
My recommendation: grab the Resemblyzer MP3 in 30 seconds to verify your Streamlit diarization tab works end-to-end, then use a VoxConverse clip if you want to show it to a technical audience with ground truth backing it up.