import os
import librosa
import numpy as np
import torch
from transformers import AutoModel
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
INDIC_CONFORMER_SR = 16000

_model = None

def _get_model() -> AutoModel:
    global _model
    if _model is None:
        _model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            token=HF_TOKEN,
            trust_remote_code=True,
        )
    return _model

def transcribe(array: np.ndarray, sample_rate: int, lang_code: str) -> str:
    """
    Transcribe audio using IndicConformer running locally on CPU/MPS.
    array       : 1-D float32 numpy array at any sample rate
    sample_rate : sample rate of the input array
    lang_code   : ISO code e.g. "ta", "te", "kn", "ml"
    """
    audio_16k = librosa.resample(array.astype(np.float32), orig_sr=sample_rate, target_sr=INDIC_CONFORMER_SR)
    wav = torch.from_numpy(audio_16k).unsqueeze(0)  # [1, samples]
    return _get_model()(wav, lang_code, "ctc")
