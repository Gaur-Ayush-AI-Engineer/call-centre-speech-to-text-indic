import os
import tempfile
import requests
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from preprocessing import preprocess

load_dotenv()

# ============================================
# CONFIG
# ============================================

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
TELEPHONY_SAMPLE_RATE = 8000

LANGUAGE_CODE_MAP = {
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "hi-IN": "Hindi",
    "en-IN": "English (India)",
    "en-US": "English (US)",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
    "bn-IN": "Bengali",
    "pa-IN": "Punjabi",
    "or-IN": "Odia",
}

# ============================================
# CORE FUNCTIONS
# ============================================

def resample_to_telephony(array: np.ndarray, orig_sr: int) -> np.ndarray:
    return librosa.resample(array.astype(np.float32), orig_sr=orig_sr, target_sr=TELEPHONY_SAMPLE_RATE)

def call_sarvam(temp_wav: str, mode: str) -> dict:
    with open(temp_wav, "rb") as audio_file:
        response = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": ("audio.wav", audio_file, "audio/wav")},
            data={"model": "saaras:v3", "mode": mode},
            timeout=30
        )
    response.raise_for_status()
    return response.json()

def transcribe(audio_bytes: bytes, filename: str) -> dict:
    """Load audio, resample to 8kHz, send to Sarvam, return result dict."""
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[-1], delete=False) as f:
        f.write(audio_bytes)
        temp_input = f.name

    try:
        array, orig_sr = sf.read(temp_input)
        array = array.astype(np.float32)

        if array.ndim == 2:
            array = array.mean(axis=1)

        audio_8k = resample_to_telephony(array, orig_sr)
        audio_8k = preprocess(audio_8k, TELEPHONY_SAMPLE_RATE)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name
        sf.write(temp_wav, audio_8k, TELEPHONY_SAMPLE_RATE)

        try:
            transcribe_data = call_sarvam(temp_wav, "transcribe")
            translate_data = call_sarvam(temp_wav, "translate")

            lang_code = transcribe_data.get("language_code", "")
            return {
                "filename": filename,
                "language": LANGUAGE_CODE_MAP.get(lang_code, lang_code or "Unknown"),
                "transcription": transcribe_data.get("transcript", ""),
                "english_translation": translate_data.get("transcript", ""),
                "error": None,
            }
        finally:
            os.unlink(temp_wav)

    except Exception as e:
        return {
            "filename": filename,
            "language": "—",
            "transcription": "—",
            "english_translation": "—",
            "error": str(e),
        }
    finally:
        os.unlink(temp_input)

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(page_title="Call Centre STT", layout="wide")
st.title("Call Centre — Speech to Text")
st.caption("Supports Tamil, Telugu, Kannada, Malayalam and other Indian languages. Audio is resampled to 8kHz (telephony simulation) before transcription.")

uploaded_files = st.file_uploader(
    "Upload audio file(s)",
    type=["wav", "mp3", "ogg", "m4a", "flac"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} file(s) selected. Click **Transcribe** to process.")

    if st.button("Transcribe", type="primary"):
        rows = []
        errors = []

        progress = st.progress(0, text="Starting...")
        status = st.empty()

        for idx, file in enumerate(uploaded_files):
            status.text(f"Processing: {file.name} ({idx + 1}/{len(uploaded_files)})")
            result = transcribe(file.read(), file.name)

            if result["error"]:
                errors.append(f"{file.name}: {result['error']}")
            else:
                rows.append({
                    "Filename": result["filename"],
                    "Language": result["language"],
                    "Transcription": result["transcription"],
                    "English Translation": result["english_translation"],
                })

            progress.progress((idx + 1) / len(uploaded_files), text=f"{idx + 1}/{len(uploaded_files)} done")

        status.empty()
        progress.empty()

        if errors:
            with st.expander(f"{len(errors)} file(s) failed", expanded=True):
                for e in errors:
                    st.error(e)

        if rows:
            df = pd.DataFrame(rows)

            st.success(f"Transcribed {len(rows)} file(s) successfully.")
            st.dataframe(df, use_container_width=True)

            # Excel export
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Transcriptions")
            excel_buffer.seek(0)

            st.download_button(
                label="Download Excel",
                data=excel_buffer,
                file_name="transcriptions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
