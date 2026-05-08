import os
import json
import time
import warnings
import tempfile
import requests
import numpy as np
import torch
import soundfile as sf
import pandas as pd
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from preprocessing import preprocess
from telephony_sim import simulate_telephony

warnings.filterwarnings("ignore", message="Accessing `__path__` from")

load_dotenv()

# ============================================
# CONFIG
# ============================================

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEPHONY_SAMPLE_RATE = 8000
SARVAM_MAX_SECONDS = 60  # Sarvam saaras:v3 rejects requests longer than 60s

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

def summarize(text: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the following call centre conversation in 2-3 sentences in English. Be concise and focus on what was discussed and any resolution."},
            {"role": "user", "content": text},
        ],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()

def call_sarvam(temp_wav: str, mode: str) -> dict:
    time.sleep(1.1)  # proactive throttle: Sarvam allows 60 req/min = 1/s
    for attempt in range(5):
        with open(temp_wav, "rb") as audio_file:
            response = requests.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": ("audio.wav", audio_file, "audio/wav")},
                data={"model": "saaras:v3", "mode": mode},
                timeout=30
            )
        if response.status_code == 429:
            print(f"[429] attempt={attempt} body={response.text}")
            time.sleep(2 ** attempt)  # fallback if throttle wasn't enough
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()

def _sarvam_call_array(audio_8k: np.ndarray, mode: str) -> dict:
    """Write an in-memory 8kHz array to a temp WAV and call Sarvam. Returns full response dict."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_wav = f.name
    try:
        sf.write(temp_wav, audio_8k, TELEPHONY_SAMPLE_RATE)
        return call_sarvam(temp_wav, mode)
    finally:
        os.unlink(temp_wav)

def _sarvam_chunked(audio_8k: np.ndarray, mode: str) -> tuple[str, str]:
    """
    Send audio to Sarvam in ≤60s chunks to stay within the API limit.
    Returns (stitched_transcript, language_code_from_first_chunk).
    """
    max_samples = SARVAM_MAX_SECONDS * TELEPHONY_SAMPLE_RATE
    chunks = [
        audio_8k[i : i + max_samples]
        for i in range(0, len(audio_8k), max_samples)
    ]
    chunks = [c for c in chunks if len(c) >= TELEPHONY_SAMPLE_RATE * 0.3]

    if not chunks:
        return "", ""

    parts = []
    lang_code = ""
    for i, chunk in enumerate(chunks):
        resp = _sarvam_call_array(chunk, mode)
        parts.append(resp.get("transcript", "").strip())
        if i == 0:
            lang_code = resp.get("language_code", "")

    return " ".join(p for p in parts if p), lang_code

def transcribe(audio_bytes: bytes, filename: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[-1], delete=False) as f:
        f.write(audio_bytes)
        temp_input = f.name

    try:
        array, orig_sr = sf.read(temp_input)
        array = array.astype(np.float32)
        if array.ndim == 2:
            array = array.mean(axis=1)

        # Auto-detect number of speakers using pyannote
        diar_pipeline = load_diarization_pipeline()
        waveform = torch.from_numpy(array[np.newaxis, :])
        diar_result = diar_pipeline({"waveform": waveform, "sample_rate": orig_sr})
        diarization = diar_result.speaker_diarization
        unique_speakers = {label for _, _, label in diarization.itertracks(yield_label=True)}

        # ── Single speaker ──────────────────────────────────────────────
        if len(unique_speakers) <= 1:
            audio_8k, _ = simulate_telephony(array, orig_sr)
            audio_8k = preprocess(audio_8k, TELEPHONY_SAMPLE_RATE)
            transcript, lang_code = _sarvam_chunked(audio_8k, "transcribe")
            if lang_code.startswith("en"):
                translation = transcript
            else:
                translation, _ = _sarvam_chunked(audio_8k, "translate")
            return {
                "filename": filename,
                "language": LANGUAGE_CODE_MAP.get(lang_code, lang_code or "Unknown"),
                "transcription": transcript,
                "english_translation": translation,
                "summary": summarize(translation),
                "error": None,
            }

        # ── Multi-speaker: per-turn transcribe + translate ───────────────
        speaker_map = {}
        raw_turns = []
        detected_lang = ""

        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if speaker_label not in speaker_map:
                speaker_map[speaker_label] = f"Speaker {len(speaker_map) + 1}"

            start_sample = int(turn.start * orig_sr)
            end_sample = int(turn.end * orig_sr)
            segment = array[start_sample:end_sample]

            if len(segment) == 0:
                continue

            transcript, translation, lang_code = _transcribe_segment_full(segment, orig_sr, detected_lang)
            if lang_code and not detected_lang:
                detected_lang = lang_code
            if not transcript:
                continue

            raw_turns.append({
                "speaker": speaker_map[speaker_label],
                "start": round(turn.start, 1),
                "end": round(turn.end, 1),
                "transcript": transcript,
                "translation": translation,
            })

        # Merge consecutive same-speaker turns with gap ≤ 1.5s
        merged = []
        for turn in raw_turns:
            if (merged
                    and merged[-1]["speaker"] == turn["speaker"]
                    and turn["start"] - merged[-1]["end"] <= 1.5):
                merged[-1]["end"] = turn["end"]
                merged[-1]["transcript"] = merged[-1]["transcript"].rstrip() + " " + turn["transcript"]
                merged[-1]["translation"] = merged[-1]["translation"].rstrip() + " " + turn["translation"]
            else:
                merged.append(dict(turn))

        transcription_block = "\n".join(
            f"{t['speaker']} [{t['start']}s]: {t['transcript']}" for t in merged
        )
        translation_block = "\n".join(
            f"{t['speaker']} [{t['start']}s]: {t['translation']}" for t in merged
        )

        return {
            "filename": filename,
            "language": LANGUAGE_CODE_MAP.get(detected_lang, detected_lang or "Unknown"),
            "transcription": transcription_block,
            "english_translation": translation_block,
            "summary": summarize(translation_block),
            "error": None,
        }

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
# DIARIZATION
# ============================================

@st.cache_resource(show_spinner="Loading speaker diarization model...")
def load_diarization_pipeline():
    from pyannote.audio import Pipeline
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )

def transcribe_segment(array: np.ndarray, sr: int) -> str:
    """Resample segment to 8kHz, preprocess, send to Sarvam in ≤60s chunks, return transcript."""
    audio_8k, _ = simulate_telephony(array, sr)
    audio_8k = preprocess(audio_8k, TELEPHONY_SAMPLE_RATE)

    if len(audio_8k) < TELEPHONY_SAMPLE_RATE * 0.3:
        return ""

    transcript, _ = _sarvam_chunked(audio_8k, "transcribe")
    return transcript

def _transcribe_segment_full(array: np.ndarray, sr: int, fallback_lang: str = "") -> tuple[str, str, str]:
    """Returns (transcript_original, translation_english, lang_code) for one speaker segment."""
    audio_8k, _ = simulate_telephony(array, sr)
    audio_8k = preprocess(audio_8k, TELEPHONY_SAMPLE_RATE)

    if len(audio_8k) < TELEPHONY_SAMPLE_RATE * 0.3:
        return "", "", fallback_lang

    transcript, lang_code = _sarvam_chunked(audio_8k, "transcribe")
    lang = lang_code or fallback_lang

    if lang.startswith("en"):
        translation = transcript
    else:
        translation, _ = _sarvam_chunked(audio_8k, "translate")

    return transcript, translation, lang_code

def run_diarization(audio_bytes: bytes, filename: str) -> list[dict]:
    """
    Returns list of turns: [{"speaker": "Speaker 1", "start": 0.0, "end": 2.3, "text": "..."}, ...]
    """
    pipeline = load_diarization_pipeline()

    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[-1], delete=False) as f:
        f.write(audio_bytes)
        temp_input = f.name

    try:
        array, orig_sr = sf.read(temp_input)
        array = array.astype(np.float32)
        if array.ndim == 2:
            array = array.mean(axis=1)

        # Pass waveform directly to avoid pyannote's file-reader sample-count mismatch bug
        waveform = torch.from_numpy(array[np.newaxis, :])
        result = pipeline({"waveform": waveform, "sample_rate": orig_sr})

        # DiarizeOutput wraps the Annotation in .speaker_diarization
        diarization = result.speaker_diarization

        speaker_map = {}
        raw_turns = []

        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if speaker_label not in speaker_map:
                speaker_map[speaker_label] = f"Speaker {len(speaker_map) + 1}"

            start_sample = int(turn.start * orig_sr)
            end_sample = int(turn.end * orig_sr)
            segment = array[start_sample:end_sample]

            if len(segment) == 0:
                continue

            text = transcribe_segment(segment, orig_sr)
            if not text:
                continue

            raw_turns.append({
                "speaker": speaker_map[speaker_label],
                "start": round(turn.start, 1),
                "end": round(turn.end, 1),
                "text": text,
            })

        # Merge consecutive turns from the same speaker if gap is ≤ 1.5s
        turns = []
        for turn in raw_turns:
            if (turns
                    and turns[-1]["speaker"] == turn["speaker"]
                    and turn["start"] - turns[-1]["end"] <= 1.5):
                turns[-1]["end"] = turn["end"]
                turns[-1]["text"] = turns[-1]["text"].rstrip() + " " + turn["text"]
            else:
                turns.append(dict(turn))

        return turns

    finally:
        os.unlink(temp_input)

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(page_title="Call Centre STT", layout="wide")
st.title("Call Centre — Speech to Text")
st.caption("Supports Tamil, Telugu, Kannada, Malayalam and other Indian languages. Audio is resampled to 8kHz (telephony simulation) before transcription.")

tab1, tab2, tab3 = st.tabs(["Transcribe", "Diarize (Speaker View)", "Model Comparison Results"])

# ============================================
# TAB 3 — BENCHMARK RESULTS FROM results.json
# ============================================

with tab3:
    st.subheader("Whisper-large-v3 vs Sarvam saaras:v3 vs IndicConformer-600M")
    st.caption("Average WER and CER per language across 20 samples from ai4bharat/indicvoices_r. Lower is better.")

    RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    LANG_NAMES = {"ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam"}
    MODELS = [
        ("whisper",         "Whisper-large-v3"),
        ("sarvam",          "Sarvam saaras:v3"),
        ("indic_conformer", "IndicConformer-600M"),
    ]

    def avg(scores):
        return round(sum(scores) / len(scores), 2) if scores else "N/A"

    def build_rows(mode_data: dict) -> pd.DataFrame:
        rows = []
        for code, name in LANG_NAMES.items():
            lang_data = mode_data.get(code, {})
            row = {"Language": name}
            for key, label in MODELS:
                row[f"{label} WER"] = avg(lang_data.get(key, {}).get("wer", []))
                row[f"{label} CER"] = avg(lang_data.get(key, {}).get("cer", []))
            rows.append(row)
        return pd.DataFrame(rows).set_index("Language")

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            benchmark = json.load(f)

        tel = benchmark.get("telephony_8kHz", {})
        ori = benchmark.get("original_48kHz", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Telephony simulation — 8kHz**")
            st.dataframe(build_rows(tel), use_container_width=True)
        with col2:
            st.markdown("**Original audio quality — 48kHz**")
            st.dataframe(build_rows(ori), use_container_width=True)

        st.info("WER > 1.0 means the model hallucinated more words than were present in the original audio. CER (Character Error Rate) is more meaningful for Tamil and Malayalam which are agglutinative languages.")
    else:
        st.warning("results.json not found. Run `python poc_call_centre_stt.py` first to generate benchmark results.")

# ============================================
# TAB 1 — TRANSCRIBE
# ============================================

with tab1:

    uploaded_files = st.file_uploader(
        "Upload audio file(s)",
        type=["wav", "mp3", "ogg", "m4a", "flac"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected. Click **Transcribe** to process.")

        for file in uploaded_files:
            st.audio(file, format=f"audio/{file.name.rsplit('.', 1)[-1]}")
            file.seek(0)

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
                        "Summary": result["summary"],
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

# ============================================
# TAB 2 — DIARIZE
# ============================================

with tab2:
    st.subheader("Speaker-separated transcript")
    st.caption("Upload call recording(s). The app will separate who said what, turn by turn, and summarise each call.")

    if not HF_TOKEN:
        st.error("HF_TOKEN not found in .env. Add your HuggingFace token to use diarization.")
    else:
        diar_files = st.file_uploader(
            "Upload call recording(s)",
            type=["wav", "mp3", "ogg", "m4a", "flac"],
            accept_multiple_files=True,
            key="diar_uploader",
        )

        if diar_files:
            st.info(f"{len(diar_files)} file(s) selected.")
            for f in diar_files:
                st.audio(f, format=f"audio/{f.name.rsplit('.', 1)[-1]}")
                f.seek(0)

            if st.button("Diarize & Transcribe", type="primary"):
                all_turns_rows = []
                summary_rows = []
                errors = []

                progress = st.progress(0, text="Starting...")
                status = st.empty()

                for idx, diar_file in enumerate(diar_files):
                    status.text(f"Processing: {diar_file.name} ({idx + 1}/{len(diar_files)})")
                    try:
                        turns = run_diarization(diar_file.read(), diar_file.name)

                        if not turns:
                            errors.append(f"{diar_file.name}: No speech segments detected.")
                        else:
                            speaker_colors = {"Speaker 1": "🔵", "Speaker 2": "🟠"}
                            st.markdown(f"**{diar_file.name}** — {len(turns)} turns")
                            for turn in turns:
                                icon = speaker_colors.get(turn["speaker"], "⚪")
                                st.markdown(f"{icon} **{turn['speaker']}** `{turn['start']}s – {turn['end']}s`")
                                st.markdown(f"> {turn['text']}")

                            # Build conversation text for summary
                            conversation_text = "\n".join(
                                f"{t['speaker']} [{t['start']}s]: {t['text']}" for t in turns
                            )
                            file_summary = summarize(conversation_text)
                            st.info(f"**Summary:** {file_summary}")

                            for turn in turns:
                                all_turns_rows.append({
                                    "Filename": diar_file.name,
                                    "Speaker": turn["speaker"],
                                    "Start (s)": turn["start"],
                                    "End (s)": turn["end"],
                                    "Transcript": turn["text"],
                                })
                            summary_rows.append({
                                "Filename": diar_file.name,
                                "Summary": file_summary,
                            })

                    except Exception as e:
                        errors.append(f"{diar_file.name}: {e}")

                    progress.progress((idx + 1) / len(diar_files), text=f"{idx + 1}/{len(diar_files)} done")

                status.empty()
                progress.empty()

                if errors:
                    with st.expander(f"{len(errors)} file(s) failed", expanded=True):
                        for e in errors:
                            st.error(e)

                if all_turns_rows:
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                        pd.DataFrame(all_turns_rows).to_excel(writer, index=False, sheet_name="Conversation")
                        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="Summaries")
                    excel_buffer.seek(0)

                    st.download_button(
                        label="Download conversation as Excel",
                        data=excel_buffer,
                        file_name="conversation.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
