# CLAUDE.md

## Project Overview
POC for call centre Speech-to-Text in South Indian regional languages (Tamil, Telugu, Kannada, Malayalam). Compares Whisper (via Groq) vs Sarvam AI on conversational audio resampled to 8kHz to simulate telephony conditions.

## Key Files
- `app.py` — Streamlit app, upload audio files, transcribe via Sarvam, download Excel
- `poc_call_centre_stt.py` — evaluation script, compares Whisper vs Sarvam, computes WER
- `preprocessing.py` — audio preprocessing pipeline, toggled via booleans at top of file
- `download_samples.py` — downloads samples from `ai4bharat/indicvoices_r` to `samples/`
- `results.json` — saved WER results from last evaluation run

## Environment
- Python env: conda env named `tts`
- All API keys in `.env` — never hardcode them
- Run with: `conda run -n tts python <script>` or activate env first

## Dataset
- `ai4bharat/indicvoices_r` — uses full language names as config: `"Tamil"`, `"Telugu"`, `"Kannada"`, `"Malayalam"`
- Ground truth column: `normalized` (fall back to `text`)
- Audio: 48kHz, streaming=True to avoid bulk download
- Samples saved locally under `samples/` with `samples/metadata.json`

## APIs
- Whisper: Groq API, OpenAI-compatible client, model `whisper-large-v3`
- Sarvam: POST `https://api.sarvam.ai/speech-to-text`, header `api-subscription-key`, fields `model=saaras:v3`, `mode=transcribe` or `mode=translate`
- Sarvam response fields: `transcript`, `language_code`, `request_id`

## Audio Pipeline Order
Load audio → Resample to 8kHz → Preprocess → Send to STT

## Preprocessing Toggles
All 4 steps in `preprocessing.py` are independently toggled at the top of the file:
`ENABLE_NOISE_REDUCTION`, `ENABLE_SILENCE_TRIM`, `ENABLE_NORMALIZATION`, `ENABLE_VAD`
