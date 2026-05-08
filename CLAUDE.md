# CLAUDE.md

## Project Overview
POC for call centre Speech-to-Text in South Indian regional languages (Tamil, Telugu, Kannada, Malayalam). Compares Whisper (via Groq API), Sarvam AI, and IndicConformer-600M on conversational audio put through a telephony simulation pipeline.

## Key Files
- `app.py` — Streamlit app: upload audio to transcribe, diarize two-speaker calls, view model comparison results
- `poc_call_centre_stt.py` — evaluation script, compares all 3 models, computes WER + CER
- `preprocessing.py` — audio preprocessing pipeline, toggled via booleans at top of file
- `telephony_sim.py` — telephony simulation: 8kHz resample + G.711 mu-law codec + VoIP packet loss
- `indic_conformer.py` — IndicConformer-600M model loading and transcription (runs locally)
- `download_samples.py` — downloads samples from `ai4bharat/indicvoices_r` to `samples/`
- `results.json` — saved WER + CER results from last evaluation run

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
- IndicConformer: local via HuggingFace `AutoModel`, needs `HF_TOKEN`, `trust_remote_code=True`, call signature: `model(wav_tensor, lang_code, "ctc")`
- Diarization: `pyannote/speaker-diarization-3.1` via HuggingFace, needs `HF_TOKEN` and gated model access accepted on HF

## Audio Pipeline Order
Load audio → `telephony_sim.simulate_telephony()` → `preprocessing.preprocess()` → Send to STT

## Telephony Simulation (`telephony_sim.py`)
1. Resample to 8kHz
2. Apply G.711 mu-law codec (encode + decode round-trip)
3. Apply VoIP packet loss (2% of 20ms chunks zeroed out)

## Preprocessing Order and Toggles
Order matters — do not change without reason:
1. Silence trim (`ENABLE_SILENCE_TRIM`) — remove edge silence before noise reduction
2. Noise reduction (`ENABLE_NOISE_REDUCTION`) — clean signal before VAD
3. VAD (`ENABLE_VAD`) — detect speech on original energy levels before normalization
4. Normalization (`ENABLE_NORMALIZATION`) — equalize volume last, only on surviving speech

VAD type toggle: `USE_SILERO_VAD = True` uses Silero neural VAD; `False` falls back to basic energy-threshold VAD (kept for comparison).

## Scoring
WER and CER both use `jiwer.Compose([ToLowerCase(), RemovePunctuation(), Strip()])` normalization before scoring. Results stored as `{"wer": [...], "cer": [...]}` per model per language in `results.json`.
