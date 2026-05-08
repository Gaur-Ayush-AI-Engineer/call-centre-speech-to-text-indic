# Call Centre Speech-to-Text

Build a Speech-to-Text (STT) pipeline for call centre audio in **South Indian regional languages** (Tamil, Telugu, Kannada, Malayalam). The goal was to evaluate how well existing STT systems handle real-world call centre conditions and demonstrate a practical approach to the problem.

---

## Why This Is Hard

Call centre audio is fundamentally different from standard speech:

- **Telephone audio quality** — recorded at 8kHz (narrow-band), not studio quality
- **Regional languages** — Tamil, Telugu, Kannada, Malayalam are underserved by general-purpose models like Whisper
- **Code-switching** — agents and customers mix English with regional languages (e.g. "sir mee account lo problem undi")
- **Noise** — background noise, hold music, overlapping speech, codec compression

---

## Approach

### 1. Dataset
Used `ai4bharat/indicvoices_r` — speech dataset by AI4Bharat (IIT Madras), covering all 4 South Indian languages. Chosen over Mozilla Common Voice (read speech, thin coverage) and BhasaAnuvaad (formal/educational content).

Note: IndicVoices-R is a prompted recording corpus processed with speech enhancement — it is near-studio quality, not real call recordings. WER numbers will be better than production call audio.

### 2. Telephony Simulation
Audio goes through a three-stage telephony simulation (`telephony_sim.py`) before transcription:
- **8kHz resampling** — narrows bandwidth to match telephone audio
- **G.711 mu-law codec** — applies the quantization distortion real phone calls have (encode → decode round-trip)
- **VoIP packet loss** — randomly zeros out 20ms chunks at 2% probability, simulating dropped network packets

### 3. Audio Preprocessing
A dedicated preprocessing pipeline (`preprocessing.py`) runs after telephony simulation:
- **Silence trimming** — removes dead air from start and end
- **Noise reduction** — suppresses background noise and static
- **Voice Activity Detection (VAD)** — keeps only speech segments using Silero VAD (neural network, far more robust than energy-threshold VAD)
- **Volume normalization** — equalizes loudness across speakers, applied last so VAD can see original energy differences

Each step can be toggled independently at the top of `preprocessing.py`.

### 4. STT Comparison
Three systems evaluated side by side:
- **Whisper-large-v3** via Groq API — general-purpose, industry standard
- **Sarvam AI saaras:v3** — purpose-built for Indian languages
- **IndicConformer-600M** (`ai4bharat/indic-conformer-600m-multilingual`) — open-source model trained on 22 Indian languages, runs locally on CPU

### 5. Evaluation Metrics
- **WER (Word Error Rate)** — percentage of words transcribed incorrectly. Lower is better.
- **CER (Character Error Rate)** — percentage of characters wrong. More meaningful for Tamil and Malayalam which are agglutinative (one word carries meaning English needs 4–5 words for).

Both metrics use text normalization (lowercase, strip punctuation) before scoring so encoding differences don't count as errors.

### 6. Speaker Diarization
The Streamlit app includes a diarization tab (powered by `pyannote/speaker-diarization-3.1`) that separates a two-speaker call recording into labelled turns:
```
Speaker 1 [0.0s – 3.2s]: sir mera account band ho gaya
Speaker 2 [3.5s – 6.1s]: aapka account number kya hai
```

---

## Results

Evaluated on 20 samples per language under telephony simulation (8kHz + G.711 + packet loss) and original quality (48kHz) conditions. Full per-sample scores in `results.json`.

### Telephony simulation — 8kHz

| Language | Whisper WER | Whisper CER | Sarvam WER | Sarvam CER | IndicConformer WER | IndicConformer CER |
|---|---|---|---|---|---|---|
| Tamil | 0.88 | 0.42 | 0.51 | 0.17 | 0.67 | 0.22 |
| Telugu | 0.88 | 0.34 | 0.37 | 0.10 | 0.36 | 0.09 |
| Kannada | 0.90 | 0.40 | 0.51 | 0.17 | 0.54 | 0.13 |
| Malayalam | 1.16 | 0.87 | 0.45 | 0.14 | 0.53 | 0.12 |

### Original quality — 48kHz

| Language | Whisper WER | Whisper CER | Sarvam WER | Sarvam CER | IndicConformer WER | IndicConformer CER |
|---|---|---|---|---|---|---|
| Tamil | 0.87 | 0.38 | 0.41 | 0.13 | 0.53 | 0.17 |
| Telugu | 0.82 | 0.33 | 0.37 | 0.09 | 0.36 | 0.09 |
| Kannada | 0.91 | 0.41 | 0.49 | 0.16 | 0.53 | 0.13 |
| Malayalam | 1.13 | 0.85 | 0.46 | 0.12 | 0.48 | 0.11 |

**Key findings:**
- Sarvam outperforms Whisper by 40–60% on WER across all 4 languages and both conditions
- Whisper WER > 1.0 on Malayalam means it hallucinated more words than existed — in one case transcribing Malayalam as Punjabi script entirely
- CER gap is even more dramatic than WER for Tamil and Malayalam, confirming the agglutinative language problem
- IndicConformer (open-source, local) is competitive with Sarvam — note potential data overlap with `indicvoices_r` since both are AI4Bharat products
- 8kHz telephony degradation is modest for Sarvam, suggesting it was trained on noisy/telephony-like data

---

## Project Structure

```
├── app.py                  # Streamlit web app — transcribe, diarize, compare results
├── poc_call_centre_stt.py  # Evaluation script — compares 3 models with WER + CER
├── preprocessing.py        # Audio preprocessing pipeline (Silero VAD, noise reduction, normalization)
├── telephony_sim.py        # Telephony simulation (8kHz resample + G.711 codec + packet loss)
├── indic_conformer.py      # IndicConformer-600M model loading and transcription
├── download_samples.py     # Downloads 20 samples per language from indicvoices_r locally
├── inspect_dataset.py      # Dataset inspection utility
├── results.json            # Saved evaluation results (WER + CER for all 3 models)
├── samples/                # Local audio samples (not committed)
├── .env                    # API keys (not committed)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with:
```
GROQ_API_KEY=your_groq_key
HF_TOKEN=your_huggingface_token
SARVAM_API_KEY=your_sarvam_key
```

Accept gated model licenses on HuggingFace (one-time, requires login):
- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`
- `ai4bharat/indic-conformer-600m-multilingual`

Download samples:
```bash
python download_samples.py
```

Run evaluation:
```bash
python poc_call_centre_stt.py
```

Run the web app:
```bash
streamlit run app.py
```

---

## Known Gaps

- **No real call centre data** — `indicvoices_r` is near-studio quality prompted speech, not actual call recordings. Real call audio will show higher WER for all systems.
- **Code-switching not evaluated** — Tanglish/Tenglish mixed speech is common in call centres but not in the dataset.
- **Single speaker per sample** — real calls have two speakers with potential overlap. The diarization tab addresses this for the app demo but the eval script does not.
- **IndicConformer data overlap** — AI4Bharat built both `indicvoices_r` and IndicConformer. The model may have been trained on this dataset, making its eval scores optimistic.
- **Bandpass filter missing** — PSTN telephony transmits 300–3400 Hz. A bandpass filter would more accurately simulate production conditions.
- **No latency measurement** — transcription latency per second of audio is not captured anywhere.
