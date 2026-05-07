# Call Centre Speech-to-Text — POC

## Assignment

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
Used `ai4bharat/indicvoices_r` — conversational speech dataset by AI4Bharat (IIT Madras), covering all 4 South Indian languages. Chosen over Mozilla Common Voice (read speech, thin coverage) and BhasaAnuvaad (formal/educational content).

### 2. Telephony Simulation
All audio is resampled from 48kHz → 8kHz before transcription to simulate actual telephone/VoIP audio quality. This makes the evaluation honest — testing on clean audio would overstate real-world performance.

### 3. Audio Preprocessing
A dedicated preprocessing pipeline (`preprocessing.py`) runs on the 8kHz audio before sending to any STT model:
- **Noise reduction** — suppresses background noise and static
- **Silence trimming** — removes dead air from start and end
- **Volume normalization** — equalizes loudness across different speakers
- **Voice Activity Detection (VAD)** — keeps only speech segments, drops silent gaps

Each step can be toggled independently.

### 4. STT Comparison
Two systems evaluated side by side:
- **Whisper-large-v3** via Groq API — general-purpose, industry standard
- **Sarvam AI saaras:v3** — purpose-built for Indian languages

### 5. Evaluation Metric
Word Error Rate (WER) — measures percentage of words transcribed incorrectly. Lower is better.

---

## Results

Evaluated on 20 samples per language, under both telephony (8kHz) and original quality (48kHz) conditions:

| Language | Whisper 8kHz | Sarvam 8kHz | Whisper 48kHz | Sarvam 48kHz |
|---|---|---|---|---|
| Tamil | 0.86 | 0.51 | 0.84 | 0.53 |
| Telugu | 0.86 | 0.36 | 0.84 | 0.35 |
| Kannada | 0.91 | 0.57 | 0.94 | 0.60 |
| Malayalam | 1.39 | 0.51 | 1.31 | 0.49 |

**Key finding:** Sarvam outperforms Whisper by 37–60% across all languages. Whisper's Malayalam WER > 1.0 means it hallucinated words that weren't there — in one case transcribing Malayalam as Punjabi script entirely.

---

## Project Structure

```
├── app.py                  # Streamlit web app — upload audio, get transcription + Excel export
├── poc_call_centre_stt.py  # Evaluation script — compares Whisper vs Sarvam with WER scoring
├── preprocessing.py        # Audio preprocessing pipeline (noise reduction, VAD, normalization)
├── download_samples.py     # Downloads 20 samples per language from indicvoices_r locally
├── inspect_dataset.py      # Dataset inspection utility
├── results.json            # Saved evaluation results
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

- **No real call centre data** — `indicvoices_r` is conversational but not actual call recordings. Real call audio will likely show higher WER for both systems.
- **Code-switching not evaluated** — Tanglish/Tenglish mixed speech is common in call centres but not present in the dataset.
- **Single speaker per sample** — real calls have two speakers (agent + customer) with potential overlap.

These are documented intentionally — in production, actual call recordings and speaker diarization would be the next steps.
