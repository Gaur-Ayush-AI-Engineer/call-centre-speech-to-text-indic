# Independent Code & Approach Review

> Reviewed by an independent agent that read README.md and all source files.
> Items already implemented have been removed from this document.

---

## Remaining Bugs

### `poc_call_centre_stt.py` — No `if __name__ == "__main__":` guard
All evaluation code runs at module level. If anything ever imports from this file, it immediately starts downloading data and making API calls.

---

## Remaining Approach Shortcomings

### `indicvoices_r` is near-studio quality, not real call audio
IndicVoices-R was built as a prompted recording corpus processed with dereverberation and speech enhancement. WER numbers will be materially better than what either system achieves on real call audio. The gap is not quantified.

### 20 samples with no shuffle is statistically weak
The download script takes the first 20 examples from the streaming `train` split with no randomization. This may be biased toward a particular speaker or recording environment. Confidence intervals on WER from 20 samples are large — the results table numbers should not be presented without this caveat.

### Bandpass filter missing from telephony simulation
PSTN telephony transmits 300–3400 Hz. A bandpass filter at this range should be the first step in telephony simulation — it removes sub-300Hz rumble and above-3400Hz noise. G.711 + packet loss are implemented but the bandpass filter is not.

---

## Remaining Missing Features

### No latency measurement anywhere
There is no timing instrumentation in either the app or the eval script. The first question from a technical stakeholder will be "what's the latency per second of audio?" The answer is never captured or displayed.

### No confidence scores
The pipeline outputs a transcript string with no confidence signal. In a call-centre QA context, the entire point is knowing which transcripts need human review.

### Timestamps from Sarvam are being discarded
Sarvam's API response includes a `timestamps` field. The current code discards everything except `transcript`. Showing "at 0:23 the customer said X" would make the demo significantly more compelling.

### `test2.py` should not be in the repo
It is a scratch exploration file using a different dataset, no preprocessing, no WER, no telephony simulation. It undermines the professionalism of the repo.

### Whisper `initial_prompt` not used
Sony AI research (arXiv 2412.19785) shows prepending a few words in the target language as `initial_prompt` to Whisper improves WER by 10–20% on Indian languages with zero fine-tuning. The current code passes only `language=lang_code`.

---

## Prioritized Fix List

| Priority | Fix | File |
|---|---|---|
| 1 | Add `if __name__ == "__main__":` guard | `poc_call_centre_stt.py` |
| 2 | Remove `test2.py` from repo | — |
| 3 | Add bandpass filter (300–3400 Hz) | `telephony_sim.py` |
| 4 | Surface latency numbers | both |
| 5 | Add Whisper `initial_prompt` for Indian languages | `poc_call_centre_stt.py` |
| 6 | Surface Sarvam timestamps in app | `app.py` |

---

*Review based on: README.md, app.py, poc_call_centre_stt.py, preprocessing.py, telephony_sim.py, indic_conformer.py*
*External sources: arXiv 2412.19785 (Whisper prompt-tuning), AI4Bharat IndicConformer, pyannote 4.0*
