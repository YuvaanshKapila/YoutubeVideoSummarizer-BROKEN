# AI Video Processor

This repository contains a simplified Streamlit application that can:

* Fetch the transcript from any public YouTube video (manual or auto-generated)
* Produce an extractive **or** abstractive (T5-based) summary
* Optionally translate the summary into 10+ languages
* Optionally generate an audio version of the summary using gTTS

---

## Quick start

1.  **Install dependencies**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # if the app does not download it automatically
```

2.  **Run the app**

```bash
streamlit run app.py
```

3.  **Open your browser** – Streamlit will show you the local URL (usually `http://localhost:8501`).

---

## Notes

* The app tries multiple fallback strategies (manual English captions, auto-generated captions) before giving up on a transcript.
* Some videos disable transcripts entirely; in that case the app will show an error.
* Abstractive summarisation relies on a small T5 model (`t5-base`). For large inputs the app truncates to 3,000 characters to avoid GPU/CPU memory issues.
* Audio generation is limited to ~1,500 characters to keep processing fast.
