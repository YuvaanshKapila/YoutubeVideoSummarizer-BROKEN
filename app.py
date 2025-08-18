import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from heapq import nlargest
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# ----------------------------
# Initial setup
# ----------------------------

st.set_page_config(page_title="AI Video Processor", layout="wide")

# Download NLTK data if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


@st.cache_resource
def load_spacy():
    """Lazy-load the spaCy model; download if absent."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download as spacy_download

        spacy_download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


nlp = load_spacy()


# ----------------------------
# Helper functions
# ----------------------------

def extract_video_id(url: str):
    """Extract the YouTube video id from a url."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_video_info(url: str):
    """Scrape a YouTube watch page for title/description metadata."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        title = soup.find("meta", property="og:title")
        title = title["content"].strip() if title else "YouTube Video"

        desc = soup.find("meta", property="og:description")
        desc = desc["content"].strip() if desc else ""
        return {"title": title, "description": desc}
    except Exception:
        return {"title": "YouTube Video", "description": ""}


def _clean_full_text(text: str):
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_transcript(video_id: str):
    """Retrieve the English transcript for a video. Attempts both manual and auto-generated."""
    languages = ["en", "en-US", "en-GB"]

    # 1. Fast path – direct API call
    try:
        data = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        return _clean_full_text(" ".join([d["text"] for d in data])), data
    except (TranscriptsDisabled, NoTranscriptFound):
        pass  # fallback below
    except Exception as e:
        st.warning(f"Unexpected error while fetching transcript: {e}")

    # 2. Use TranscriptList helper for more granular control
    try:
        tl = YouTubeTranscriptApi.list_transcripts(video_id)

        # Prefer manually created transcripts
        for lang in languages:
            try:
                trans = tl.find_transcript([lang])
                data = trans.fetch()
                return _clean_full_text(" ".join([d["text"] for d in data])), data
            except Exception:
                continue

        # Fallback to generated transcript
        try:
            trans = tl.find_generated_transcript(languages)
            data = trans.fetch()
            return _clean_full_text(" ".join([d["text"] for d in data])), data
        except Exception:
            pass
    except Exception:
        pass

    return None, None


# ---------- Summarisation utilities ----------

def _safe_return(text: str, max_len: int = 800):
    return text[:max_len] + ("…" if len(text) > max_len else "")


def sumy_summarise(text: str, sentences: int = 5):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summariser = LexRankSummarizer()
        summary = summariser(parser.document, sentences)
        return " ".join(str(s) for s in summary)
    except Exception:
        return _safe_return(text)


def nltk_summarise(text: str, ratio: float = 0.3):
    try:
        sens = sent_tokenize(text)
        if len(sens) < 3:
            return text

        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text.lower())

        freq = {}
        for w in words:
            if w.isalnum() and w not in stop_words:
                freq[w] = freq.get(w, 0) + 1
        if not freq:
            return text

        max_freq = max(freq.values())
        freq = {k: v / max_freq for k, v in freq.items()}

        scores = {}
        for i, s in enumerate(sens):
            ws = word_tokenize(s.lower())
            if ws:
                scores[i] = sum(freq.get(w, 0) for w in ws) / len(ws)

        top_n = nlargest(max(1, int(len(sens) * ratio)), scores, key=scores.get)
        top_n.sort()
        return " ".join(sens[i] for i in top_n)
    except Exception:
        return _safe_return(text)


def spacy_summarise(text: str, ratio: float = 0.3):
    try:
        doc = nlp(text)
        sens = list(doc.sents)
        if len(sens) < 3:
            return text

        freq = {}
        for token in doc:
            if token.is_alpha and not token.is_stop and not token.is_punct:
                lemma = token.lemma_.lower()
                freq[lemma] = freq.get(lemma, 0) + 1
        if not freq:
            return text

        max_freq = max(freq.values())
        freq = {k: v / max_freq for k, v in freq.items()}

        scores = {}
        for i, s in enumerate(sens):
            lemmas = [t.lemma_.lower() for t in s if t.lemma_.lower() in freq]
            if lemmas:
                scores[i] = sum(freq[l] for l in lemmas) / len(lemmas)

        top_n = nlargest(max(1, int(len(sens) * ratio)), scores, key=scores.get)
        top_n.sort()
        return " ".join(str(sens[i]) for i in top_n)
    except Exception:
        return _safe_return(text)


@st.cache_resource
def load_t5():
    try:
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        tok = T5Tokenizer.from_pretrained("t5-base")
        return model, tok
    except Exception:
        return None, None


def t5_summarise(text: str):
    model, tok = load_t5()
    if model is None:
        return _safe_return(text, 500)

    if len(text) > 3000:
        text = text[:3000]

    inp = tok.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    out = model.generate(
        inp,
        max_length=200,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tok.decode(out[0], skip_special_tokens=True)


# ---------- Translation & TTS ----------

def translate(text: str, target_lang: str):
    if not text.strip() or target_lang == "en":
        return text

    translator = GoogleTranslator(source="auto", target=target_lang)
    if len(text) > 4500:
        chunks = [text[i : i + 4500] for i in range(0, len(text), 4500)]
        return " ".join(translator.translate(c) for c in chunks)
    return translator.translate(text)


def text_to_speech(text: str, lang: str = "en"):
    if not text.strip():
        return None

    if len(text) > 1500:
        text = text[:1500]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        gTTS(text=text, lang=lang, slow=False).save(fp.name)
        with open(fp.name, "rb") as audio:
            data = audio.read()
        os.unlink(fp.name)
        return data


# ----------------------------
# UI
# ----------------------------

st.title("AI Video Processor")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Summarisation")
    mode = st.radio("Type", ["Extractive", "Abstractive (T5)"])
    if mode == "Extractive":
        algo = st.selectbox("Algorithm", ["Sumy (LexRank)", "NLTK", "spaCy"])
        ratio = st.slider("Summary length (%)", 10, 60, 30) / 100
with col2:
    st.subheader("Output")
    lang_map = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Japanese": "ja",
        "Korean": "ko",
        "Chinese": "zh",
        "Arabic": "ar",
        "Hindi": "hi",
    }
    tgt_lang = st.selectbox("Translate to", list(lang_map.keys()))
    audio_enable = st.checkbox("Generate audio", value=False)
    show_full = st.checkbox("Show full transcript", value=False)


if st.button("Process"):
    if not url:
        st.error("Please enter a YouTube URL.")
        st.stop()

    vid = extract_video_id(url)
    if not vid:
        st.error("Invalid YouTube URL.")
        st.stop()

    prog = st.progress(0)

    st.write("Fetching video info…")
    info = get_video_info(url)
    prog.progress(10)

    st.write("Fetching transcript…")
    text, data = get_transcript(vid)
    prog.progress(40)

    if not text:
        st.error("Could not retrieve transcript for this video. The transcript may be disabled or unavailable.")
        st.stop()

    st.write("Summarising…")
    if mode == "Extractive":
        if algo == "Sumy (LexRank)":
            sent_num = max(3, int(len(sent_tokenize(text)) * ratio))
            summary = sumy_summarise(text, sent_num)
        elif algo == "NLTK":
            summary = nltk_summarise(text, ratio)
        else:
            summary = spacy_summarise(text, ratio)
    else:
        summary = t5_summarise(text)
    prog.progress(70)

    st.write("Translating…")
    tgt_code = lang_map[tgt_lang]
    summary_t = translate(summary, tgt_code)
    prog.progress(85)

    st.header(info["title"])
    st.subheader("Summary")
    st.write(summary_t)

    prog.progress(100)

    if audio_enable:
        st.subheader("Audio")
        with st.spinner("Generating audio…"):
            audio = text_to_speech(summary_t, tgt_code)
            if audio:
                st.audio(audio, format="audio/mp3")
            else:
                st.warning("Failed to generate audio.")

    if show_full:
        st.subheader("Full Transcript")
        st.text_area("Transcript", text, height=400)