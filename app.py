import re
import requests
import streamlit as st
from typing import Optional, List, Dict, Tuple
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk


# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


st.set_page_config(page_title="YouTube Transcript Summarizer", layout="wide")


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract the 11-char YouTube video ID from a URL or return the input if it already looks like an ID."""
    candidate = url_or_id.strip()

    # If already 11-char ID
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", candidate):
        return candidate

    try:
        parsed = urlparse(candidate)
    except Exception:
        return None

    if parsed.netloc.endswith("youtu.be") and parsed.path:
        vid = parsed.path.lstrip("/")
        return vid[:11] if re.fullmatch(r"[0-9A-Za-z_-]{11}", vid[:11]) else None

    if parsed.netloc.endswith("youtube.com"):
        if parsed.path == "/watch":
            query = parse_qs(parsed.query)
            vid_list = query.get("v", [])
            if vid_list:
                vid = vid_list[0]
                return vid[:11] if re.fullmatch(r"[0-9A-Za-z_-]{11}", vid[:11]) else None
        # /embed/<id> or /v/<id>
        m = re.match(r"^/(embed|v)/([0-9A-Za-z_-]{11})", parsed.path)
        if m:
            return m.group(2)

    # Fallback: generic regex
    m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", candidate)
    if m:
        return m.group(1)
    return None


@st.cache_data(show_spinner=False)
def fetch_video_title(video_url: str) -> str:
    """Get a reliable title via YouTube oEmbed; fallback to generic if it fails."""
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": video_url, "format": "json"},
            timeout=10,
        )
        if resp.ok:
            return resp.json().get("title", "YouTube Video")
    except Exception:
        pass
    return "YouTube Video"


@st.cache_data(show_spinner=False)
def fetch_transcript_data(video_id: str) -> Tuple[List[Dict], str, bool]:
    """Return (entries, language_code, is_generated). Entries is a list of {text, start, duration}.

    Tries: English first; otherwise any available language.
    """
    # Prefer English directly
    try:
        entries = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US", "en-GB"]
        )
        return entries, "en", True
    except TranscriptsDisabled as e:
        raise RuntimeError("Transcripts are disabled for this video.") from e
    except NoTranscriptFound:
        pass
    except VideoUnavailable as e:
        raise RuntimeError("Video is unavailable.") from e
    except Exception:
        # Continue to list transcripts fallback
        pass

    # Fallback: search transcript list, try English then anything
    try:
        tlist = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try to find an English transcript (manual or auto)
        try:
            t = tlist.find_transcript(["en", "en-US", "en-GB"])
            return t.fetch(), t.language_code, t.is_generated
        except Exception:
            pass

        # Otherwise fetch the first available transcript (any language)
        for t in tlist:
            try:
                entries = t.fetch()
                return entries, t.language_code, t.is_generated
            except Exception:
                continue
    except TranscriptsDisabled as e:
        raise RuntimeError("Transcripts are disabled for this video.") from e
    except VideoUnavailable as e:
        raise RuntimeError("Video is unavailable.") from e
    except Exception:
        pass

    raise RuntimeError(
        "No transcript was found. It may be disabled by the uploader or not available for this video."
    )


def build_full_transcript(entries: List[Dict]) -> str:
    """Join transcript entries into a readable transcript with sentence-like spacing."""
    if not entries:
        return ""
    parts: list[str] = []
    for e in entries:
        text = e.get("text", "").strip()
        if text:
            parts.append(text)
    transcript = " ".join(parts)
    # Normalize whitespace
    transcript = re.sub(r"\s+", " ", transcript).strip()
    return transcript


def summarize_lexrank(text: str, num_sentences: int) -> str:
    if not text:
        return ""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    try:
        summary_sentences = summarizer(parser.document, num_sentences)
        return " ".join(str(s) for s in summary_sentences)
    except Exception:
        # Fallback: truncate if summarizer fails for any reason
        return text[:1000]


st.title("YouTube Transcript Summarizer")
st.write(
    "Paste a YouTube URL to extract the transcript and generate a concise summary."
)

with st.form("input_form"):
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    col_a, col_b = st.columns(2)
    with col_a:
        num_sentences = st.slider("Summary length (sentences)", 3, 15, 6)
    with col_b:
        show_video = st.checkbox("Show video", value=True)
    submitted = st.form_submit_button("Process")

if submitted:
    if not url.strip():
        st.error("Please enter a YouTube URL.")
    else:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Could not parse a valid YouTube video ID from the URL.")
        else:
            with st.spinner("Fetching transcript..."):
                try:
                    entries, lang_code, is_generated = fetch_transcript_data(video_id)
                    transcript_text = build_full_transcript(entries)

                    if not transcript_text:
                        st.error("The transcript appears to be empty.")
                    else:
                        title = fetch_video_title(url)
                        if show_video:
                            st.video(url)

                        st.subheader("Title")
                        st.write(title)

                        st.subheader("Summary")
                        summary = summarize_lexrank(transcript_text, num_sentences)
                        st.write(summary)

                        with st.expander("Full transcript"):
                            st.text_area(
                                "Transcript",
                                transcript_text,
                                height=400,
                            )
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error("Unexpected error while fetching transcript.")

