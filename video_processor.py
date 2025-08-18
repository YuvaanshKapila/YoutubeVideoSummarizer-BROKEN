import streamlit as st
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
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
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Video Processor",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean, professional CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem 0;
    border-bottom: 3px solid #3498db;
}

.content-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #3498db;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.summary-card {
    background: #ffffff;
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid #e9ecef;
    margin: 2rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.transcript-box {
    background: #f1f3f4;
    padding: 1rem;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    line-height: 1.4;
    max-height: 400px;
    overflow-y: auto;
}

.stButton > button {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 4px;
    font-weight: 600;
    width: 100%;
}

.stButton > button:hover {
    background-color: #2980b9;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 4px;
    border: 1px solid #dee2e6;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Download required NLTK data
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download("punkt", quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download("stopwords", quiet=True)

download_nltk_data()

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy English model not found. Please install it with: python -m spacy download en_core_web_sm")
        return None

nlp = load_spacy_model()

def extract_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
        r'(?:youtube\.com.*[\?&]v=)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_info(video_id):
    """Get video title and description from YouTube"""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Try multiple methods to get title
        title = "YouTube Video"
        
        # Method 1: og:title meta tag
        meta_title = soup.find("meta", property="og:title")
        if meta_title and meta_title.get("content"):
            title = meta_title.get("content")
        else:
            # Method 2: title tag
            title_tag = soup.find("title")
            if title_tag and title_tag.text:
                title = title_tag.text.replace("- YouTube", "").strip()
        
        # Get description
        description = ""
        meta_desc = soup.find("meta", property="og:description")
        if meta_desc and meta_desc.get("content"):
            description = meta_desc.get("content")
            
        return {"title": title, "description": description}
        
    except Exception as e:
        logger.error(f"Error fetching video info: {e}")
        return {"title": "YouTube Video", "description": ""}

def get_transcript_with_fallbacks(video_id):
    """Get transcript with multiple fallback methods"""
    try:
        # Method 1: Try to get transcript directly
        logger.info(f"Attempting to get transcript for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        full_text = " ".join([entry['text'] for entry in transcript_list])
        logger.info("Successfully retrieved transcript using direct method")
        return clean_transcript_text(full_text), transcript_list
        
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        logger.warning(f"Direct transcript failed: {e}")
        
        try:
            # Method 2: Try to find available transcripts
            logger.info("Trying to find available transcripts...")
            transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try English transcripts first
            for transcript in transcript_list_obj:
                if transcript.language_code.startswith('en'):
                    logger.info(f"Found English transcript: {transcript.language_code}")
                    transcript_data = transcript.fetch()
                    full_text = " ".join([entry['text'] for entry in transcript_data])
                    return clean_transcript_text(full_text), transcript_data
            
            # Try any available transcript
            for transcript in transcript_list_obj:
                logger.info(f"Trying transcript in language: {transcript.language_code}")
                try:
                    transcript_data = transcript.fetch()
                    full_text = " ".join([entry['text'] for entry in transcript_data])
                    return clean_transcript_text(full_text), transcript_data
                except:
                    continue
                    
        except Exception as e2:
            logger.error(f"All transcript methods failed: {e2}")
    
    except Exception as e:
        logger.error(f"Unexpected error getting transcript: {e}")
    
    return None, None

def clean_transcript_text(text):
    """Clean and format transcript text"""
    if not text:
        return ""
    
    # Remove common transcript artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove [Music], [Applause], etc.
    text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content
    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
    text = text.strip()
    
    # Split into sentences and rejoin for better formatting
    sentences = sent_tokenize(text)
    return " ".join(sentences)

def sumy_summarize(text, sentence_count=5):
    """Summarize using Sumy LexRank"""
    try:
        if not text.strip():
            return "No content to summarize."
            
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        
        # Ensure we don't ask for more sentences than available
        available_sentences = len(list(parser.document.sentences))
        sentence_count = min(sentence_count, available_sentences, 10)
        
        if sentence_count < 1:
            return text[:500] + "..."
            
        summary = summarizer(parser.document, sentence_count)
        return " ".join([str(sentence) for sentence in summary])
        
    except Exception as e:
        logger.error(f"Sumy summarization failed: {e}")
        return text[:1000] + "..."

def nltk_summarize(text, ratio=0.3):
    """Summarize using NLTK frequency analysis"""
    try:
        if not text.strip():
            return "No content to summarize."
            
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return text
            
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text.lower())
        
        # Calculate word frequencies
        word_freq = {}
        for word in words:
            if word.isalnum() and word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if not word_freq:
            return sentences[0] if sentences else text
            
        # Normalize frequencies
        max_freq = max(word_freq.values())
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words_in_sentence = word_tokenize(sentence.lower())
            score = sum(word_freq.get(word, 0) for word in words_in_sentence if word in word_freq)
            if len(words_in_sentence) > 0:
                sentence_scores[i] = score / len(words_in_sentence)
        
        if not sentence_scores:
            return sentences[0] if sentences else text
            
        # Select top sentences
        num_sentences = max(1, min(int(len(sentences) * ratio), len(sentences)))
        top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        top_sentences.sort()
        
        return " ".join([sentences[i] for i in top_sentences])
        
    except Exception as e:
        logger.error(f"NLTK summarization failed: {e}")
        return text[:1000] + "..."

def spacy_summarize(text, ratio=0.3):
    """Summarize using spaCy NLP"""
    try:
        if not nlp:
            return nltk_summarize(text, ratio)
            
        if not text.strip():
            return "No content to summarize."
            
        doc = nlp(text)
        sentences = list(doc.sents)
        
        if len(sentences) < 2:
            return text
        
        # Calculate word frequencies using lemmatization
        word_freq = {}
        for token in doc:
            if not token.is_stop and token.is_alpha and not token.is_punct and len(token.text) > 2:
                lemma = token.lemma_.lower()
                word_freq[lemma] = word_freq.get(lemma, 0) + 1
        
        if not word_freq:
            return str(sentences[0]) if sentences else text
            
        # Normalize frequencies
        max_freq = max(word_freq.values())
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            score = 0
            word_count = 0
            for token in sentence:
                if token.lemma_.lower() in word_freq:
                    score += word_freq[token.lemma_.lower()]
                    word_count += 1
            if word_count > 0:
                sentence_scores[i] = score / word_count
        
        # Select top sentences
        num_sentences = max(1, min(int(len(sentences) * ratio), len(sentences)))
        if sentence_scores:
            top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
            top_sentences.sort()
            return " ".join([str(sentences[i]) for i in top_sentences])
        else:
            return str(sentences[0]) if sentences else text
            
    except Exception as e:
        logger.error(f"spaCy summarization failed: {e}")
        return nltk_summarize(text, ratio)  # Fallback to NLTK

@st.cache_resource
def load_t5_model():
    """Load T5 model for abstractive summarization"""
    try:
        model = T5ForConditionalGeneration.from_pretrained("t5-small")  # Use smaller model for better performance
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load T5 model: {e}")
        return None, None

def t5_summarize(text, max_length=200):
    """Summarize using T5 transformer"""
    try:
        model, tokenizer = load_t5_model()
        if model is None or tokenizer is None:
            return text[:500] + "..."
            
        # Limit input length for T5
        if len(text) > 2000:
            text = text[:2000]
            
        inputs = tokenizer.encode(
            "summarize: " + text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        outputs = model.generate(
            inputs, 
            max_length=max_length, 
            min_length=30, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    except Exception as e:
        logger.error(f"T5 summarization failed: {e}")
        return text[:500] + "..."

def translate_text(text, target_lang):
    """Translate text to target language"""
    try:
        if target_lang == "en" or not text.strip():
            return text
            
        translator = GoogleTranslator(source="auto", target=target_lang)
        
        # Handle long texts by chunking
        if len(text) > 4000:
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            translated_chunks = []
            for chunk in chunks:
                if chunk.strip():
                    translated_chunks.append(translator.translate(chunk))
            return " ".join(translated_chunks)
        else:
            return translator.translate(text)
            
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text

def create_audio(text, lang="en"):
    """Create audio from text using gTTS"""
    try:
        if not text.strip():
            return None
            
        # Limit text length for audio generation
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(tmp_file.name)
            
            with open(tmp_file.name, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            os.unlink(tmp_file.name)
            return audio_bytes
            
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        return None

# Main UI
st.markdown('<h1 class="main-header">AI Video Processor</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="content-card">
<h3>Extract and Summarize YouTube Video Transcripts</h3>
<p>Enter a YouTube URL below to automatically extract the transcript and generate an AI-powered summary.</p>
</div>
""", unsafe_allow_html=True)

# Input section
video_url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

# Settings in sidebar
with st.sidebar:
    st.header("Settings")
    
    st.subheader("Summarization")
    summary_method = st.radio(
        "Method:",
        ["Extractive (Fast)", "Abstractive (T5)"],
        help="Extractive selects important sentences. Abstractive generates new text."
    )
    
    if summary_method == "Extractive (Fast)":
        algorithm = st.selectbox(
            "Algorithm:",
            ["Sumy LexRank", "NLTK Frequency", "spaCy NLP"],
            help="Different algorithms for extractive summarization"
        )
        length_ratio = st.slider("Summary Length (%):", 10, 60, 30)
    
    st.subheader("Output Options")
    languages = {
        "English": "en", "Spanish": "es", "French": "fr", "German": "de", 
        "Italian": "it", "Portuguese": "pt", "Russian": "ru", "Japanese": "ja",
        "Korean": "ko", "Chinese": "zh", "Arabic": "ar", "Hindi": "hi"
    }
    target_language = st.selectbox("Translate to:", list(languages.keys()))
    
    enable_audio = st.checkbox("Generate Audio Summary", value=False)
    show_full_transcript = st.checkbox("Show Full Transcript", value=True)

# Process button
if st.button("Process Video", type="primary"):
    if not video_url.strip():
        st.error("Please enter a YouTube URL!")
    else:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL format! Please check the URL and try again.")
        else:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Get video info
                status_text.text("Fetching video information...")
                progress_bar.progress(20)
                video_info = get_video_info(video_id)
                
                # Step 2: Get transcript
                status_text.text("Extracting transcript...")
                progress_bar.progress(40)
                transcript_text, transcript_data = get_transcript_with_fallbacks(video_id)
                
                if not transcript_text:
                    st.error("""
                    Could not retrieve transcript for this video. This could be because:
                    - The video has no transcript/captions available
                    - Transcripts are disabled by the video owner
                    - The video is private or restricted
                    - The video ID is invalid
                    
                    Please try a different video that has captions enabled.
                    """)
                else:
                    # Step 3: Generate summary
                    status_text.text("Generating summary...")
                    progress_bar.progress(60)
                    
                    if summary_method == "Extractive (Fast)":
                        if algorithm == "Sumy LexRank":
                            sentence_count = max(3, int(len(sent_tokenize(transcript_text)) * (length_ratio/100)))
                            summary = sumy_summarize(transcript_text, sentence_count)
                        elif algorithm == "NLTK Frequency":
                            summary = nltk_summarize(transcript_text, length_ratio/100)
                        else:  # spaCy NLP
                            summary = spacy_summarize(transcript_text, length_ratio/100)
                    else:  # Abstractive T5
                        summary = t5_summarize(transcript_text)
                    
                    # Step 4: Translation
                    status_text.text("Translating...")
                    progress_bar.progress(80)
                    lang_code = languages[target_language]
                    final_summary = translate_text(summary, lang_code)
                    
                    # Step 5: Complete
                    status_text.text("Complete!")
                    progress_bar.progress(100)
                    
                    # Display results
                    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
                    
                    # Video title and embed
                    st.markdown(f"### {video_info['title']}")
                    
                    try:
                        st.video(video_url)
                    except:
                        st.markdown(f"**[Watch Video]({video_url})**")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Original Length", f"{len(transcript_text):,} chars")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Summary Length", f"{len(final_summary):,} chars")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        compression = round((1 - len(final_summary) / len(transcript_text)) * 100, 1)
                        st.metric("Compression", f"{compression}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Summary
                    st.markdown("### Summary")
                    st.write(final_summary)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Audio generation
                    if enable_audio:
                        st.markdown("### Audio Summary")
                        with st.spinner("Generating audio..."):
                            audio_bytes = create_audio(final_summary, lang_code)
                            if audio_bytes:
                                st.audio(audio_bytes, format="audio/mp3")
                            else:
                                st.warning("Could not generate audio for this text.")
                    
                    # Full transcript
                    if show_full_transcript:
                        st.markdown("### Full Transcript")
                        st.markdown(f'<div class="transcript-box">{transcript_text}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                logger.error(f"Processing error: {e}")
            
            finally:
                progress_bar.empty()
                status_text.empty()

# Footer
st.markdown("---")
st.markdown("""
**About this tool:**
- Automatically extracts YouTube video transcripts
- Uses advanced NLP algorithms for summarization
- Supports multiple languages for translation
- Generates audio summaries using text-to-speech

**Supported algorithms:**
- **Sumy LexRank**: Graph-based extractive summarization
- **NLTK Frequency**: Statistical frequency analysis
- **spaCy NLP**: Advanced linguistic processing
- **T5 Transformer**: Neural abstractive summarization
""")