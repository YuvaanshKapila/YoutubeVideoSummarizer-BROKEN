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

st.set_page_config(
    page_title="AI Video Processor",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(45deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
}
.feature-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
}
.stButton > button {
    background: linear-gradient(45deg, #667eea, #764ba2);
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 25px;
    color: white;
    font-weight: bold;
    font-size: 1.1rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
}
.result-container {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords", quiet=True)

@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except:
        st.warning("Spacy model not found. Install with: python -m spacy download en_core_web_sm")
        return None

nlp = load_models()

def extract_video_id(url):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_info(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        
        title = "YouTube Video"
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.text.replace("- YouTube", "").strip()
        
        meta_title = soup.find("meta", property="og:title")
        if meta_title:
            title = meta_title.get("content", title)
            
        description = ""
        meta_desc = soup.find("meta", property="og:description")
        if meta_desc:
            description = meta_desc.get("content", "")
            
        return {"title": title, "description": description}
    except:
        return {"title": "YouTube Video", "description": ""}

def get_transcript(video_id):
    try:
        # First try to get transcript directly
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry['text'] for entry in transcript_data])
        
        # Clean the text
        full_text = re.sub(r'\[.*?\]', '', full_text)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        return full_text, transcript_data
        
    except (TranscriptsDisabled, NoTranscriptFound):
        try:
            # Try to find available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to find English transcript first
            try:
                transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
                transcript_data = transcript.fetch()
                full_text = " ".join([entry['text'] for entry in transcript_data])
                full_text = re.sub(r'\[.*?\]', '', full_text)
                full_text = re.sub(r'\s+', ' ', full_text).strip()
                return full_text, transcript_data
            except:
                # If no English transcript, try any available language
                available_transcripts = transcript_list._manually_created_transcripts
                if available_transcripts:
                    # Get the first available transcript
                    first_transcript = list(available_transcripts.values())[0]
                    transcript_data = first_transcript.fetch()
                    full_text = " ".join([entry['text'] for entry in transcript_data])
                    full_text = re.sub(r'\[.*?\]', '', full_text)
                    full_text = re.sub(r'\s+', ' ', full_text).strip()
                    return full_text, transcript_data
                else:
                    return None, None
        except Exception as e:
            st.error(f"Error accessing transcript list: {str(e)}")
            return None, None
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None, None

def clean_transcript_text(transcript_data):
    if not transcript_data:
        return ""
        
    sentences = []
    current_sentence = ""
    
    for entry in transcript_data:
        text = entry['text'].strip()
        current_sentence += text + " "
        
        if text.endswith(('.', '!', '?')) or len(current_sentence) > 100:
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return " ".join(sentences)

def sumy_summarize(text, sentence_count=5):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return " ".join([str(sentence) for sentence in summary])
    except Exception as e:
        return text[:1000] + "..."

def nltk_summarize(text, ratio=0.3):
    try:
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return text
            
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text.lower())
        
        word_freq = {}
        for word in words:
            if word.isalnum() and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if not word_freq:
            return text
            
        max_freq = max(word_freq.values())
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words_in_sentence = word_tokenize(sentence.lower())
            score = sum(word_freq.get(word, 0) for word in words_in_sentence if word in word_freq)
            if len(words_in_sentence) > 0:
                sentence_scores[i] = score / len(words_in_sentence)
        
        if not sentence_scores:
            return sentences[0] if sentences else text
            
        num_sentences = max(1, int(len(sentences) * ratio))
        top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        top_sentences.sort()
        
        return " ".join([sentences[i] for i in top_sentences])
    except Exception as e:
        return text[:1000] + "..."

def spacy_summarize(text, ratio=0.3):
    if nlp is None:
        return nltk_summarize(text, ratio)
        
    try:
        doc = nlp(text)
        sentences = list(doc.sents)
        
        if len(sentences) < 3:
            return text
        
        word_freq = {}
        for token in doc:
            if not token.is_stop and token.is_alpha and not token.is_punct:
                word_freq[token.lemma_.lower()] = word_freq.get(token.lemma_.lower(), 0) + 1
        
        if not word_freq:
            return text
            
        max_freq = max(word_freq.values())
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
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
        
        num_sentences = max(1, int(len(sentences) * ratio))
        if sentence_scores:
            top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
            top_sentences.sort()
            return " ".join([str(sentences[i]) for i in top_sentences])
        else:
            return str(sentences[0]) if sentences else text
    except Exception as e:
        return nltk_summarize(text, ratio)

@st.cache_resource
def load_t5_model():
    try:
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        return model, tokenizer
    except:
        return None, None

def t5_summarize(text):
    try:
        model, tokenizer = load_t5_model()
        if model is None or tokenizer is None:
            return text[:500] + "..."
            
        if len(text) > 3000:
            text = text[:3000]
            
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return text[:500] + "..."

def translate_text(text, target_lang):
    try:
        if target_lang == "en" or not text.strip():
            return text
        translator = GoogleTranslator(source="auto", target=target_lang)
        
        if len(text) > 4500:
            chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
            translated_chunks = [translator.translate(chunk) for chunk in chunks if chunk.strip()]
            return " ".join(translated_chunks)
        else:
            return translator.translate(text)
    except Exception as e:
        return text

def create_audio(text, lang="en"):
    try:
        if len(text) > 1000:
            text = text[:1000]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(tmp_file.name)
            
            with open(tmp_file.name, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            os.unlink(tmp_file.name)
            return audio_bytes
    except Exception as e:
        return None

st.markdown('<h1 class="main-header">AI Video Processor</h1>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown('<div class="feature-card"><h3>Extract • Transcribe • Summarize</h3><p>Advanced AI-powered video content analysis</p></div>', unsafe_allow_html=True)

video_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Summarization Settings")
    summary_type = st.radio("Method:", ["Extractive", "Abstractive (T5)"])
    
    if summary_type == "Extractive":
        algorithm = st.selectbox("Algorithm:", ["Sumy (LexRank)", "NLTK", "Spacy"])
        length_ratio = st.slider("Summary Length:", 10, 60, 30) / 100
    else:
        st.info("Using T5 transformer for abstractive summarization")

with col2:
    st.subheader("Output Settings")
    languages = {
        "English": "en", "Spanish": "es", "French": "fr", "German": "de", 
        "Italian": "it", "Portuguese": "pt", "Russian": "ru", "Japanese": "ja",
        "Korean": "ko", "Chinese": "zh", "Arabic": "ar", "Hindi": "hi"
    }
    target_language = st.selectbox("Translate to:", list(languages.keys()))
    
    enable_audio = st.checkbox("Generate Audio", value=True)
    show_transcript = st.checkbox("Show Full Transcript", value=True)

if st.button("Process Video", type="primary"):
    if not video_url:
        st.error("Please enter a YouTube URL!")
    else:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL format!")
        else:
            progress = st.progress(0)
            status = st.empty()
            
            status.text("Fetching video information...")
            progress.progress(20)
            
            video_info = get_video_info(video_url)
            
            status.text("Extracting transcript...")
            progress.progress(40)
            
            transcript_text, transcript_data = get_transcript(video_id)
            
            if not transcript_text:
                st.error("Could not retrieve transcript for this video. Transcript may be disabled or unavailable.")
                st.info("Try a different video or check if the video has captions enabled.")
            else:
                status.text("Processing with AI...")
                progress.progress(60)
                
                cleaned_text = clean_transcript_text(transcript_data) if transcript_data else transcript_text
                
                if summary_type == "Extractive":
                    if algorithm == "Sumy (LexRank)":
                        sentence_count = max(3, int(len(sent_tokenize(cleaned_text)) * length_ratio))
                        summary = sumy_summarize(cleaned_text, sentence_count)
                    elif algorithm == "NLTK":
                        summary = nltk_summarize(cleaned_text, length_ratio)
                    else:
                        summary = spacy_summarize(cleaned_text, length_ratio)
                else:
                    summary = t5_summarize(cleaned_text)
                
                status.text("Translating...")
                progress.progress(80)
                
                lang_code = languages[target_language]
                final_summary = translate_text(summary, lang_code)
                
                status.text("Finalizing...")
                progress.progress(100)
                
                st.markdown(f'<div class="result-container">', unsafe_allow_html=True)
                st.markdown(f"### {video_info['title']}")
                
                try:
                    st.video(video_url)
                except:
                    st.markdown(f"**Watch Video: {video_url}**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Length", f"{len(transcript_text)} chars")
                with col2:
                    st.metric("Summary Length", f"{len(final_summary)} chars")
                with col3:
                    compression_ratio = round((1 - len(final_summary) / len(transcript_text)) * 100, 1)
                    st.metric("Compression", f"{compression_ratio}%")
                
                st.markdown("### AI Summary")
                st.markdown(final_summary)
                st.markdown('</div>', unsafe_allow_html=True)
                
                if enable_audio:
                    st.markdown("### Audio Summary")
                    with st.spinner("Creating audio..."):
                        audio_bytes = create_audio(final_summary, lang_code)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                        else:
                            st.warning("Could not generate audio")
                
                if show_transcript:
                    with st.expander("Full Transcript"):
                        st.text_area("Complete Transcript:", transcript_text, height=400)
                
                progress.progress(0)
                status.empty()

st.markdown("---")
with st.expander("About This Tool"):
    st.markdown("""
    **AI Video Processor** uses advanced natural language processing to:
    - Extract YouTube video transcripts automatically  
    - Generate intelligent summaries using multiple AI algorithms
    - Translate content into 12+ languages
    - Create audio versions of summaries
    - Clean and format transcript text for better readability
    
    **Algorithms Available:**
    - **Sumy LexRank**: Graph-based extractive summarization
    - **NLTK**: Frequency-based sentence scoring  
    - **Spacy**: Advanced linguistic analysis with lemmatization
    - **T5 Transformer**: Neural abstractive summarization
    """)