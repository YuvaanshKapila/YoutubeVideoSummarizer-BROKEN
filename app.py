import streamlit as st
st.set_page_config(
    page_title="YouTube Summarizer",
    page_icon='🎥',
    layout="wide",
    initial_sidebar_state="expanded",
)
import base64
import re
import os
from bs4 import BeautifulSoup
import requests
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api import (
        TranscriptsDisabled,
        VideoUnavailable,
        RequestBlocked,
        AgeRestricted,
        VideoUnplayable,
        PoTokenRequired
    )
    TRANSCRIPT_API_AVAILABLE = True
    print("[DEBUG] YouTube Transcript API v1.2.2+ imported successfully")
except ImportError as e:
    print(f"[DEBUG] Failed to import YouTube Transcript API: {e}")
    TRANSCRIPT_API_AVAILABLE = False

from urllib.parse import urlparse, parse_qs
from textwrap import dedent

from deep_translator import GoogleTranslator
from gtts import gTTS

try:
    import transformers
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"[DEBUG] Failed to import Transformers: {e}")
    TRANSFORMERS_AVAILABLE = False

import nltk
from string import punctuation
from heapq import nlargest
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    SPACY_AVAILABLE = True
except ImportError as e:
    print(f"[DEBUG] Failed to import Spacy: {e}")
    SPACY_AVAILABLE = False

import math
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords

def generate_transcript(video_id):
    if not TRANSCRIPT_API_AVAILABLE:
        raise Exception("YouTube Transcript API is not available. Please install youtube-transcript-api package.")
    
    try:
        print(f"[DEBUG] Attempting to get transcript for video ID: {video_id}")
        
        # New API method: Create an instance and then call fetch()
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id).to_raw_data()

        script = " ".join([entry["text"] for entry in transcript if entry["text"] != '[Music]'])
        print(f"[DEBUG] Successfully retrieved transcript with {len(script.split())} words")
        return script, len(script.split())

    except RequestBlocked:
        raise Exception("Request blocked by YouTube. This may be due to IP restrictions. Try using a VPN or different network.")
    except AgeRestricted:
        raise Exception("This video is age-restricted and requires authentication. Cannot access transcript.")
    except VideoUnplayable:
        raise Exception("Video is unplayable.")
    except VideoUnavailable:
        raise Exception("Video is unavailable or private. Cannot access transcript.")
    except TranscriptsDisabled:
        raise Exception("Transcripts are disabled for this video.")
    except PoTokenRequired:
        raise Exception("This video requires a PO token for transcript access. This is a YouTube restriction.")
    except Exception as e:
        raise Exception(f"No transcript available for this video. Error: {str(e)}")

def extract_video_id(url):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def nltk_summarize(text_content, percent):
    try:
        tokens = word_tokenize(text_content)
        stop_words = stopwords.words('english')
        punctuation_items = punctuation + '\n'

        word_frequencies = {}
        for word in tokens:
            if word.lower() not in stop_words:
                if word.lower() not in punctuation_items:
                    if word not in word_frequencies.keys():
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1
        
        if not word_frequencies:
            return "Unable to generate summary from this content."
        
        max_frequency = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency
        sentence_token = sent_tokenize(text_content)
        sentence_scores = {}
        for sent in sentence_token:
            sentence = sent.split(" ")
            for word in sentence:
                if word.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.lower()]

        select_length = int(len(sentence_token) * (int(percent) / 100))
        if select_length == 0:
            select_length = 1
        summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
        final_summary = [word for word in summary]
        summary = ' '.join(final_summary)
        return summary
    except Exception as e:
        return f"Error in NLTK summarization: {str(e)}"

def spacy_summarize(text_content, percent):
    if not SPACY_AVAILABLE:
        return nltk_summarize(text_content, percent)
        
    try:
        try:
            import en_core_web_sm
            nlp = en_core_web_sm.load()
        except:
            st.error("spaCy English model not available. Using NLTK instead.")
            return nltk_summarize(text_content, percent)
        
        stop_words = list(STOP_WORDS)
        punctuation_items = punctuation + '\n'

        nlp_object = nlp(text_content)
        word_frequencies = {}
        for word in nlp_object:
            if word.text.lower() not in stop_words:
                if word.text.lower() not in punctuation_items:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1
        
        if not word_frequencies:
            return "Unable to generate summary from this content."
        
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency
        sentence_token = [sentence for sentence in nlp_object.sents]
        sentence_scores = {}
        for sent in sentence_token:
            sentence = sent.text.split(" ")
            for word in sentence:
                if word.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.lower()]

        select_length = int(len(sentence_token) * (int(percent) / 100))
        if select_length == 0:
            select_length = 1
        summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
        final_summary = [word.text for word in summary]
        summary = ' '.join(final_summary)
        return summary
    except Exception as e:
        return f"Error in spaCy summarization: {str(e)}"

def tfidf_summarize(text_content):
    try:
        sentences = sent_tokenize(text_content)
        if not sentences:
            return "Unable to process transcript for summarization."
        total_documents = len(sentences)
        freq_matrix = _create_frequency_matrix(sentences)
        tf_matrix = _create_tf_matrix(freq_matrix)
        count_doc_per_words = _create_documents_per_words(freq_matrix)
        idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
        tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
        sentence_scores = _score_sentences(tf_idf_matrix)
        threshold = _find_average_score(sentence_scores)
        summary = _generate_summary(sentences, sentence_scores, 1.0 * threshold)
        return summary
    except Exception as e:
        return f"Error in TF-IDF summarization: {str(e)}"

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''
    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1
    return summary if summary else "Unable to generate summary with current threshold."

def _find_average_score(sentenceValue):
    if not sentenceValue:
        return 0
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    average = (sumValues / len(sentenceValue))
    return average

def _score_sentences(tf_idf_matrix):
    sentenceValue = {}
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        if count_words_in_sentence == 0:
            continue
        for word, score in f_table.items():
            total_score_per_sentence += score
        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
    return sentenceValue

def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sent1] = tf_idf_table
    return tf_idf_matrix

def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}
    for sent, f_table in freq_matrix.items():
        idf_table = {}
        for word in f_table.keys():
            if count_doc_per_words[word] > 0:
                idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))
            else:
                idf_table[word] = 0
        idf_matrix[sent] = idf_table
    return idf_matrix

def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}
    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1
    return word_per_doc_table

def _create_tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, f_table in freq_matrix.items():
        tf_table = {}
        count_words_in_sentence = len(f_table)
        if count_words_in_sentence == 0:
            continue
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence
        tf_matrix[sent] = tf_table
    return tf_matrix

def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1
        frequency_matrix[sent[:15]] = freq_table
    return frequency_matrix

def get_key_from_dict(val, dic):
    key_list = list(dic.keys())
    val_list = list(dic.values())
    if val in val_list:
        ind = val_list.index(val)
        return key_list[ind]
    return 'en'

# Hide Streamlit style
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
header[data-testid="stHeader"] {display:none;}
.block-container {padding-top: 1rem;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Main header
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); padding: 2.5rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 3rem; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        YouTube Summarizer Pro
    </h1>
    <p style="color: rgba(255,255,255,0.95); text-align: center; margin: 1rem 0 0 0; font-size: 1.3rem; font-weight: 300;">
        Transform Video Content into Intelligent Insights
    </p>
    <div style="text-align: center; margin-top: 1.5rem;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: white; font-size: 0.9rem;">
            Powered by Advanced AI - Multi-Language Support - Audio Generation
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar header
st.sidebar.markdown("""
<div style="background: linear-gradient(145deg, #f8f9fa, #e9ecef); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #dee2e6;">
    <h3 style="margin: 0; color: #2c3e50; font-weight: 600;">Control Panel</h3>
    <p style="margin: 0.5rem 0 0 0; color: #6c757d; font-size: 0.95rem;">Configure your summarization preferences</p>
</div>
""", unsafe_allow_html=True)

# URL input
url = st.sidebar.text_input('Video URL', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', help="Paste any YouTube video URL here")

# Video title and player
try:
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, 'html.parser')
    link = soup.find_all(name="title")[0]
    title = str(link)
    title = title.replace("<title>","")
    title = title.replace("</title>","")
    title = title.replace("&amp;","&")
    
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, #ffffff, #f8f9fa); padding: 2rem; border-radius: 15px; border: 1px solid #e9ecef; margin-bottom: 1.5rem; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
        <h3 style="margin: 0 0 1.5rem 0; color: #2c3e50; font-weight: 600; font-size: 1.4rem;">{title}</h3>
    </div>
    """, unsafe_allow_html=True)
    st.video(url)
except Exception as e:
    st.error(f"Could not load video or title. Please check the URL format. Error: {e}")

# Summarization type selection
sumtype = st.sidebar.selectbox(
    'Summarization Type',
    options=['Extractive', 'Abstractive (T5 Algorithm)'],
    help="Choose between extractive (key sentence selection) or abstractive (AI-generated) summarization"
)

# Languages dictionary
languages_dict = {'en':'English' ,'af':'Afrikaans' ,'sq':'Albanian' ,'am':'Amharic' ,'ar':'Arabic' ,'hy':'Armenian' ,'az':'Azerbaijani' ,'eu':'Basque' ,'be':'Belarusian' ,'bn':'Bengali' ,'bs':'Bosnian' ,'bg':'Bulgarian' ,'ca':'Catalan' ,'ceb':'Cebuano' ,'ny':'Chichewa' ,'zh-cn':'Chinese (simplified)' ,'zh-tw':'Chinese (traditional)' ,'co':'Corsican' ,'hr':'Croatian' ,'cs':'Czech' ,'da':'Danish' ,'nl':'Dutch' ,'eo':'Esperanto' ,'et':'Estonian' ,'tl':'Filipino' ,'fi':'Finnish' ,'fr':'French' ,'fy':'Frisian' ,'gl':'Galician' ,'ka':'Georgian' ,'de':'German' ,'el':'Greek' ,'gu':'Gujarati' ,'ht':'Haitian creole' ,'ha':'Hausa' ,'haw':'Hawaiian' ,'he':'Hebrew' ,'hi':'Hindi' ,'hmn':'Hmong' ,'hu':'Hungarian' ,'is':'Icelandic' ,'ig':'Igbo' ,'id':'Indonesian' ,'ga':'Irish' ,'it':'Italian' ,'ja':'Japanese' ,'jw':'Javanese' ,'kn':'Kannada' ,'kk':'Kazakh' ,'km':'Khmer' ,'ko':'Korean' ,'ku':'Kurdish (kurmanji)' ,'ky':'Kyrgyz' ,'lo':'Lao' ,'la':'Latin' ,'lv':'Latvian' ,'lt':'Lithuanian' ,'lb':'Luxembourgish' ,'mk':'Macedonian' ,'mg':'Malagasy' ,'ms':'Malay' ,'ml':'Malayalam' ,'mt':'Maltese' ,'mi':'Maori' ,'mr':'Marathi' ,'mn':'Mongolian' ,'my':'Myanmar (burmese)' ,'ne':'Nepali' ,'no':'Norwegian' ,'or':'Odia' ,'ps':'Pashto' ,'fa':'Persian' ,'pl':'Polish' ,'pt':'Portuguese' ,'pa':'Punjabi' ,'ro':'Romanian' ,'ru':'Russian' ,'sm':'Samoan' ,'gd':'Scots gaelic' ,'sr':'Serbian' ,'st':'Sesotho' ,'sn':'Shona' ,'sd':'Sindhi' ,'si':'Sinhala' ,'sk':'Slovak' ,'sl':'Slovenian' ,'so':'Somali' ,'es':'Spanish' ,'su':'Sundanese' ,'sw':'Swahili' ,'sv':'Swedish' ,'tg':'Tajik' ,'ta':'Tamil' ,'te':'Telugu' ,'th':'Thai' ,'tr':'Turkish' ,'uk':'Ukrainian' ,'ur':'Urdu' ,'ug':'Uyghur' ,'uz':'Uzbek' ,'vi':'Vietnamese' ,'cy':'Welsh' ,'xh':'Xhosa' ,'yi':'Yiddish' ,'yo':'Yoruba' ,'zu':'Zulu'}

# Language selection
add_selectbox = st.sidebar.selectbox(
    "Output Language",
    list(languages_dict.values()),
    help="Choose the language for the summary output"
)

summ = "" # Initialize summ variable
translated = "" # Initialize translated variable

if sumtype == 'Extractive':
    # Algorithm selection for extractive
    sumalgo = st.sidebar.selectbox(
        'Algorithm',
        options=['NLTK', 'Spacy', 'TF-IDF'],
        help="Choose the algorithm for extractive summarization"
    )

    # Length selection
    length = st.sidebar.select_slider(
        'Summary Length',
        options=['10%', '20%', '30%', '40%', '50%'],
        value='30%',
        help="Select what percentage of the original content to include in summary"
    )
    if st.sidebar.button('Generate Summary', type='primary', use_container_width=True, key='generate_extractive_summary'):
        try:
            with st.spinner('Analyzing video and generating summary...'):
                st.markdown("""
                <div style="background: linear-gradient(90deg, #d4edda, #c3e6cb); border: 1px solid #c3e6cb; color: #155724; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; display: flex; align-items: center;">
                        Summary Generated Successfully
                    </h4>
                </div>
                """, unsafe_allow_html=True)

                video_id = extract_video_id(url)
                if not video_id:
                    st.error("Could not extract video ID from URL. Please check the URL format.")
                    st.stop()

                transcript, no_of_words = generate_transcript(video_id)
                st.info(f"Original transcript: {no_of_words:,} words")

                if sumalgo == 'NLTK':
                    summ = nltk_summarize(transcript, int(length[:2]))
                elif sumalgo == 'Spacy':
                    summ = spacy_summarize(transcript, int(length[:2]))
                elif sumalgo == 'TF-IDF':
                    summ = tfidf_summarize(transcript)

                try:
                    translated = GoogleTranslator(source='auto', target=get_key_from_dict(add_selectbox, languages_dict)).translate(summ)
                except Exception as e:
                    translated = summ
                    st.warning(f"Translation failed, showing original text: {str(e)}")
                
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #f8f9fa, #ffffff); border-left: 5px solid #667eea; padding: 2rem; border-radius: 12px; margin: 1.5rem 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0 0 1.5rem 0; color: #2c3e50; display: flex; align-items: center;">
                        Summary ({sumalgo} Algorithm)
                    </h4>
                    <p style="line-height: 1.8; color: #34495e; margin: 0; font-size: 1.1rem; text-align: justify;">{translated}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style="background: linear-gradient(90deg, #e8f4fd, #bee5eb); border: 1px solid #bee5eb; color: #0c5460; padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; display: flex; align-items: center;">
                        Audio Summary
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                
                no_support = ['Amharic', 'Azerbaijani', 'Basque', 'Belarusian', 'Cebuano', 'Chichewa', 'Chinese (simplified)', 'Chinese (traditional)', 'Corsican', 'Frisian', 'Galician', 'Georgian', 'Haitian creole', 'Hausa', 'Hawaiian', 'Hmong', 'Igbo', 'Irish', 'Kazakh', 'Kurdish (kurmanji)', 'Kyrgyz', 'Lao', 'Lithuanian', 'Luxembourgish', 'Malagasy', 'Maltese', 'Maori', 'Mongolian', 'Odia', 'Pashto', 'Persian', 'Punjabi', 'Samoan', 'Scots gaelic', 'Sesotho', 'Shona', 'Sindhi', 'Slovenian', 'Somali', 'Tajik', 'Uyghur', 'Uzbek', 'Xhosa', 'Yiddish', 'Yoruba', 'Zulu']
                
                if add_selectbox in no_support:
                    st.warning("Audio support for this language is currently unavailable")
                else:
                    try:
                        speech = gTTS(text=translated, lang=get_key_from_dict(add_selectbox, languages_dict), slow=False)
                        speech.save('user_trans.mp3')
                        with open('user_trans.mp3', 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/mp3', start_time=0)
                    except Exception as audio_error:
                        st.warning(f"Could not generate audio: {audio_error}")
                
        except Exception as e:
            st.error(f"An error occurred during AI summarization: {str(e)}")

elif sumtype == 'Abstractive (T5 Algorithm)':
    if st.sidebar.button('Generate AI Summary', type='primary', use_container_width=True, key='generate_abstractive_summary'):
        if not TRANSFORMERS_AVAILABLE:
            st.error("The 'transformers' library is not installed. Abstractive summarization is unavailable.")
            st.stop()
            
        try:
            with st.spinner('Loading T5 model and generating AI summary...'):
                st.markdown("""
                <div style="background: linear-gradient(90deg, #d4edda, #c3e6cb); border: 1px solid #c3e6cb;
                                    color: #155724; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                                    box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; display: flex; align-items: center;">
                        AI Summary Generated Successfully
                    </h4>
                </div>
                """, unsafe_allow_html=True)

                video_id = extract_video_id(url)
                if not video_id:
                    st.error("Could not extract video ID from URL. Please check the URL format.")
                    st.stop()

                transcript, no_of_words = generate_transcript(video_id)
                st.info(f"Original transcript: {no_of_words:,} words")

                # Load T5 model
                model = T5ForConditionalGeneration.from_pretrained("t5-base")
                tokenizer = T5Tokenizer.from_pretrained("t5-base")
                inputs = tokenizer.encode("summarize: " + transcript, return_tensors="pt",
                                            max_length=512, truncation=True)

                outputs = model.generate(
                    inputs,
                    max_length=150,
                    min_length=40,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
                summ = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Translate if needed
                try:
                    translated = GoogleTranslator(
                        source='auto',
                        target=get_key_from_dict(add_selectbox, languages_dict)
                    ).translate(summ)
                except Exception as e:
                    translated = summ
                    st.warning(f"Translation failed, showing original text: {str(e)}")

                # Show summary
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #f8f9fa, #ffffff);
                                    border-left: 5px solid #764ba2; padding: 2rem; border-radius: 12px;
                                    margin: 1.5rem 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0 0 1.5rem 0; color: #2c3e50; display: flex; align-items: center;">
                        AI-Generated Summary (T5 Algorithm)
                    </h4>
                    <p style="line-height: 1.8; color: #34495e; margin: 0; font-size: 1.1rem; text-align: justify;">
                        {translated}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Audio generation
                st.markdown("""
                <div style="background: linear-gradient(90deg, #e8f4fd, #bee5eb);
                                    border: 1px solid #bee5eb; color: #0c5460; padding: 1.5rem;
                                    border-radius: 12px; margin: 1.5rem 0;
                                    box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; display: flex; align-items: center;">
                        Audio Summary
                    </h4>
                </div>
                """, unsafe_allow_html=True)

                no_support = ['Amharic', 'Azerbaijani', 'Basque', 'Belarusian', 'Cebuano', 'Chichewa', 'Chinese (simplified)', 'Chinese (traditional)', 'Corsican', 'Frisian', 'Galician', 'Georgian', 'Haitian creole', 'Hausa', 'Hawaiian', 'Hmong', 'Igbo', 'Irish', 'Kazakh', 'Kurdish (kurmanji)', 'Kyrgyz', 'Lao', 'Lithuanian', 'Luxembourgish', 'Malagasy', 'Maltese', 'Maori', 'Mongolian', 'Odia', 'Pashto', 'Persian', 'Punjabi', 'Samoan', 'Scots gaelic', 'Sesotho', 'Shona', 'Sindhi', 'Slovenian', 'Somali', 'Tajik', 'Uyghur', 'Uzbek', 'Xhosa', 'Yiddish', 'Yoruba', 'Zulu']
                if add_selectbox in no_support:
                    st.warning("Audio support for this language is currently unavailable")
                else:
                    try:
                        speech = gTTS(
                            text=translated,
                            lang=get_key_from_dict(add_selectbox, languages_dict),
                            slow=False
                        )
                        speech.save('user_trans.mp3')
                        with open('user_trans.mp3', 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/mp3', start_time=0)
                    except Exception as audio_error:
                        st.warning(f"Could not generate audio: {audio_error}")
        except Exception as e:
            st.error(f"An error occurred during AI summarization: {str(e)}")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: linear-gradient(145deg, #f8f9fa, #e9ecef);
             padding: 1rem; border-radius: 10px; text-align: center;">
    <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
        <strong>YouTube Summarizer Pro v2.0</strong><br>
        Multi-Algorithm AI Processing<br>
        100+ Languages - Audio Generation
    </p>
</div>
""", unsafe_allow_html=True)

# Debug information (startup logs)
print("YouTube Summarizer Pro v2.0 is running!")
print("Features: Advanced error handling, modern UI, multi-algorithm summarization")
print("API Version: YouTube Transcript API v1.2.2+")
print("Ready to transform videos into intelligent insights!")