import os
from flask import Flask, render_template, request, jsonify
from yt_dlp import YoutubeDL
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
from transformers import pipeline

app = Flask(__name__)

# Temporary path to store downloaded files
TEMP_PATH = './temp'

# Ensure the temp directory exists
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

# Google Cloud credentials for Speech-to-Text
CREDENTIALS_PATH = r"C:\Users\User\Downloads\YoutubeAiSummarizer\groovy-gearbox-439302-f5-ac8f93205135.json"
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)

# Initialize Hugging Face summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def transcribe_audio_chunk_google(chunk_path):
    """Transcribe a single audio chunk using Google Speech-to-Text."""
    client = speech.SpeechClient(credentials=credentials)
    with open(chunk_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)
    # Combine all results into a single string
    transcript = " ".join(result.alternatives[0].transcript for result in response.results)
    return transcript

def split_audio(audio_path, chunk_length_ms=30000):
    """Split audio into smaller chunks and export as low-bitrate WAV."""
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = os.path.join(TEMP_PATH, f"chunk_{i // chunk_length_ms}.wav")
        # Export chunk with reduced bitrate and mono channel to minimize size
        chunk.export(chunk_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
        # Check file size
        if os.path.getsize(chunk_path) > 10485760:  # 10 MB
            print(f"Chunk {chunk_path} exceeds 10 MB. Reducing further.")
            os.remove(chunk_path)
        else:
            chunks.append(chunk_path)
    return chunks

def summarize_text(text, max_length=500, min_length=100):
    """Generate a summary of the given text."""
    try:
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            truncation=True
        )
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Unable to generate summary."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_video():
    if 'video_url' not in request.form:
        return jsonify({'error': 'No URL provided'})

    video_url = request.form['video_url']

    try:
        # Download audio from YouTube
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(TEMP_PATH, 'audio.%(ext)s'),
            'quiet': True,
            'noplaylist': True
        }

        with YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading audio from {video_url}")
            ydl.download([video_url])

        # Locate the downloaded audio file
        audio_files = [f for f in os.listdir(TEMP_PATH) if f.startswith('audio')]
        if not audio_files:
            return jsonify({'error': 'No audio file found. Unable to download from the given video URL.'})

        audio_file_path = os.path.join(TEMP_PATH, audio_files[0])
        print(f"Downloaded audio file: {audio_file_path}")

        # Convert audio to WAV format
        audio_wav_path = os.path.join(TEMP_PATH, 'audio.wav')
        try:
            print(f"Converting audio to WAV format: {audio_file_path}")
            audio = AudioSegment.from_file(audio_file_path)
            audio.export(audio_wav_path, format="wav")
        except Exception as e:
            return jsonify({'error': f"Failed to convert audio: {str(e)}"})

        # Split audio into chunks and transcribe
        print(f"Splitting audio into chunks: {audio_wav_path}")
        chunks = split_audio(audio_wav_path)

        full_transcript = ""
        for i, chunk_path in enumerate(chunks):
            print(f"Transcribing chunk {i + 1}/{len(chunks)}: {chunk_path}")
            try:
                full_transcript += transcribe_audio_chunk_google(chunk_path) + " "
            except Exception as e:
                print(f"Error transcribing chunk {i + 1}: {e}")

            # Cleanup chunk file
            os.remove(chunk_path)

        # Cleanup original audio files
        os.remove(audio_file_path)
        os.remove(audio_wav_path)

        # Dynamically adjust the max_length based on the input length
        transcript_length = len(full_transcript.split())
        max_length = min(500, max(171, transcript_length // 2))  # Reduce max_length if the transcript is shorter

        # Generate summary
        print("Generating summary...")
        summary = summarize_text(full_transcript, max_length=max_length)

        return jsonify({
            'summary': summary,
            'full_transcription': full_transcript[:5000],  # Optional limit for transcription preview
            'max_length': max_length  # Send max_length for display on the website
        })

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': f"Failed to process video: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
