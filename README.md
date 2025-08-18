# AI Video Processor

A powerful Streamlit application that extracts YouTube video transcripts, generates AI-powered summaries, and provides translation and audio generation capabilities.

## Features

- **YouTube Transcript Extraction**: Automatically extracts transcripts from YouTube videos
- **Multiple Summarization Algorithms**: 
  - Sumy LexRank (graph-based)
  - NLTK (frequency-based)
  - Spacy (linguistic analysis)
  - T5 Transformer (neural abstractive)
- **Multi-language Translation**: Support for 12+ languages
- **Audio Generation**: Convert summaries to speech
- **Clean UI**: Modern, responsive interface

## Installation

1. **Clone or download the files**
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Spacy language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. **Enter a YouTube URL** in the input field
3. **Choose summarization settings:**
   - Method: Extractive or Abstractive (T5)
   - Algorithm: Sumy, NLTK, or Spacy
   - Summary length ratio
4. **Configure output settings:**
   - Target language for translation
   - Audio generation
   - Full transcript display
5. **Click "Process Video"** to start analysis

## Troubleshooting

### Common Issues

1. **Transcript not available**: Some videos have disabled captions or no transcript available
2. **Spacy model missing**: Run `python -m spacy download en_core_web_sm`
3. **T5 model loading**: The first run may take time to download the T5 model

### Video Requirements

- Video must have captions/transcripts enabled
- Public videos work best
- Some private or restricted videos may not work

## Dependencies

- **Core**: streamlit, youtube-transcript-api, beautifulsoup4
- **NLP**: nltk, spacy, transformers, sumy
- **Translation**: deep-translator
- **Audio**: gTTS
- **ML**: torch

## License

This project is open source and available under the MIT License.
