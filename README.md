# AI Video Processor

A powerful Streamlit application for extracting, summarizing, and translating YouTube video transcripts using advanced NLP techniques.

## Features

- **Automatic Transcript Extraction**: Retrieves YouTube video transcripts with multiple fallback methods
- **Multiple Summarization Algorithms**: 
  - Sumy LexRank (graph-based)
  - NLTK frequency analysis
  - spaCy NLP processing
  - T5 transformer (abstractive)
- **Multi-language Support**: Translate summaries to 12+ languages
- **Audio Generation**: Text-to-speech for summaries
- **Clean, Professional UI**: No clutter, focused on functionality

## Installation

1. Clone or download the files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Run the application:
```bash
streamlit run video_processor.py
```

2. Enter a YouTube URL
3. Configure summarization settings in the sidebar
4. Click "Process Video" to extract and summarize

## Fixed Issues

- **YouTube Transcript API**: Fixed incorrect API usage and added proper error handling
- **Multiple Fallback Methods**: Implemented robust transcript extraction with language detection
- **Clean UI**: Removed emojis and created professional interface
- **Better Error Messages**: Clear feedback when transcripts are unavailable
- **Improved Performance**: Optimized model loading and text processing

## Requirements

- Python 3.8+
- All packages listed in requirements.txt
- Internet connection for YouTube access and translation services

## Troubleshooting

If you encounter transcript extraction issues:
- Ensure the video has captions/transcripts enabled
- Try videos from different channels
- Check that the YouTube URL is valid and accessible