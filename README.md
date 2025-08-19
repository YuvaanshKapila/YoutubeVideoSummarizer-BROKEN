# YouTube Transcript Summarizer

A clean, efficient tool that extracts YouTube video transcripts and generates intelligent summaries using AI-powered natural language processing.

## Features

- **Automatic Transcript Extraction**: Retrieves transcripts from YouTube videos automatically
- **Multiple Summarization Algorithms**: 
  - NLTK (Frequency-based with stop word removal)
  - SpaCy (Advanced linguistic analysis)
- **Multi-language Support**: Translate summaries into 12+ languages
- **Audio Generation**: Convert summaries to speech (optional)
- **Clean UI**: Professional interface without excessive emojis
- **Error Handling**: Robust error handling for various transcript scenarios

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd youtube-summarizer
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Run the setup script**
   ```bash
   python setup.py
   ```

4. **Or install manually**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown (usually `http://localhost:8501`)

3. **Enter a YouTube URL** in the input field

4. **Configure settings**:
   - Choose summarization algorithm (NLTK or SpaCy)
   - Set summary length (10-50% of original)
   - Select target language for translation
   - Enable/disable audio generation
   - Show/hide full transcript

5. **Click "Process Video"** and wait for results

## How It Works

### Transcript Extraction
The tool uses the `youtube-transcript-api` to:
- Extract video transcripts in multiple languages
- Handle videos with disabled transcripts gracefully
- Fall back to alternative language options when needed

### Summarization
Two AI algorithms are available:

**NLTK Algorithm:**
- Tokenizes text into sentences and words
- Removes stop words and short words
- Calculates word frequency scores
- Ranks sentences by importance
- Selects top-scoring sentences

**SpaCy Algorithm:**
- Uses advanced linguistic analysis
- Lemmatizes words for better understanding
- Considers part-of-speech information
- Provides more sophisticated sentence scoring

### Translation
- Uses Google Translate API via `deep-translator`
- Handles long text by chunking
- Supports 12+ languages

## Troubleshooting

### Common Issues

**"Could not retrieve transcript"**
- The video may not have transcripts enabled
- Try a different YouTube video
- Check if the video is age-restricted or private

**"Error loading spaCy model"**
- Run: `python -m spacy download en_core_web_sm`
- The tool will fall back to NLTK summarization

**"Translation failed"**
- Check your internet connection
- The target language may not be supported
- Long texts may exceed API limits

**"Audio generation failed"**
- Check your internet connection
- Some languages may not be supported by gTTS
- Text may be too long (limited to 1000 characters)

### Performance Tips

- Use shorter summary lengths for faster processing
- Disable audio generation if not needed
- The first run may be slower due to model loading

## Dependencies

- **streamlit**: Web interface framework
- **youtube-transcript-api**: YouTube transcript extraction
- **nltk**: Natural language processing toolkit
- **spacy**: Advanced NLP library
- **deep-translator**: Translation services
- **gTTS**: Text-to-speech conversion
- **beautifulsoup4**: HTML parsing
- **requests**: HTTP requests

## Limitations

- Only works with videos that have transcripts enabled
- Transcript quality depends on YouTube's automatic transcription
- Translation accuracy varies by language
- Audio generation limited to 1000 characters
- Requires internet connection for all features

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the tool.

## License

This project is open source and available under the MIT License.
