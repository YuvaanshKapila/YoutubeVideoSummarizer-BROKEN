# YouTube AI Summarizer

A Flask-based web application that generates concise summaries from YouTube videos using AI-driven natural language processing. Powered by OpenAI and deployed with Firebase for secure authentication and hosting.

## Features

- Extracts transcripts from YouTube videos
- Generates smart summaries using AI
- Interactive web interface built with Flask
- Firebase authentication and hosting integration

## Project Structure

├── app.py # Main Flask application ├── templates/ # HTML templates ├── static/ # Static assets (CSS, JS) ├── requirements.txt # Python dependencies ├── groovy-gearbox-*.json # Firebase service account key ├── path_to_your_service_account_key.json # Placeholder for secret key ├── pycache/ # Python cache files


## Setup Instructions

### Prerequisites

- Python 3.11+ (recommended)
- A Firebase project (for hosting and authentication)
- OpenAI API key

### Installation

1. Clone the repository:
git clone https://github.com/YuvaanshKapila/YoutubeVideoSummarizer.git cd YoutubeVideoSummarizer


2. Set up a virtual environment (optional but recommended):
python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate


3. Install dependencies:
pip install -r requirements.txt


4. Add your service account key:
Replace `path_to_your_service_account_key.json` with your actual Firebase key file.

5. Set your API keys and configurations as environment variables or a `.env` file.

### Running Locally

Start the Flask app with:

python app.py


Access it at `http://localhost:5000`.

## Deployment

You can deploy on platforms like Render or Firebase Hosting. Be sure to set Python version compatibility and keep credentials secure.

## License

This project is licensed under the MIT
