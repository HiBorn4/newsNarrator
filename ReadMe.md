# News Sentiment Analyzer and Text-to-Speech Application

## Overview

This project is a web-based application that extracts key details from multiple news articles related to a given company, performs sentiment analysis, conducts a comparative analysis, and generates a text-to-speech (TTS) output in Hindi. The tool allows users to input a company name and receive a structured sentiment report along with an audio output.

## Features

- **News Extraction**: Extract and display the title, summary, and other relevant metadata from at least 10 unique news articles related to the given company.
- **Sentiment Analysis**: Perform sentiment analysis on the article content (positive, negative, neutral).
- **Comparative Analysis**: Conduct a comparative sentiment analysis across the articles to derive insights on how the company's news coverage varies.
- **Text-to-Speech**: Convert the summarized content into Hindi speech using an open-source TTS model.
- **Interactive Dashboard**: Provide a simple web-based interface using Streamlit or Gradio.
- **API Development**: The communication between the frontend and backend happens via APIs.
- **Deployment**: Deploy the application on Hugging Face Spaces for testing.

## Project Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/HiBorn4/newsNarrator.git
   cd news-sentiment-analyzer
   ```
2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables**:
   Create a `.env` file in the root directory and add your GNews API key:

   ```plaintext
   GNEWS_API_KEY=your_gnews_api_key_here
   ```

### Running the Application

1. **Start the backend**:

   ```bash
   uvicorn api:app --reload
   ```
2. **Start the frontend**:

   ```bash
   streamlit run app.py
   ```
3. **Access the application**:

   - Backend: `http://localhost:8000`
   - Frontend: `http://localhost:8501`

## Model Details

### Summarization Model

- **Model**: `facebook/bart-large-cnn`
- **Description**: BART (Bidirectional and Auto-Regressive Transformers) is a sequence-to-sequence model pretrained on large corpora and fine-tuned for summarization tasks. The `facebook/bart-large-cnn` model is specifically fine-tuned on the CNN/DailyMail dataset, making it ideal for news summarization.

### Sentiment Analysis Model

- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Description**: DistilBERT is a smaller, faster, and lighter version of BERT. The `distilbert-base-uncased-finetuned-sst-2-english` model is fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset, which is a binary sentiment classification task.

### Text-to-Speech (TTS) Model

- **Model**: `gTTS` (Google Text-to-Speech)
- **Description**: `gTTS` is a Python library and CLI tool to interface with Google Translate's text-to-speech API. It converts text into speech in multiple languages, including Hindi.

## API Development

### Backend API

The backend is built using FastAPI and provides the following endpoints:

- **GET /get-news**

  - **Description**: Fetches and analyzes news articles for a given company.
  - **Parameters**:
    - `company` (str): The name of the company to analyze.
  - **Response**:
    ```json
    {
      "articles": [
        {
          "title": "Article Title",
          "summary": "Article Summary",
          "sentiment": "Positive/Negative/Neutral",
          "topics": ["Topic1", "Topic2"]
        }
      ],
      "analysis": {
        "sentiment_distribution": {
          "POSITIVE": 5,
          "NEGATIVE": 3,
          "NEUTRAL": 2
        },
        "dominant_sentiment": "POSITIVE",
        "coverage_differences": [
          {
            "Comparison": "Article 1 highlights growth, while Article 2 discusses risks.",
            "Impact": "Positive news boosts confidence, while negative news raises concerns."
          }
        ],
        "topic_overlap": {
          "common_topics": ["Topic1"],
          "unique_topics": ["Topic2", "Topic3"]
        },
        "final_sentiment": "The overall sentiment is mostly positive.",
        "audio_file": "company_name_uuid.mp3"
      }
    }
    ```
- **GET /audio/{filename}**

  - **Description**: Serves the audio file for playback.
  - **Parameters**:
    - `filename` (str): The name of the audio file.
  - **Response**: Audio file in MP3 format.

### API Usage with Postman

1. **Fetch News Articles**:

   - **Method**: GET
   - **URL**: `http://localhost:8000/get-news?company=Tesla`
   - **Response**: JSON containing analyzed articles and insights.
2. **Fetch Audio File**:

   - **Method**: GET
   - **URL**: `http://localhost:8000/audio/company_name_uuid.mp3`
   - **Response**: MP3 audio file.

## Third-Party APIs

### GNews API

- **Purpose**: Fetch news articles related to a given company.
- **Integration**: Used in the backend to fetch news articles based on the company name.
- **Documentation**: [GNews API Documentation](https://gnews.io/docs/v4)

### Google Text-to-Speech (gTTS)

- **Purpose**: Convert summarized text into Hindi speech.
- **Integration**: Used in the backend to generate audio files from Hindi text.
- **Documentation**: [gTTS Documentation](https://gtts.readthedocs.io/en/latest/)

## Assumptions & Limitations

### Assumptions

1. **News Articles**: The application assumes that the fetched news articles are relevant to the given company and contain sufficient content for analysis.
2. **Sentiment Analysis**: The sentiment analysis model assumes that the content of the articles is in English.
3. **Text-to-Speech**: The TTS model assumes that the summarized content can be accurately translated into Hindi.

### Limitations

1. **News Source**: The application relies on the GNews API for fetching news articles, which may have limitations in terms of coverage and availability.
2. **Language**: The sentiment analysis model is trained on English text, so non-English content may not be accurately analyzed.
3. **Content Length**: The summarization model has a token limit, so very long articles may be truncated.
4. **Translation Accuracy**: The translation of summarized content into Hindi may not always be perfect, especially for complex sentences.

## Conclusion

This application provides a comprehensive tool for analyzing news sentiment and generating audio summaries in Hindi. By following the setup instructions and understanding the API usage, you can easily deploy and use this application for your needs. For any issues or further enhancements, feel free to contribute to the project or raise an issue on the GitHub repository.
