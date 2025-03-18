import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent tokenizer warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re
from keybert import KeyBERT
import spacy
from gtts import gTTS
from googletrans import Translator
from collections import Counter
from gtts import gTTS
from pydub import AudioSegment
import os


# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Initialize KeyBERT model
kw_model = KeyBERT()

# Initialize sentiment analysis and summarization pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Get GNews API key from environment variables
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GNEWS_URL = "https://gnews.io/api/v4/search"


def clean_text(text):
    """Clean and preprocess text content"""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    # return text.strip()[:2000]  # Limit to first 2000 characters
    return text

def scrape_article_content(url):
    """Scrape main article content using BeautifulSoup"""
    print(f"[SCRAPER] Attempting to scrape: {url}")
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        selectors = ['article', '.article-content', '#article-body', '[itemprop="articleBody"]']
        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                print(f"[SCRAPER] Found content using selector: {selector}")
                return clean_text(content.get_text())
        
        # Fallback extraction
        print("[SCRAPER] Using fallback text extraction")
        paragraphs = soup.find_all('p')
        return clean_text(' '.join([p.get_text() for p in paragraphs]))
    except Exception as e:
        print(f"[SCRAPER] Error scraping {url}: {str(e)}")
        return None

def generate_summary(text):
    """Generate a summary for the given text using a summarization model"""
    try:
        summary = summarizer(text[:1024], max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"[SUMMARY] Error generating summary: {str(e)}")
        return "Summary not available."



def extract_keywords(text, top_n=3):
    """Extract single-word, important keywords using spaCy"""
    try:
        print("[TOPICS] Starting keyword extraction")
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Extract nouns and proper nouns
        keywords = [
            token.text for token in doc
            if token.pos_ in ["NOUN", "PROPN"]  # Only nouns and proper nouns
            and not token.is_stop              # Exclude stopwords
            and len(token.text) > 2            # Exclude very short words
        ]
        
        # Remove duplicates and limit to top_n
        keywords = list(set(keywords))[:top_n]
        
        # Capitalize the first letter of each keyword
        keywords = [keyword.capitalize() for keyword in keywords]
        
        print(f"[TOPICS] Extracted keywords: {keywords}")
        return keywords
    except Exception as e:
        print(f"[KEYWORD EXTRACTION] Error extracting keywords: {str(e)}")
        return []


def generate_topic_overlap(articles):
    """Identify common and unique topics across articles based on frequency instead of strict intersection."""
    
    # Flatten the list of all topics across articles
    all_topics = [topic for article in articles for topic in article.get("topics", [])]
    
    # Count topic occurrences
    topic_counts = Counter(all_topics)
    
    # Consider topics appearing at least 2 times as "common topics"
    common_topics = [topic for topic, count in topic_counts.items() if count >= 2]
    
    # Identify unique topics for each article
    unique_topics = {
        f"Unique Topics in Article {i+1}": [
            topic for topic in article.get("topics", []) if topic_counts[topic] == 1
        ] for i, article in enumerate(articles)
    }
    return {"Common Topics": common_topics, **unique_topics}

def generate_coverage_differences(articles):
    """Generate dynamic coverage differences between articles using a transformer model"""

    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for article in articles:
        sentiment_counts[article['sentiment']] += 1

    positive_articles = [(i, a) for i, a in enumerate(articles) if a['sentiment'] == "POSITIVE"]
    negative_articles = [(i, a) for i, a in enumerate(articles) if a['sentiment'] == "NEGATIVE"]

    positive_count = len(positive_articles)
    negative_count = len(negative_articles)

    # If all articles are positive or negative, return a generic message
    if positive_count == len(articles):
        return [{"Comparison": "All analyzed articles present a positive outlook, leaving little room for contrasting perspectives.", 
                 "Impact": "This may reinforce confidence in the subject but lacks a critical counterbalance."}]
    elif negative_count == len(articles):
        return [{"Comparison": "All analyzed articles take a negative stance, with no alternative viewpoints available.", 
                 "Impact": "This may lead to heightened concerns without a balanced perspective."}]

    # Determine number of comparisons
    num_comparisons = min(positive_count, negative_count)
    comparisons = []

    for i in range(num_comparisons):
        pos_idx, article_pos = positive_articles[i]
        neg_idx, article_neg = negative_articles[i]

        # Generate comparison summary
        input_text = (f"Compare these perspectives:\n"
                      f"Article {pos_idx+1}: {article_pos['summary']}\n"
                      f"Article {neg_idx+1}: {article_neg['summary']}")
        comparison_result = summarizer(input_text, max_length=50, min_length=10, do_sample=False)
        comparison_text = comparison_result[0]['summary_text']

        # Generate impact analysis
        impact_text = (f"Analyze the impact of these perspectives:\n"
                       f"Article {pos_idx+1}: {article_pos['summary']}\n"
                       f"Article {neg_idx+1}: {article_neg['summary']}")
        impact_result = summarizer(impact_text, max_length=50, min_length=10, do_sample=False)
        impact_text_generated = impact_result[0]['summary_text']

        comparisons.append({
            "Comparison": f"Article {pos_idx+1} highlights {comparison_text}, while Article {neg_idx+1} provides a different perspective.",
            "Impact": impact_text_generated
        })

    return comparisons

def generate_final_sentiment_summary(articles):
    """Summarizes the overall sentiment analysis in two lines, considering all comparisons and impacts."""
    
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    
    for article in articles:
        sentiment_counts[article['sentiment']] += 1

    # Create meaningful content for summarization
    sentiment_content = (
        f"There are {sentiment_counts['POSITIVE']} positive articles, {sentiment_counts['NEGATIVE']} negative articles, "
        f"and {sentiment_counts['NEUTRAL']} neutral articles. Positive articles highlight growth, innovation, and success, "
        f"while negative articles focus on challenges, risks, and controversies. Neutral articles provide balanced perspectives."
    )

    # Generate summary
    summary_result = summarizer(sentiment_content, max_length=50, min_length=30, do_sample=False)
    
    return summary_result[0]['summary_text']


def generate_hindi_summary_report(articles):
    """Generates a one-page Hindi summary report considering all articles, sentiment analysis, and key insights."""

    translator = Translator()
    
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    key_topics = []
    
    for article in articles:
        sentiment_counts[article['sentiment']] += 1
        key_topics.extend(article["topics"])

    unique_topics = list(set(key_topics))
    
    # Prepare input for the summarizer
    summary_input = (f"Create a one-page summary report considering these aspects:\n\n"
                     f"1. Overall sentiment analysis: {sentiment_counts}\n"
                     f"2. Key topics covered: {', '.join(unique_topics)}\n"
                     f"3. Major insights and trends from the articles\n"
                     f"4. Possible impact on public perception\n\n"
                     f"Summarize in an engaging and informative manner.")

    # Generate English summary
    english_summary = summarizer(summary_input, max_length=2000, min_length=1000, do_sample=False)[0]['summary_text']
    
    # Translate to Hindi
    hindi_summary = translator.translate(text=english_summary, src="en", dest="hi").text

    return hindi_summary

def hindi_text_to_speech(text):
    """
    Converts Hindi text to speech and saves it as an MP3 file.
    
    Parameters:
        text (str): Hindi text to be converted to speech.
    
    Returns:
        str: Path to the generated MP3 audio file.
    """
    
    # Ensure the audio_files directory exists
    os.makedirs("audio_files", exist_ok=True)
    
    # Define the output MP3 file name
    output_path = "audio_files/hindi_audio.mp3"
    
    # Generate speech from Hindi text using gTTS
    tts = gTTS(text=text, lang="hi", slow=False)
    
    # Save the MP3 file
    tts.save(output_path)

    print(f"Audio saved as: {output_path}")
    return "hindi_audio.mp3"  # Return the filename, not the full path