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
from pydub import AudioSegment
from datetime import datetime
import yfinance as yf
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.metrics import pairwise_distances
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Dict, Optional

# Load NLP models
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# External API configuration
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GNEWS_URL = "https://gnews.io/api/v4/search"

def clean_text(text: str) -> str:
    """Normalize and clean text content"""
    return re.sub(r'\s+', ' ', text).strip()

def scrape_article_content(url: str) -> Optional[str]:
    """Extract main article content using multiple strategies"""
    print(f"[SCRAPER] Processing: {url}")
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try known content selectors
        for selector in ['article', '.article-content', '#article-body', '[itemprop="articleBody"]']:
            if content := soup.select_one(selector):
                print(f"[SCRAPER] Matched selector: {selector}")
                return clean_text(content.get_text())
                
        # Fallback to paragraph extraction
        print("[SCRAPER] Using paragraph fallback")
        return clean_text(' '.join(p.get_text() for p in soup.find_all('p')))
    except Exception as e:
        print(f"[SCRAPER] Error: {str(e)}")
        return None

def generate_summary(text: str) -> str:
    """Create abstractive summary using BART model"""
    try:
        return summarizer(text[:1024], max_length=150, min_length=50)[0]['summary_text']
    except Exception as e:
        print(f"[SUMMARY] Error: {str(e)}")
        return "Summary unavailable"

def extract_keywords(text: str, top_n: int = 3) -> List[str]:
    """Extract key terms using spaCy linguistic features"""
    try:
        doc = nlp(text)
        keywords = list({
            token.text.capitalize() for token in doc 
            if token.pos_ in ("NOUN", "PROPN") 
            and not token.is_stop 
            and len(token) > 2
        })[:top_n]
        print(f"[KEYWORDS] Extracted: {keywords}")
        return keywords
    except Exception as e:
        print(f"[KEYWORDS] Error: {str(e)}")
        return []

def generate_topic_overlap(articles: List[Dict]) -> Dict:
    """Analyze topic distribution across articles"""
    all_topics = [topic for art in articles for topic in art.get("topics", [])]
    topic_counts = Counter(all_topics)
    
    return {
        "common_topics": [t for t, c in topic_counts.items() if c >= 2],
        **{f"unique_topics_{i+1}": [t for t in art['topics'] if topic_counts[t] == 1] 
           for i, art in enumerate(articles)}
    }

def generate_coverage_differences(articles: List[Dict]) -> List[Dict]:
    """Compare contrasting viewpoints between articles"""
    sentiments = Counter(art['sentiment'] for art in articles)
    positive = [a for a in articles if a['sentiment'] == 'POSITIVE']
    negative = [a for a in articles if a['sentiment'] == 'NEGATIVE']
    
    comparisons = []
    for pos, neg in zip(positive[:len(negative)], negative[:len(positive)]):
        try:
            comparison = summarizer(
                f"Compare: {pos['summary']} vs {neg['summary']}",
                max_length=40
            )[0]['summary_text']
            impact = summarizer(
                f"Analyze impact: {pos['summary']} vs {neg['summary']}",
                max_length=40
            )[0]['summary_text']
            comparisons.append({
                "comparison": comparison,
                "impact": impact
            })
        except Exception as e:
            print(f"[COMPARISON] Error: {str(e)}")
    
    return comparisons

def generate_final_sentiment_summary(articles: List[Dict]) -> str:
    """Create consolidated sentiment overview"""
    sentiment_counts = Counter(art['sentiment'] for art in articles)
    summary_input = (
        f"Sentiment distribution: {sentiment_counts}. "
        f"Key articles: {[a['title'] for a in articles]}. "
        "Provide a concise market position summary."
    )
    return summarizer(summary_input, max_length=100)[0]['summary_text']

def generate_hindi_summary_report(articles: List[Dict]) -> str:
    """Create translated summary report in Hindi"""
    translator = Translator()
    report_content = " ".join(a['summary'] for a in articles)
    english_summary = summarizer(report_content, max_length=500)[0]['summary_text']
    return translator.translate(english_summary, dest='hi').text

def hindi_text_to_speech(text: str) -> str:
    """Convert Hindi text to audio file"""
    os.makedirs("audio_files", exist_ok=True)
    tts = gTTS(text=text, lang='hi')
    filename = "hindi_audio.mp3"
    tts.save(f"audio_files/{filename}")
    return filename

def generate_advanced_analysis(articles: List[Dict], company: str) -> Dict:
    """Perform multi-dimensional news analytics"""
    # Sentiment trend analysis
    sentiment_trends = [
        {
            "date": datetime.strptime(a['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").date(),
            "sentiment_score": 1 if a['sentiment'] == 'POSITIVE' else -1
        } for a in articles
    ]
    
    # Topic modeling
    topic_model = BERTopic().fit([a['content'][:1000] for a in articles])
    
    return {
        "sentiment_trends": sentiment_trends,
        "topics": topic_model.get_topics(),
        "stock_correlation": analyze_stock_correlation(company, sentiment_trends)
    }

def analyze_stock_correlation(company: str, sentiment: List[Dict]) -> Optional[float]:
    """Calculate news sentiment vs stock price correlation"""
    try:
        stock_data = yf.Ticker(company).history(period="1mo")
        merged = pd.merge(
            pd.DataFrame(sentiment),
            stock_data[['Close']],
            left_on='date',
            right_index=True
        )
        return np.corrcoef(merged['sentiment_score'], merged['Close'])[0,1]
    except Exception as e:
        print(f"[STOCK_CORR] Error: {str(e)}")
        return None

class NewsAnalyticsEngine:
    def __init__(self, articles: List[Dict]):
        self.articles = self._preprocess(articles)
    
    def _preprocess(self, articles: List[Dict]) -> List[Dict]:
        """Normalize and validate article data"""
        processed = []
        for art in articles:
            try:
                processed.append({
                    'title': art['title'].lower().strip(),
                    'content': clean_text(art['content']),
                    'sentiment': art['sentiment'].upper(),
                    'date': datetime.strptime(art['publishedAt'], "%Y-%m-%dT%H:%M:%SZ"),
                    'topics': [t.lower() for t in art.get('topics', [])]
                })
            except KeyError as e:
                raise HTTPException(400, f"Missing field: {e}")
        return processed
    
    def search(self, query: str) -> List[Dict]:
        """Full-text search across articles"""
        query = query.lower()
        return [a for a in self.articles 
                if query in a['title'] or query in a['content']]
    
    def filter(self, 
               sentiment: Optional[str] = None,
               start_date: Optional[str] = None,
               end_date: Optional[str] = None) -> List[Dict]:
        """Filter articles by criteria"""
        filtered = self.articles
        if sentiment:
            filtered = [a for a in filtered if a['sentiment'] == sentiment.upper()]
        if start_date:
            filtered = [a for a in filtered if a['date'] >= datetime.fromisoformat(start_date)]
        if end_date:
            filtered = [a for a in filtered if a['date'] <= datetime.fromisoformat(end_date)]
        return filtered
    
    def generate_dashboard(self) -> Dict:
        """Create analytics dashboard"""
        return {
            "sentiment_distribution": Counter(a['sentiment'] for a in self.articles),
            "temporal_trends": self._get_temporal_trends(),
            "wordcloud": self._generate_wordcloud(),
            "topic_analysis": self._get_topic_analysis()
        }
    
    def _get_temporal_trends(self) -> Dict:
        """Analyze sentiment over time"""
        df = pd.DataFrame([{
            'date': a['date'].date(),
            'sentiment': a['sentiment']
        } for a in self.articles])
        return df.groupby(['date', 'sentiment']).size().unstack(fill_value=0).to_dict()
    
    def _generate_wordcloud(self) -> str:
        """Create base64 encoded word cloud"""
        wc = WordCloud().generate(' '.join(a['content'] for a in self.articles))
        img = io.BytesIO()
        wc.to_image().save(img, format='PNG')
        return base64.b64encode(img.getvalue()).decode('utf-8')
    
    def _get_topic_analysis(self) -> List[str]:
        """Extract common topics"""
        return Counter(t for a in self.articles for t in a['topics']).most_common(10)