import os
import uuid
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from transformers import pipeline
from dotenv import load_dotenv
from utils import *
# Load environment variables
load_dotenv()

# Configuration constants
AUDIO_FOLDER = "audio_files"
os.makedirs(AUDIO_FOLDER, exist_ok=True)  # Ensure audio storage directory exists

# Initialize FastAPI application
app = FastAPI(
    title="News Analytics API",
    description="Advanced news analysis and sentiment detection system",
    version="1.0.0"
)

# Configure CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NLP pipelines
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# External API configuration
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GNEWS_URL = "https://gnews.io/api/v4/search"


@app.get("/get-news", response_model=Dict)
async def get_news(company: str):
    """
    Fetch and analyze news articles for specified company.
    
    Args:
        company (str): Organization name for news search
        
    Returns:
        Dict: Processed articles with analysis insights
    """
    print(f"\n[BACKEND] Processing request for: {company}")
    try:
        # Fetch news data from GNews API
        params = {
            "q": company,
            "lang": "en",
            "max": 20,
            "apikey": GNEWS_API_KEY
        }
        response = requests.get(GNEWS_URL, params=params)
        response.raise_for_status()
        data = response.json().get("articles", [])
        
        # Deduplicate articles by title
        processed_articles = []
        seen_titles = set()
        for article in data:
            title = article.get("title", "").strip()
            if title and title not in seen_titles:
                processed_articles.append({
                    "title": title,
                    "url": article.get("url"),
                    "publishedAt": article.get("publishedAt"),
                    "source": article.get("source", {}).get("name")
                })
                seen_titles.add(title)
        
        # Analyze up to 10 unique articles
        analyzed_articles = []
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        
        for idx, article in enumerate(processed_articles[:10]):
            print(f"\n[PROCESSING ARTICLE {idx+1}] {article['title']}")
            
            # Scrape article content
            content = scrape_article_content(article["url"])
            if not content:
                continue
            
            # Generate summary and extract topics
            article["summary"] = generate_summary(content)
            article["topics"] = extract_keywords(content, top_n=3) or []
            
            # Perform sentiment analysis
            result = sentiment_analyzer(content[:512])[0]
            sentiment = result['label'].upper()
            confidence = result['score']
            
            # Update sentiment counts with confidence threshold
            if confidence >= 0.7:
                sentiment_counts[sentiment] += 1
            else:
                sentiment_counts["NEUTRAL"] += 1
            
            article.update({
                "sentiment": sentiment,
                "confidence": round(confidence, 2)
            })
            analyzed_articles.append(article)
            
        # Generate Hindi audio summary
        hindi_summary = generate_hindi_summary_report(analyzed_articles)
        audio_filename = hindi_text_to_speech(hindi_summary)
        
        # Compile analysis insights
        insights = {
            "sentiment_distribution": sentiment_counts,
            "dominant_sentiment": max(sentiment_counts, key=sentiment_counts.get),
            "coverage_differences": generate_coverage_differences(analyzed_articles),
            "topic_overlap": generate_topic_overlap(analyzed_articles),
            "final_sentiment": generate_final_sentiment_summary(analyzed_articles),
            "audio_file": audio_filename,
        }
        
        return {"articles": analyzed_articles, "analysis": insights}
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}", response_class=FileResponse)
def get_audio(filename: str):
    """
    Serve generated audio files for playback
    
    Args:
        filename (str): Name of audio file to retrieve
        
    Returns:
        FileResponse: Audio file stream
    """
    file_path = os.path.join(AUDIO_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Audio file not found")


@app.post("/analyze", response_model=Dict)
async def analyze_news(
    company: str,
    search_query: Optional[str] = None,
    sentiment_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Advanced analysis endpoint with filtering capabilities
    
    Args:
        company (str): Target company name
        search_query (Optional[str]): Full-text search term
        sentiment_filter (Optional[str]): Sentiment filter (POSITIVE/NEGATIVE/NEUTRAL)
        start_date (Optional[str]): Filter start date (YYYY-MM-DD)
        end_date (Optional[str]): Filter end date (YYYY-MM-DD)
        
    Returns:
        Dict: Filtered analysis results and dashboard data
    """
    try:
        engine = NewsAnalyticsEngine(articles)
        
        # Apply search and filters
        if search_query:
            articles = engine.search_articles(search_query)
        else:
            articles = engine.filter_articles(
                sentiment=sentiment_filter,
                start_date=start_date,
                end_date=end_date
            )
            
        # Generate analysis dashboard
        dashboard = engine.generate_dashboard(articles)
        
        return {
            "metadata": {
                "total_articles": len(articles),
                "filters": {
                    "search": search_query,
                    "sentiment": sentiment_filter,
                    "date_range": f"{start_date} to {end_date}"
                }
            },
            "analysis": dashboard
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")