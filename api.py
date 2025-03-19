import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from transformers import pipeline
from utils import *
import uvicorn

# Constants
AUDIO_FOLDER = "audio_files"
os.makedirs(AUDIO_FOLDER, exist_ok=True)  # Create audio folder if it doesn't exist

# Initialize FastAPI app
app = FastAPI()

# Configure CORS (if frontend is hosted separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Get GNews API key from environment variables
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GNEWS_URL = "https://gnews.io/api/v4/search"


@app.get("/get-news")
async def get_news(company: str):
    """
    Fetch and analyze news articles for a given company.
    """
    print(f"\n[BACKEND] Received request for company: {company}")
    try:
        # Fetch news from GNews API
        params = {"q": company, "lang": "en", "max": 20, "apikey": GNEWS_API_KEY}
        print("[BACKEND] Making GNews API call")
        response = requests.get(GNEWS_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get("articles", [])
        print(f"[BACKEND] Found {len(articles)} raw articles")
        
        # Deduplicate articles
        unique_articles = []
        seen_titles = set()
        for article in articles:
            title = article.get("title", "").strip()
            if title and title not in seen_titles:
                unique_articles.append({
                    "title": title,
                    "url": article.get("url"),
                    "publishedAt": article.get("publishedAt"),
                    "source": article.get("source", {}).get("name")
                })
                seen_titles.add(title)
        
        print(f"[BACKEND] Processing {len(unique_articles[:10])} articles")
        
        analyzed_articles = []
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        
        # Analyze each article
        for idx, article in enumerate(unique_articles[:10]):
            print(f"\n[ARTICLE {idx+1}] Processing: {article['title']}")
            content = scrape_article_content(article["url"])
            if not content:
                print(f"[ARTICLE {idx+1}] Failed to scrape content")
                continue
            
            # Generate summary
            summary = generate_summary(content)
            article["summary"] = summary

            # Extract major keywords
            print(f"[ARTICLE {idx+1}] Extracting topics...")
            article["topics"] = extract_keywords(content, top_n=3) or []
            print(f"[ARTICLE {idx+1}] Topics attached: {article['topics']}")
            
            # Perform sentiment analysis
            print(f"[ARTICLE {idx+1}] Analyzing sentiment...")
            try:
                result = sentiment_analyzer(content[:512])[0]
                sentiment = result['label'].upper()
                score = result['score']
                print(f"[ARTICLE {idx+1}] Sentiment: {sentiment} ({score:.2f})")
                
                if score >= 0.7:
                    sentiment_counts[sentiment] += 1
                else:
                    sentiment_counts["NEUTRAL"] += 1
                
                article["sentiment"] = sentiment
                article["confidence"] = score
            except Exception as e:
                print(f"[ARTICLE {idx+1}] Sentiment analysis failed: {str(e)}")
                article["sentiment"] = "NEUTRAL"
                article["confidence"] = 0.0
            
            analyzed_articles.append(article)
            
        # Generate Hindi summary and audio
        hindi_speech = generate_hindi_summary_report(analyzed_articles)
        audio_filename = f"{company}_{uuid.uuid4()}.mp3"
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        hindi_text_to_speech(hindi_speech, audio_path)
        
        # Prepare insights
        insights = {
            "sentiment_distribution": sentiment_counts,
            "dominant_sentiment": max(sentiment_counts, key=sentiment_counts.get),
            "coverage_differences": generate_coverage_differences(analyzed_articles),
            "topic_overlap": generate_topic_overlap(analyzed_articles),
            "final_sentiment": generate_final_sentiment_summary(analyzed_articles),
            "audio_file": audio_filename  # Return the audio filename
        }
        
        return {"articles": analyzed_articles, "analysis": insights}
        
    except Exception as e:
        print(f"[BACKEND] Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}")
def get_audio(filename: str):
    """
    Serve the audio file for playback.
    """
    file_path = os.path.join(AUDIO_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mp3")
    raise HTTPException(status_code=404, detail="Audio file not found")


@app.get("/")
def read_root():
    return {"message": "API is running!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)