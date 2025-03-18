import streamlit as st
import requests

# Configure backend URL
BACKEND_URL = "http://localhost:8000"

st.title("News Sentiment Analyzer")
company_name = st.text_input("Enter Company Name", "Mahindra")

if st.button("Analyze News Coverage"):
    try:
        response = requests.get(
            f"{BACKEND_URL}/get-news",
            params={"company": company_name}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Build structured output
            output = {
                "Company": company_name,
                "Articles": [],
                "Comparative Sentiment Score": {
                    "Sentiment Distribution": data['analysis']['sentiment_distribution'],
                    "Coverage Differences": data['analysis']['coverage_differences'],
                    "Topic Overlap": data['analysis']['topic_overlap'],
                },
                "Final Sentiment Analysis": data['analysis']['final_sentiment'],
                "Audio": data['analysis']['hindi_summary'],  # Hindi summary filename
            }
            
            # Process articles
            for article in data['articles']:
                output["Articles"].append({
                    "Title": article['title'],
                    "Summary": article["summary"],
                    "Sentiment": article['sentiment'].capitalize(),
                    "Topics": article["topics"]
                })
                        
            # Display formatted output
            st.subheader(f"Analysis Report for {company_name}")
            st.json(output)
            
            # Display Hindi Summary Audio Player
            audio_file = data['analysis']['hindi_summary']
            if audio_file:
                st.subheader("Hindi Summary Audio")
                
                # Stream the audio file from the backend
                audio_url = f"{BACKEND_URL}/get-audio?filename={audio_file}"
                st.audio(audio_url, format="audio/mpeg")  # Correct format for MP3

        else:
            st.error(f"Backend Error: {response.text}")
            
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
