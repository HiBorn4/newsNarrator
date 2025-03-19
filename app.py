import base64
import streamlit as st
import requests
import pandas as pd

# Application Configuration
BACKEND_URL = "http://localhost:8000"  # Backend API endpoint
DEFAULT_COMPANY = "Mahindra"  # Default company for analysis

# Streamlit App Configuration
st.title("Multilingual News Sentiment Analyzer")
st.markdown("#### Real-time news analysis with sentiment detection and multilingual summaries")

# User Input Section
company_name = st.text_input(
    "Enter Organization Name 🏢", 
    DEFAULT_COMPANY,
    help="Company name for news analysis (e.g., 'Tesla', 'Reliance')"
)

# Analysis Execution
if st.button("🚀 Begin Analysis"):
    try:
        # API request to backend
        with st.spinner("Fetching and analyzing news articles..."):
            response = requests.get(
                f"{BACKEND_URL}/get-news",
                params={"company": company_name}
            )
        
        # Response Handling
        if response.status_code == 200:
            data = response.json()
            
            # Construct analysis report
            analysis_report = {
                "organization": company_name,
                "Articles": [],
                "news_analysis": {
                    "sentiment_distribution": data['analysis']['sentiment_distribution'],
                    "coverage_differences": data['analysis']['coverage_differences'],
                    "topic_overlap": data['analysis']['topic_overlap']
                },
                "executive_summary": data['analysis']['final_sentiment'],
                "hindi_audio_summary": data['analysis']['audio_file']
            }

            # Process articles
            for article in data['articles']:
                analysis_report["Articles"].append({
                "Title": article['title'],
                "Summary": article["summary"],
                "Sentiment": article['sentiment'].capitalize(),
                "Topics": article["topics"]
                })
            
            # Process individual articles
            processed_articles = []
            for article in data['articles']:
                processed_articles.append({
                    "headline": article['title'],
                    "key_insights": article["summary"],
                    "sentiment": article['sentiment'].capitalize(),
                    "major_topics": article["topics"]
                })
            
            # Display Interface
            st.subheader(f"🔍 Analysis Results for {company_name}")
            
            # Main analysis panel
            with st.expander("View Detailed Report", expanded=True):
                st.json(analysis_report)
            
            # Hindi audio summary
            if data['analysis']['audio_file']:
                st.subheader("🗣️ Hindi Audio Summary")
                audio_url = f"{BACKEND_URL}/audio/{data['analysis']['audio_file']}"
                st.audio(audio_url, format="audio/mpeg")
            
            # Article display
            st.subheader("📰 Article Analysis")
            for article in processed_articles:
                with st.container():
                    st.markdown(f"**{article['headline']}**")
                    st.write(f"Sentiment: {article['sentiment']}")
                    st.write(f"Summary: {article['key_insights']}")
                    st.write(f"Topics: {', '.join(article['major_topics'])}")
                    st.divider()
            
            # Advanced analysis section (commented for future use)
            # with st.expander("딥 Advanced Analytics"):
            #     adv = data['analysis']['adv_analysis']
                
            #     # Sentiment trends visualization
            #     st.subheader("Temporal Sentiment Analysis")
            #     df = pd.DataFrame(adv['sentiment_trends'])
            #     df['date'] = pd.to_datetime(df['date'])
            #     st.line_chart(df.set_index('date')['sentiment'])
                
            #     # Topic modeling visualization
            #     st.subheader("Topic Landscape")
            #     topics = adv['topic_modeling']
            #     topic_df = pd.DataFrame(topics)
            #     st.bar_chart(topic_df['Count'])
                
            #     # Industry benchmarking
            #     st.subheader("Industry Benchmark Comparison")
            #     bench = adv['industry_benchmark']
            #     benchmark_df = pd.DataFrame({
            #         'Metric': ['Positive', 'Negative', 'Neutral'],
            #         'Our Analysis': [
            #             analysis_report['news_analysis']['sentiment_distribution']['POSITIVE'],
            #             analysis_report['news_analysis']['sentiment_distribution']['NEGATIVE'],
            #             analysis_report['news_analysis']['sentiment_distribution']['NEUTRAL']
            #         ],
            #         'Industry Average': [
            #             bench['industry_avg_positive'],
            #             bench['industry_avg_negative'],
            #             bench['industry_avg_neutral']
            #         ]
            #     })
            #     st.bar_chart(benchmark_df.set_index('Metric'))
                
            #     # Stock correlation
            #     st.subheader("Market Correlation")
            #     st.metric(
            #         "Sentiment-Price Correlation",
            #         f"{adv.get('stock_correlation', 0):.2f}",
            #         help="Correlation between news sentiment and stock price movement"
            #     )
                
        else:
            st.error(f"Backend Error: {response.text}")
            
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        st.write("Please ensure:")
        st.write("- Backend server is running at http://localhost:8000")
        st.write("- Network connectivity is available")