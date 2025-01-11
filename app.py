import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import cohere
import pandas as pd
import plotly.express as px

# Initialize models and tokenizer
@st.cache_resource
def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device="cpu"
    )
    return sentiment_pipeline

# Initialize Cohere client
@st.cache_resource
def get_cohere_client():
    COHERE_API_KEY = st.secrets.get("COHERE_API_KEY", "your-default-key")
    return cohere.Client(COHERE_API_KEY)

# Authentication credentials
VALID_USERNAME = st.secrets.get("ADMIN_USERNAME", "admin")
VALID_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "1234")

# Styling
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
    }
    .st-bw {
        background-color: #2d2d2d;
    }
    .stButton>button {
        width: 100%;
        background-color: #1e3a5f;
        color: white;
    }
    .metric-container {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def show_login():
    st.title("🔐 Text Analysis Dashboard")
    
    with st.container():
        st.markdown("### Welcome! Please login to continue")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if username == VALID_USERNAME and password == VALID_PASSWORD:
                    st.session_state["logged_in"] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

def analyze_text(text, sentiment_pipeline, threshold=0.6):
    try:
        # Get sentiment
        sentiment_result = sentiment_pipeline(text)
        if not sentiment_result:
            return None
            
        top_label = sentiment_result[0]["label"].upper()
        top_score = sentiment_result[0]["score"]
        
        # Apply threshold
        final_label = top_label
        if top_score < threshold:
            final_label = "NEUTRAL"
            
        return {
            "sentiment_label": final_label,
            "confidence": top_score
        }
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return None

def create_visualizations(valid_df):
    col1, col2 = st.columns(2)
    
    with col1:
        fig_sentiment = px.pie(
            valid_df,
            names="sentiment_label",
            title="Sentiment Distribution",
            color_discrete_sequence=["#4CAF50", "#f44336"],
            hole=0.4
        )
        fig_sentiment.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        fig_conf = px.histogram(
            valid_df,
            x="confidence",
            color="sentiment_label",
            title="Confidence Distribution",
            color_discrete_sequence=["#4CAF50", "#f44336"]
        )
        fig_conf.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig_conf, use_container_width=True)

def show_main_app():
    st.title("📊 Text Analysis Dashboard")
    st.markdown(f"Welcome back, *{VALID_USERNAME}*!")
    
    # Load models
    sentiment_pipeline = load_sentiment_model()
    co = get_cohere_client()
    
    # Settings sidebar
    with st.sidebar:
        st.markdown("### ⚙ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05
        )
        
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.rerun()
    
    # File upload
    uploaded_files = st.file_uploader(
        "📎 Upload text files for analysis",
        type=["txt"],
        accept_multiple_files=True
    )
    
    if st.button("🔍 Analyze Files"):
        if not uploaded_files:
            st.warning("Please upload at least one file to analyze.")
            return
        
        results = []
        valid_results = []
        
        with st.spinner("Processing files..."):
            for file in uploaded_files:
                try:
                    # Read and decode file content
                    content = file.read().decode("utf-8")
                    
                    # Get summary using Cohere
                    summary = co.summarize(
                        text=content,
                        length="long",
                        format="paragraph",
                        extractiveness="auto",
                        temperature=0.3
                    ).summary
                    
                    # Get sentiment analysis
                    sentiment_data = analyze_text(content, sentiment_pipeline, confidence_threshold)
                    
                    if sentiment_data:
                        result = {
                            "filename": file.name,
                            "summary": summary,
                            **sentiment_data
                        }
                        results.append(result)
                        valid_results.append({
                            "filename": file.name,
                            "sentiment_label": sentiment_data["sentiment_label"],
                            "confidence": sentiment_data["confidence"]
                        })
                    
                except Exception as e:
                    results.append({
                        "filename": file.name,
                        "error": str(e)
                    })
        
        if valid_results:
            df = pd.DataFrame(valid_results)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files", len(df))
            with col2:
                positive_pct = (df["sentiment_label"] == "POSITIVE").mean() * 100
                st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            with col3:
                avg_conf = df["confidence"].mean() * 100
                st.metric("Average Confidence", f"{avg_conf:.1f}%")
            with col4:
                max_conf = df["confidence"].max() * 100
                st.metric("Highest Confidence", f"{max_conf:.1f}%")
            
            # Display results table
            st.markdown("### 📋 Analysis Results")
            for i, result in enumerate(results, 1):
                with st.expander(f"File {i}: {result['filename']}"):
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.markdown(f"**Sentiment:** {result['sentiment_label']}")
                        st.markdown(f"**Confidence:** {result['confidence']:.2%}")
                        st.markdown("**Summary:**")
                        st.markdown(f">{result['summary']}")
            
            # Show visualizations
            st.markdown("### 📊 Analysis Visualizations")
            create_visualizations(df)

def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    
    if st.session_state["logged_in"]:
        show_main_app()
    else:
        show_login()

if __name__ == "__main__":
    main()