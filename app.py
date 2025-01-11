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

@st.cache_resource
def get_cohere_client():
    COHERE_API_KEY = st.secrets.get("COHERE_API_KEY", "your-default-key")
    return cohere.Client(COHERE_API_KEY)

VALID_USERNAME = st.secrets.get("ADMIN_USERNAME", "admin")
VALID_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "1234")

st.markdown("""
<style>
    body {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stApp {
        background-color: #1a1a1a;
    }
    .main {
        background-color: #1a1a1a;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2d2d2d;
        color: white;
        border: 1px solid #404040;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #404040;
        border-color: #505050;
    }
    .styled-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 1.5rem 0;
        background-color: #2d2d2d;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .styled-table th {
        background-color: #1e3a5f;
        color: #ffffff;
        padding: 16px;
        text-align: left;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .styled-table td {
        padding: 16px;
        border-top: 1px solid #404040;
        color: #ffffff;
        font-size: 0.9rem;
    }
    .styled-table tr:hover {
        background-color: #363636;
        transition: background-color 0.2s ease;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .metric-card {
        background-color: #1e3a5f;
        padding: 1.5rem;
        border-radius: 8px;
        flex: 1;
        min-width: 200px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .detailed-analysis {
        background-color: #2d2d2d;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-success {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
    .summary-container {
        background-color: #363636;
        padding: 1.5rem;
        border-radius: 4px;
        margin-top: 1rem;
        border-left: 4px solid #1e3a5f;
    }
    .sentiment-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #f44336;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #FFC107;
        font-weight: bold;
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

def get_text_summary(text, co):
    try:
        response = co.summarize(
            text=text,
            length='medium',
            format='paragraph',
            extractiveness="high",
            temperature=0.3,
            additional_command="Focus on emotional tone and key points that indicate sentiment."
        )
        return response.summary
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        return text[:1000]

def analyze_text(text, sentiment_pipeline, threshold=0.6):
    try:
        sentiment_result = sentiment_pipeline(text[:512])
        if not sentiment_result:
            return None
            
        top_label = sentiment_result[0]["label"].upper()
        top_score = sentiment_result[0]["score"]
        
        final_label = top_label
        if top_score < threshold:
            final_label = "NEUTRAL"
            
        return {
            "sentiment_label": final_label,
            "confidence": top_score
        }
        
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return None

def process_file_content(file_content, co, sentiment_pipeline, confidence_threshold):
    try:
        summary = get_text_summary(file_content, co)
        sentiment_data = analyze_text(summary, sentiment_pipeline, confidence_threshold)
        
        if sentiment_data:
            return {
                "summary": summary,
                **sentiment_data
            }
        return None
        
    except Exception as e:
        st.error(f"Error in processing: {str(e)}")
        return None

def create_visualizations(valid_df):
    col1, col2 = st.columns(2)
    
    with col1:
        fig_sentiment = px.pie(
            valid_df,
            names="sentiment_label",
            title="Sentiment Distribution",
            color_discrete_sequence=["#4CAF50", "#f44336", "#FFC107"],
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
            color_discrete_sequence=["#4CAF50", "#f44336", "#FFC107"]
        )
        fig_conf.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    fig_timeline = px.line(
        valid_df,
        x=valid_df.index,
        y="confidence",
        color="sentiment_label",
        title="Confidence Timeline",
        color_discrete_sequence=["#4CAF50", "#f44336", "#FFC107"]
    )
    fig_timeline.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis_title="File Number",
        yaxis_title="Confidence Score"
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

def get_sentiment_class(sentiment):
    if sentiment == "POSITIVE":
        return "sentiment-positive"
    elif sentiment == "NEGATIVE":
        return "sentiment-negative"
    return "sentiment-neutral"

def show_main_app():
    st.title("📊 Text Analysis Dashboard")
    st.markdown(f"Welcome back, *{VALID_USERNAME}*!")
    
    sentiment_pipeline = load_sentiment_model()
    co = get_cohere_client()
    
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
        
        with st.spinner("Processing files... This might take a moment."):
            for file in uploaded_files:
                try:
                    content = file.read().decode("utf-8")
                    processed_data = process_file_content(
                        content, 
                        co, 
                        sentiment_pipeline, 
                        confidence_threshold
                    )
                    
                    if processed_data:
                        result = {
                            "filename": file.name,
                            **processed_data
                        }
                        results.append(result)
                        valid_results.append({
                            "filename": file.name,
                            "sentiment_label": processed_data["sentiment_label"],
                            "confidence": processed_data["confidence"]
                        })
                    
                except Exception as e:
                    results.append({
                        "filename": file.name,
                        "error": str(e)
                    })
        
        if valid_results:
            df = pd.DataFrame(valid_results)
            
            # Metrics Dashboard
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-label">Total Files</div>
                    <div class="metric-value">{len(df)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Positive Sentiment</div>
                    <div class="metric-value">{(df['sentiment_label'] == 'POSITIVE').mean() * 100:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Average Confidence</div>
                    <div class="metric-value">{df['confidence'].mean() * 100:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Highest Confidence</div>
                    <div class="metric-value">{df['confidence'].max() * 100:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Analysis Results Table
            st.markdown("### 📋 Analysis Results")
            
            table_html = "<table class='styled-table'><thead><tr>"
            headers = ["#", "Filename", "Sentiment", "Confidence", "Status"]
            
            for header in headers:
                table_html += f"<th>{header}</th>"
            
            table_html += "</tr></thead><tbody>"
            
            for i, result in enumerate(results, 1):
                table_html += "<tr>"
                table_html += f"<td>{i}</td>"
                table_html += f"<td>{result['filename']}</td>"
                
                if "error" in result:
                    table_html += "<td colspan='2'>N/A</td>"
                    table_html += f"<td class='status-error'>❌ Error</td>"
                else:
                    sentiment_class = get_sentiment_class(result['sentiment_label'])
                    table_html += f"<td class='{sentiment_class}'>{result['sentiment_label']}</td>"
                    table_html += f"<td>{result['confidence']:.2%}</td>"
                    table_html += f"<td class='status-success'>✅ Success</td>"
                
                table_html += "</tr>"
            
            table_html += "</tbody></table>"
            st.markdown(table_html, unsafe_allow_html=True)
            
            # Detailed Analysis Section
            st.markdown("### 📝 Detailed Analysis")
            for i, result in enumerate(results, 1):
                with st.expander(f"File {i}: {result['filename']}"):
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.markdown(f"""
                        <div class='detailed-analysis'>
                            <h4>Analysis Results</h4>
                            <p><strong>Sentiment:</strong> <span class='{get_sentiment_class(result["sentiment_label"])}'>{result["sentiment_label"]}</span></p>
                            <p><strong>Confidence Score:</strong> {result["confidence"]:.2%}</p>
                            <div class='summary-container'>
                                <h4>Content Summary</h4>
                                <p>{result["summary"]}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("### 📊 Analysis Visualizations")
            create_visualizations(df)

def main():
    """Main application entry point"""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    
    if st.session_state["logged_in"]:
        show_main_app()
    else:
        show_login()

if __name__ == "__main__":
    main()
