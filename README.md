# 📊 Sentiment Analysis Dashboard

A powerful web application for analyzing sentiment in text files using state-of-the-art NLP models. This application combines the summarization capabilities of Cohere with BERT-based sentiment analysis to provide accurate and insightful text analysis.

## 🌟 Features

- **Advanced Text Analysis**: Combines Cohere's summarization with BERT sentiment analysis
- **Batch Processing**: Analyze multiple text files simultaneously
- **Interactive Dashboard**: Real-time visualization of sentiment analysis results
- **Secure Access**: User authentication system
- **Professional UI**: Clean, modern interface with dark theme
- **Detailed Analytics**: 
  - Sentiment distribution
  - Confidence scores
  - Individual file analysis
  - Summary generation

## 🚀 Live Demo

You can access the live application here: [Sentiment Analysis Dashboard](https://fixitappdeploy.streamlit.app/)

Login credentials:
- Username: `admin`
- Password: `1234`

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **NLP Models**: 
  - Cohere API for text summarization
  - DistilBERT for sentiment analysis
- **Data Visualization**: Plotly
- **Styling**: Custom CSS with dark theme
- **Authentication**: Streamlit native session state

## 📋 Requirements

```
streamlit==1.29.0
pandas==2.1.4
plotly==5.18.0
cohere==4.32
transformers==4.35.2
torch==2.2.0
python-dotenv==1.0.0
```

## 🔧 Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/sentiment-analysis-dashboard.git
   cd sentiment-analysis-dashboard
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create `.streamlit/secrets.toml`:
   ```toml
   ADMIN_USERNAME = "your_username"
   ADMIN_PASSWORD = "your_password"
   COHERE_API_KEY = "your_cohere_api_key"
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📝 Usage

1. **Login**: Access the dashboard using your credentials
2. **Upload Files**: Select one or more text files for analysis
3. **Analysis**: Click "Analyze Files" to process the uploads
4. **Results**: View results in three formats:
   - Summary metrics
   - Detailed analysis table
   - Interactive visualizations

## 🎯 Features in Detail

### Metrics Dashboard
- Total Files Processed
- Positive Sentiment Percentage
- Average Confidence Score
- Highest Confidence Score

### Analysis Results
- File-wise sentiment classification
- Confidence scores
- Processing status
- Detailed summaries

### Visualizations
- Sentiment Distribution (Pie Chart)
- Confidence Distribution (Histogram)
- Confidence Timeline (Line Chart)

## 💡 Implementation Details

The application uses a two-step process for analysis:
1. **Text Summarization**: Uses Cohere's AI to generate concise summaries
2. **Sentiment Analysis**: Processes summaries through BERT for sentiment classification

This approach solves the token limit constraints while maintaining accuracy.

---------

## 🙏 Acknowledgments

- Cohere for their powerful NLP API
- Hugging Face for the transformers library
- Streamlit for the amazing web framework

---
Created with ❤️ by Deepak Thirukkumaran
