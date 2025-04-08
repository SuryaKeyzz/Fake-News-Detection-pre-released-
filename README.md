# Fake News Detection System using Spark LLM with RAG

This repository contains a comprehensive fake news detection system that combines modern NLP techniques, embeddings, and the Spark LLM to evaluate the truthfulness of claims.

## 🔍 Overview

The system uses a Retrieval-Augmented Generation (RAG) approach to fact-check claims:

1. **Web Search**: Retrieves relevant articles using Google Search API
2. **Embedding-based Similarity**: Pre-filters content using semantic embeddings 
3. **FAISS Vector Search**: Efficiently finds the most similar documents
4. **Spark LLM Fact Checking**: Uses the Spark LLM for fact-checking and verdict generation
5. **Sentiment Analysis**: Detects emotional manipulation in claims

## 🛠️ System Components

### 1. WebSearchEngine
Retrieves articles from the web using the Google Custom Search API.

### 2. EmbeddingEngine 
Creates vector embeddings for text using SentenceTransformer models and calculates similarity.

### 3. FAISSIndexer
Enables efficient similarity search with Facebook AI Similarity Search (FAISS).

### 4. SparkLLM
Uses the Spark LLM via HTTP API to perform sophisticated fact-checking on claims against evidence.

### 5. SentimentAnalyzer
Analyzes emotional content and manipulation techniques in text.

### 6. FakeNewsDetectionSystem
Integrates all components into a comprehensive detection system.

## 📋 Prerequisites

- Python 3.8+  
- pip (Python package installer)

## 📦 Required Libraries

```
numpy 
pandas
requests
scikit-learn
sentence-transformers
faiss-cpu (or faiss-gpu for GPU support)
nltk
python-dotenv
```

## 🔑 API Keys and Environment Setup

This system requires the following API keys:
- Google API Key (for web search)
- Google Custom Search Engine ID (CX)  
- Spark LLM API Key and Secret

### Getting API Keys

#### Google Search API:
1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to "APIs & Services" > "Library"
4. Search for and enable the "Custom Search API"
5. Go to "APIs & Services" > "Credentials"  
6. Click "Create Credentials" > "API Key"
7. Copy your API key

#### Google Custom Search Engine ID:
1. Go to the [Google Programmable Search Engine](https://programmablesearchengine.google.com/)
2. Click "Create a programmable search engine"
3. Configure your search engine (select "Search the entire web" for best results)
4. After creation, find your "Search engine ID" (CX) in the setup page

### Setting Up Environment Variables

Create a `.env` file in the project root with your API keys:

```
# Google API Credentials
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CX=your_google_custom_search_engine_id_here

# Spark LLM API 
SPARK_API_KEY=your_spark_api_key_here
SPARK_API_SECRET=your_spark_api_secret_here
```

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection  
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create and set up your `.env` file as described above

## 💻 Usage

Run the `project.py` script to analyze a set of example claims:

```bash 
python project.py
```

This will output the analysis results for each claim, including:
- Sentiment analysis (emotional intensity, dominant emotion)
- Fact-checking result from Spark LLM with reasoning and sources

### Using as a Library

You can also use the `FakeNewsDetectionSystem` class in your own code:

```python
from Try_model import FakeNewsDetectionSystem

detector = FakeNewsDetectionSystem()
claim = "Example claim to analyze"
result = detector.analyze_claim(claim)

if result["status"] == "success":
    print(result["fact_check_result"])
else:
    print(f"Error: {result['message']}")
```

## 🔒 Privacy and Ethics  

This system is designed for educational and research purposes. Please use responsibly and respect privacy and ethical considerations when deploying fact-checking systems.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
