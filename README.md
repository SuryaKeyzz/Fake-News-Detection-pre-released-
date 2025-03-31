# Fake News Detection System using BERT with RAG

This repository contains a comprehensive fake news detection system that combines modern NLP techniques, embeddings, and BERT models to evaluate the truthfulness of claims.

## 🔍 Overview

The system uses a Retrieval-Augmented Generation (RAG) approach to fact-check claims:

1. **Web Search**: Retrieves relevant articles using Google Search API
2. **Embedding-based Similarity**: Pre-filters content using semantic embeddings
3. **FAISS Vector Search**: Efficiently finds the most similar documents
4. **BERT-based Fact Checking**: Uses a fine-tuned BERT model for fact-checking and verdict generation
5. **Sentiment Analysis**: Detects emotional manipulation in claims

## 🛠️ System Components

### 1. WebSearchEngine
Retrieves articles from the web using the Google Custom Search API.

### 2. EmbeddingEngine
Creates vector embeddings for text using SentenceTransformer models and calculates similarity.

### 3. FAISSIndexer
Enables efficient similarity search with Facebook AI Similarity Search (FAISS).

### 4. BERTFactChecker
Uses a fine-tuned BERT model to perform sophisticated fact-checking on claims against evidence.

### 5. SentimentAnalyzer
Analyzes emotional content to detect potential manipulation.

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
transformers
torch
faiss-cpu (or faiss-gpu for GPU support)
nltk
python-dotenv
```

## 🔑 API Keys Required

This system requires the following API keys:
- Google API Key (for web search)
- Google Custom Search Engine ID

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

3. Create a `.env` file in the project root with your API keys:
```
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CX=your_google_custom_search_id
```

## 💻 Usage

Basic usage example:

```python
from Try_model import FakeNewsDetectionSystem

# Initialize the system
detector = FakeNewsDetectionSystem()

# Analyze a claim
claim = "COVID-19 vaccines contain microchips for tracking people"
result = detector.analyze_claim(claim)

# Print results
if result["status"] == "success":
    print(f"Claim: {result['claim']}")
    print(f"Fact Check Result: {result['fact_check_result']}")
```

## 🔄 How It Works

1. The system searches for relevant articles about the claim
2. It creates embeddings for the articles and the claim
3. Using FAISS, it retrieves the most similar articles to the claim
4. It analyzes the sentiment of the claim to detect emotional manipulation
5. The BERT model performs fact-checking based on the retrieved articles
6. Results are compiled into a comprehensive analysis

## 📊 Example Output

For the claim "COVID-19 vaccines contain microchips for tracking people":

```
=== ANALYSIS RESULTS ===
Claim: COVID-19 vaccines contain microchips for tracking people

Sentiment Analysis:
Emotional Intensity: 0.34
Dominant Emotion: mildly negative
Is Emotionally Charged: False

Fact Check Result:
Verdict: False
Confidence: 95%
Reasoning:
- Step 1: Key entities identified - COVID-19 vaccines, microchips, tracking people
- Step 2: None of the evidence supports the claim that vaccines contain microchips
- Step 3: Multiple reputable sources directly contradict the claim
- Step 4: Claim uses fear-inducing language about surveillance
- Step 5: Evidence is sufficient to make a determination
Explanation: The claim that COVID-19 vaccines contain microchips is a debunked conspiracy theory. Multiple health authorities and fact-checking organizations have confirmed vaccines do not contain tracking devices.
Sources: [Article 1], [Article 2], [Article 3]
```

## 🔒 Privacy and Ethics

This system is designed for educational and research purposes. Please use responsibly and respect privacy and ethical considerations when deploying fact-checking systems.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
