"""
Fake News Detection System using Spark LLM with RAG
===================================================
This system combines:
1. Web search for retrieving relevant articles
2. Embedding-based similarity for pre-filtering
3. FAISS for efficient similarity search
4. Spark LLM for fact-checking and verdict generation
5. Sentiment analysis to detect emotional manipulation
6. Named Entity Recognition to extract key entities
"""

# Required imports
import os
import json
import base64
import hashlib
import hmac
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, UTC
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load environment variables (still keeping this for other credentials)
load_dotenv()

# API Credentials
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CX = os.getenv('GOOGLE_CX')

# Directly setting the Spark API credentials
# Using the provided API password: csBSUzDSssGCMPgUOvdO:qNoapyBeUOJMJSkCWjdN
SPARK_API_KEY = "owiAWXktxkcuMTmNBGRT"
SPARK_API_SECRET = "JsNMhluvPYyNODivAMsA"


# Validate credentials
def validate_credentials():
    """Validate that all required API credentials are present"""
    # Check for Google credentials
    missing_google = []
    if not GOOGLE_API_KEY:
        missing_google.append('Google Search API Key')
    if not GOOGLE_CX:
        missing_google.append('Google Custom Search Engine ID')
    
    if missing_google:
        print(f"Warning: Missing Google credentials: {', '.join(missing_google)}")
        print("Web search functionality will be limited.")
    
     # Check Spark API credentials
    if not SPARK_API_KEY:
        raise ValueError("Missing required Spark API Key")
    if not SPARK_API_SECRET:
        raise ValueError("Missing required Spark API Secret")
    
    print("✓ Spark API credentials are configured")
    print(f"API Key: {SPARK_API_KEY[:5]}... (length: {len(SPARK_API_KEY)})")
    
    return True

# Web search component
class WebSearchEngine:
    """Component for retrieving articles from the web using Google Search API"""
    
    def __init__(self, api_key: str, cx: str):
        """Initialize the search engine with API credentials
        
        Args:
            api_key: Google API Key
            cx: Google Custom Search Engine ID
        """
        self.api_key = api_key
        self.cx = cx
        self.search_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """Search for articles related to a query
        
        Args:
            query: The search query
            num_results: Number of results to retrieve (max 10 for free tier)
            
        Returns:
            List of dictionaries containing article information
        """
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": min(num_results, 10)  # Max 10 for free tier
        }
        
        try:
            response = requests.get(self.search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant information
            articles = []
            for item in data.get("items", []):
                article = {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "source": item.get("displayLink", ""),
                    "published_date": item.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time", "")
                }
                
                # Create a full text field combining title and snippet
                article["full_text"] = f"{article['title']} {article['snippet']}"
                articles.append(article)
                
            return articles
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print("Full API Response:", response.text)
            return []
        except Exception as e:
            print(f"Error: {e}")
            return []

# Embedding and similarity component
class EmbeddingEngine:
    """Component for creating and comparing text embeddings"""
    
    def __init__(self, model_name: str = "distilbert-base-nli-mean-tokens"):
        """Initialize with a sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            print(f"✓ Model '{model_name}' loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error loading model {model_name}: {e}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
            
        return self.model.encode(texts)
    
    def compute_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and documents
        
        Args:
            query_embedding: Embedding of the query
            document_embeddings: Embeddings of the documents
            
        Returns:
            Array of similarity scores
        """
        return cosine_similarity(query_embedding.reshape(1, -1), document_embeddings)[0]
    
    def filter_by_threshold(self, texts: List[str], similarities: np.ndarray, threshold: float = 0.70) -> List[str]:
        """Filter texts based on similarity threshold
        
        Args:
            texts: List of text strings
            similarities: Array of similarity scores
            threshold: Minimum similarity score to keep
            
        Returns:
            Filtered list of texts
        """
        return [text for text, sim in zip(texts, similarities) if sim >= threshold]

# FAISS index for efficient similarity search
class FAISSIndexer:
    """Component for efficient similarity search using FAISS"""
    
    def __init__(self):
        """Initialize the FAISS indexer"""
        self.index = None
        self.dimension = None
    
    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create a FAISS index from embeddings
        
        Args:
            embeddings: Document embeddings
            
        Returns:
            FAISS index
        """
        if embeddings is None or embeddings.size == 0:
            raise ValueError("No embeddings provided for FAISS index")
            
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        return self.index
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar documents
        
        Args:
            query_embedding: Embedding of the query
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("FAISS index not created yet")
            
        return self.index.search(query_embedding, k)
    
    def retrieve_docs(self, query_embedding: np.ndarray, articles: List[Dict[str, str]], k: int = 5) -> List[Dict[str, str]]:
        """Retrieve similar documents for a query
        
        Args:
            query_embedding: Embedding of the query
            articles: List of article dictionaries
            k: Number of results to return
            
        Returns:
            List of retrieved articles
        """
        _, indices = self.search(query_embedding, k)
        return [articles[i] for i in indices[0] if i < len(articles)]

# Spark LLM API component (HTTP method only)
class SparkLLM:
    """Component for interacting with the Spark LLM API (HTTP API only)"""
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        """Initialize with Spark API credentials
        
        Args:
            api_key: Spark API Key
            api_secret: Spark API Secret
        """
        # Handle various initialization patterns
        if api_key and ":" in api_key and not api_secret:
            # Format: "key:secret" passed as first parameter
            self.api_key, self.api_secret = api_key.split(":", 1)
        elif api_key and api_secret:
            # Format: key and secret passed separately
            self.api_key = api_key
            self.api_secret = api_secret
        else:
            # No proper credentials provided
            self.api_key = api_key
            self.api_secret = api_secret
            print("Warning: Incomplete API credentials provided")
        
        # HTTP API URL from the documentation
        self.api_url = "https://spark-api-open.xf-yun.com/v1/chat/completions"
    
    def build_fact_check_prompt(self, claim: str, retrieved_articles: List[Dict[str, str]]) -> str:
        """Build a prompt for fact-checking
        
        Args:
            claim: The claim to fact-check
            retrieved_articles: List of retrieved articles
            
        Returns:
            Prompt string
        """
        # Format evidence from retrieved articles
        evidence_text = ""
        for i, article in enumerate(retrieved_articles[:3], 1):  # Use top 3 articles
            title = article.get("title", "Untitled")
            snippet = article.get("snippet", "")
            source = article.get("source", "Unknown")
            link = article.get("link", "")
            
            evidence_text += f"[Article {i}] {title}\n"
            evidence_text += f"Source: {source}\n"
            evidence_text += f"Content: {snippet}\n"
            evidence_text += f"URL: {link}\n\n"
        
        # Truncate if too long
        evidence_text = evidence_text[:2500]
        
        # Build the prompt with chain-of-thought and few-shot examples
        prompt = f"""
[Role]
You are a professional fact-checker working for Reuters. Analyze the claim below using the provided evidence.

[Claim]
{claim}

[Evidence]
{evidence_text}

[Instructions]
Follow these steps in sequence:
1. Identify key entities (people, organizations, dates) in the claim
2. For each key entity, compare with evidence to check consistency
3. Note any contradictions or confirmations with evidence
4. Analyze the sentiment and emotional language in both claim and evidence
5. Determine if there's sufficient evidence to make a determination
6. If evidence is insufficient, specify what additional information would be needed
7. Make a final determination with confidence rating

[Few-shot Examples]
Example 1:
Claim: "WHO announced vaccines cause heart attacks in 30% of recipients"
Analysis:
- Key entities: WHO, vaccines, heart attacks, 30% statistic
- WHO statements in evidence contradict the claim
- No peer-reviewed studies cited support the 30% statistic
- Emotional language detected: fear-inducing terminology
- Verdict: False (95% confidence)

Example 2:
Claim: "New study shows coffee reduces cancer risk by 15%"
Analysis:
- Key entities: coffee, cancer risk, 15% reduction, new study
- Evidence confirms Harvard published a study on coffee and cancer
- The 15% figure matches the study's findings
- Emotional language: neutral presentation of scientific finding
- Verdict: True (90% confidence)

[Final Format]
Verdict: "True" or "False" or "Partially True" or "Unverified"
Confidence: 0-100%
Reasoning:
- Step 1: Entity analysis
- Step 2: Evidence comparison
- Step 3: Contradiction/confirmation analysis
- Step 4: Sentiment analysis
- Step 5: Evidence sufficiency
Explanation: 2-3 sentences summarizing the reasoning
Sources: List supporting or contradicting articles
"""
        return prompt
    
    
    def generate_auth_headers(self, host="spark-api-open.xf-yun.com", method="POST", path="/v1/chat/completions"):
        """Generate HMAC authentication headers for Spark API
        
        Args:
            host: API host
            method: HTTP method
            path: API endpoint path
            
        Returns:
            Dictionary with authentication headers
        """
        # Create RFC 1123 date format (required by many API services for HTTP Date header)
        # Format example: "Tue, 01 Apr 2025 10:52:31 GMT"
        date_header = datetime.now(UTC).strftime('%a, %d %b %Y %H:%M:%S GMT')
        
        # Create signature string - using proper date format
        signature_str = f"host: {host}\ndate: {date_header}\n{method} {path} HTTP/1.1"
        
        # Create HMAC signature using the API secret
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                signature_str.encode('utf-8'),
                digestmod=hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        # Format the authorization header
        authorization = f"api_key=\"{self.api_key}\", algorithm=\"hmac-sha256\", headers=\"host date request-line\", signature=\"{signature}\""
        
        # Return headers with proper Date header format
        return {
            "Content-Type": "application/json",
            "Host": host,
            "Date": date_header,
            "Authorization": authorization
        }
    
    
    
    
    def fact_check(self, claim: str, retrieved_articles: List[Dict[str, str]]) -> Optional[str]:
        """Perform fact-checking using Spark LLM via HTTP API
        
        Args:
            claim: The claim to fact-check
            retrieved_articles: List of retrieved articles
            
        Returns:
            Fact-checking result
        """
        if not self.api_key or not self.api_secret:
            print("Error: API Key and Secret are required for HMAC authentication")
            return None
            
        prompt = self.build_fact_check_prompt(claim, retrieved_articles)
        
        # Using the format from the documentation
        payload = {
            "model": "generalv3.5",  # Using general version
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1500,
            "stream": False  # Non-streaming mode
        }
        
        # Generate authentication headers
        headers = self.generate_auth_headers()
        
        try:
            print("Sending request to Spark HTTP API...")
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            # Debug information
            print(f"API Response Status: {response.status_code}")
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the content from the response (based on documentation format)
            try:
                content = result["choices"][0]["message"]["content"]
                return content
            except KeyError as e:
                print(f"Error extracting content from response: {e}")
                print(f"Response structure: {json.dumps(result, indent=2)}")
                return None
                
        except Exception as e:
            print(f"Spark API Error: {str(e)}")
            if 'response' in locals():
                print(f"Response text: {response.text}")
            return None

# Sentiment analysis component
class SentimentAnalyzer:
    """Component for analyzing sentiment and emotional manipulation in text"""
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.vader = SentimentIntensityAnalyzer()
        
        # Emotion keywords
        self.emotion_keywords = {
            'fear': ['fear', 'danger', 'threat', 'alarming', 'frightening', 'terrifying', 'scary', 'panic'],
            'anger': ['anger', 'rage', 'fury', 'outrage', 'angry', 'furious', 'enraged', 'mad'],
            'disgust': ['disgust', 'disgusting', 'repulsive', 'sickening', 'revolting', 'gross', 'vile'],
            'surprise': ['surprise', 'shocking', 'unbelievable', 'incredible', 'astonishing', 'stunning', 'dramatic'],
            'urgency': ['urgent', 'emergency', 'immediately', 'hurry', 'act now', 'don\'t wait', 'limited time'],
            'conspiracy': ['conspiracy', 'secret', 'cover-up', 'hidden', 'they don\'t want you to know', 'censored'],
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment in text using VADER
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of sentiment scores
        """
        return self.vader.polarity_scores(text)
    
    def analyze_emotion_keywords(self, text: str) -> Dict[str, float]:
        """Analyze the presence of emotion-laden keywords in the text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion keyword scores
        """
        text_lower = text.lower()
        results = {}
        
        total_words = len(text_lower.split())
        if total_words == 0:
            return {emotion: 0.0 for emotion in self.emotion_keywords}
        
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(text_lower.count(keyword) for keyword in keywords)
            # Normalize by text length to get a relative score
            score = count / total_words
            results[emotion] = float(score)
            
        return results
    
    def analyze_sentence_polarities(self, text: str) -> Dict[str, float]:
        """Analyze the distribution of sentence polarities in the text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with polarity statistics
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        if not sentences:
            return {
                "polarity_mean": 0.0,
                "polarity_std": 0.0,
                "polarity_max": 0.0,
                "polarity_min": 0.0,
                "subjectivity_mean": 0.0,
                "extreme_sentence_ratio": 0.0,
            }
        
        # Calculate polarity for each sentence
        polarities = []
        extreme_sentences = 0
        
        for sentence in sentences:
            scores = self.analyze_sentiment(sentence)
            polarities.append(scores["compound"])
            
            # Count extreme polarity sentences (strong positive or negative)
            if abs(scores["compound"]) > 0.5:
                extreme_sentences += 1
        
        # Calculate statistics
        polarities_array = np.array(polarities)
        
        return {
            "polarity_mean": float(np.mean(polarities_array)),
            "polarity_std": float(np.std(polarities_array)),
            "polarity_max": float(np.max(polarities_array)),
            "polarity_min": float(np.min(polarities_array)),
            "extreme_sentence_ratio": float(extreme_sentences / len(sentences)),
        }
    
    def detect_emotional_manipulation(self, text: str) -> Dict[str, Any]:
        """Detect potential emotional manipulation in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with manipulation metrics
        """
        # Get basic sentiment
        sentiment_scores = self.analyze_sentiment(text)
        
        # Get emotion keywords
        emotion_keywords = self.analyze_emotion_keywords(text)
        
        # Get sentence polarity statistics
        polarity_stats = self.analyze_sentence_polarities(text)
        
        # Calculate manipulation score based on several factors:
        # 1. High emotional content (extreme sentiment)
        # 2. High variance in sentence polarities (emotional rollercoaster)
        # 3. High frequency of emotional keywords
        # 4. High subjectivity
        
        factors = []
        
        # Factor 1: Extreme sentiment (very negative or very positive)
        sentiment_extremity = abs(sentiment_scores["compound"])
        factors.append(sentiment_extremity)
        
        # Factor 2: Sentence polarity variance
        polarity_variance = polarity_stats["polarity_std"]
        factors.append(min(polarity_variance * 2, 1.0))  # Scale up but cap at 1.0
        
        # Factor 3: Emotional keyword density
        emotion_keyword_score = sum(emotion_keywords.values())
        factors.append(min(emotion_keyword_score * 5, 1.0))  # Scale up but cap at 1.0
        
        # Factor 4: Extreme sentence ratio
        extreme_sentence_ratio = polarity_stats["extreme_sentence_ratio"]
        factors.append(extreme_sentence_ratio)
        
        # Calculate the overall manipulation score (weighted average)
        weights = [0.3, 0.2, 0.3, 0.2]  # Weights for each factor
        manipulation_score = sum(f * w for f, w in zip(factors, weights))
        
        # Qualitative assessment
        if manipulation_score < 0.3:
            manipulation_level = "LOW"
            explanation = "The text appears to be relatively neutral and factual, with limited emotional manipulation."
        elif manipulation_score < 0.6:
            manipulation_level = "MODERATE"
            explanation = "The text contains some emotional language and manipulation techniques, but is not extremely manipulative."
        else:
            manipulation_level = "HIGH"
            explanation = "The text shows strong signs of emotional manipulation, using extreme language and emotional appeals."
        
        # Add detailed explanation based on the highest contributing factors
        explanation_details = []
        if sentiment_extremity > 0.7:
            explanation_details.append("Uses extremely emotional language.")
        if polarity_variance > 0.5:
            explanation_details.append("Contains dramatic shifts in emotional tone.")
        if emotion_keyword_score > 0.2:
            top_emotions = sorted(emotion_keywords.items(), key=lambda x: x[1], reverse=True)[:2]
            emotion_str = " and ".join([f"{emotion}" for emotion, score in top_emotions if score > 0])
            if emotion_str:
                explanation_details.append(f"Appeals heavily to {emotion_str}.")
        if extreme_sentence_ratio > 0.5:
            explanation_details.append("Contains many emotionally charged statements.")
            
        if explanation_details:
            explanation += " " + " ".join(explanation_details)
        
        # Determine dominant emotion
        if sentiment_scores["compound"] >= 0.5:
            dominant_emotion = "strongly positive"
        elif sentiment_scores["compound"] > 0 and sentiment_scores["compound"] < 0.5:
            dominant_emotion = "mildly positive"
        elif sentiment_scores["compound"] > -0.5 and sentiment_scores["compound"] <= 0:
            dominant_emotion = "mildly negative"
        else:
            dominant_emotion = "strongly negative"
        
        # Add dominant emotion from keywords if available
        keyword_emotions = [(emotion, score) for emotion, score in emotion_keywords.items() if score > 0.05]
        if keyword_emotions:
            top_keyword_emotion = max(keyword_emotions, key=lambda x: x[1])[0]
            dominant_emotion += f" with {top_keyword_emotion} undertones"
        
        return {
            "sentiment_scores": sentiment_scores,
            "emotional_intensity": sentiment_extremity,
            "is_emotionally_charged": manipulation_score > 0.5,
            "dominant_emotion": dominant_emotion,
            "manipulation_score": float(manipulation_score),
            "manipulation_level": manipulation_level,
            "explanation": explanation,
            "details": {
                "emotion_keywords": emotion_keywords,
                "polarity_stats": polarity_stats,
                "contributing_factors": {
                    "sentiment_extremity": float(sentiment_extremity),
                    "polarity_variance": float(polarity_variance),
                    "emotion_keyword_density": float(emotion_keyword_score),
                    "extreme_sentence_ratio": float(extreme_sentence_ratio)
                }
            }
        }

# Main Fake News Detection System
class FakeNewsDetectionSystem:
    """Complete system for fake news detection using Spark LLM and RAG"""
    
    def __init__(self):
        """Initialize the fake news detection system"""
        validate_credentials()
        
        # Initialize components
        self.search_engine = WebSearchEngine(GOOGLE_API_KEY, GOOGLE_CX)
        self.embedding_engine = EmbeddingEngine()
        self.faiss_indexer = FAISSIndexer()
        
        # Initialize Spark LLM with API password
        self.spark_llm = SparkLLM(SPARK_API_KEY, SPARK_API_SECRET)
        
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze_claim(self, claim: str) -> Dict[str, Any]:
        """Analyze a claim for fake news detection
        
        Args:
            claim: The claim to analyze
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\n--- Analyzing claim: {claim} ---\n")
        
        # Step 1: Search for relevant articles
        print("Searching for relevant articles...")
        articles = self.search_engine.search(claim)
        
        if not articles:
            return {
                "status": "error",
                "message": "No articles found for the claim"
            }
        
        print(f"Found {len(articles)} articles")
        
        # Step 2: Create embeddings
        print("Creating embeddings...")
        texts = [article["full_text"] for article in articles]
        embeddings = self.embedding_engine.get_embeddings(texts)
        
        # Step 3: Create FAISS index
        print("Creating FAISS index...")
        self.faiss_indexer.create_index(embeddings)
        
        # Step 4: Retrieve similar documents
        print("Retrieving similar documents...")
        claim_embedding = self.embedding_engine.get_embeddings([claim])
        retrieved_articles = self.faiss_indexer.retrieve_docs(claim_embedding, articles)
        
        print(f"Retrieved {len(retrieved_articles)} relevant articles")
        
        # Step 5: Analyze sentiment of the claim
        print("Analyzing sentiment...")
        sentiment_analysis = self.sentiment_analyzer.detect_emotional_manipulation(claim)
        
        # Step 6: Fact-check using Spark LLM
        print("Fact-checking with Spark LLM...")
        fact_check_result = self.spark_llm.fact_check(claim, retrieved_articles)
        
        if not fact_check_result:
            return {
                "status": "error",
                "message": "Failed to get fact-checking result from Spark LLM"
            }
        
        # Step 7: Compile and return results
        print("Compiling results...")
        return {
            "status": "success",
            "claim": claim,
            "retrieved_articles": retrieved_articles,
            "sentiment_analysis": sentiment_analysis,
            "fact_check_result": fact_check_result
        }
