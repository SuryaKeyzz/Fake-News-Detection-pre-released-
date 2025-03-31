import os
import json
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import List,Dict,Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

sia = SentimentIntensityAnalyzer()

# load environment
load_dotenv()

# using APi credentials
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CX  = os.getenv('GOOGLE_CX')
HF_API_TOKEN = os.getenv('HF_API_TOKEN')

# validating the credentials
def validate_credentials():
    required_var = {
        'Google Search API key' : GOOGLE_API_KEY,
        'Google Custome Search Engine ID' : GOOGLE_CX
    }
    
    missing = [name for name,value in required_var.items() if not value]
    
    if missing:
        raise ValueError(f"Missing required API credentials : {', '.join(missing)}")

    print("All Api credentials are available")
    if HF_API_TOKEN :
        print(f"Hugging Face API Token is available: {HF_API_TOKEN[:3]}")
    else:
        print("Hugging Face API cannot be found. will use local models only.")
    
    return True

class WebSearchEngine:
    def __init__(self,api_key: str, cx: str):
        self.api_key = api_key
        self.cx = cx
        self.search_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self,query:str,num_results:int = 10) -> List[Dict[str,str]]:
        params = {
            "key" : self.api_key,
            "cx" : self.cx,
            "q" : query,
            "num" : min(num_results, 10)
        }
        
        try:
            response = requests.get(self.search_url, params = params)
            response.raise_for_status()
            data = response.json()
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

class EmbeddingEngine:
    
    def __init__(self, model_name: str = "distilbert-base-nli-mean-tokens"):

        try:
            self.model = SentenceTransformer(model_name)
            print(f"✓ Model '{model_name}' loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error loading model {model_name}: {e}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
       
        if not texts:
            return np.array([])
            
        return self.model.encode(texts)
    
    def compute_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        return cosine_similarity(query_embedding.reshape(1, -1), document_embeddings)[0]
    
    def filter_by_threshold(self, texts: List[str], similarities: np.ndarray, threshold: float = 0.70) -> List[str]:
      
        return [text for text, sim in zip(texts, similarities) if sim >= threshold]

class FAISSIndexer:
    
    
    def __init__(self):
       
        self.index = None
        self.dimension = None
    
    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
       
        if embeddings is None or embeddings.size == 0:
            raise ValueError("No embeddings provided for FAISS index")
            
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        return self.index
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
       
        if self.index is None:
            raise ValueError("FAISS index not created yet")
            
        return self.index.search(query_embedding, k)
    
    def retrieve_docs(self, query_embedding: np.ndarray, articles: List[Dict[str, str]], k: int = 5) -> List[Dict[str, str]]:
      
        _, indices = self.search(query_embedding, k)
        return [articles[i] for i in indices[0] if i < len(articles)]

# BERT Fake News Detector component
class BERTFakeNewsDetector:
   
    
    def __init__(self, model_name: str = "bert-base-uncased-fake-news", use_local: bool = True):
        self.model_name = model_name
        self.use_local = use_local
        
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.nli_api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
        
        if use_local:
            try:
                print(f"Loading BERT model '{model_name}' locally...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                print(f"✓ BERT model '{model_name}' loaded successfully on {self.device}")
                
                # Create a zero-shot classification pipeline for textual entailment
                try:
                    self.nli_pipeline = pipeline(
                        "zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    print("✓ NLI pipeline loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load NLI pipeline: {e}")
                    self.nli_pipeline = None
                    self.use_local = False
            except Exception as e:
                print(f"Error loading model locally: {e}")
                print("Falling back to Hugging Face API")
                self.use_local = False
    
    def classify_text(self, text: str) -> Dict[str, float]:
       
        if self.use_local:
            # Use local model
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                   max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                probabilities = probabilities.cpu().numpy()[0]
            
            # Assuming the model has binary classification: real (0) and fake (1)
            return {
                "real_probability": float(probabilities[0]),
                "fake_probability": float(probabilities[1])
            }
        else:
            # Use Hugging Face API
            headers = {"Content-Type": "application/json"}
            if HF_API_TOKEN:
                headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
            
            payload = {
                "inputs": text,
                "parameters": {"return_all_scores": True}
            }
            
            try:
                response = requests.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                results = response.json()
                
                # Extract probabilities based on response format
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], list) and len(results[0]) >= 2:
                        # First format: list of lists
                        prob_real = results[0][0].get("score", 0.5)
                        prob_fake = results[0][1].get("score", 0.5)
                    else:
                        # Second format: list of dicts with label/score
                        prob_real = next((item.get("score", 0.5) for item in results if item.get("label", "").lower() in ["real", "true", "0"]), 0.5)
                        prob_fake = next((item.get("score", 0.5) for item in results if item.get("label", "").lower() in ["fake", "false", "1"]), 0.5)
                    
                    return {
                        "real_probability": prob_real,
                        "fake_probability": prob_fake
                    }
                
                # Fallback for unexpected format
                return {"real_probability": 0.5, "fake_probability": 0.5}
                
            except Exception as e:
                print(f"API Error: {e}")
                return {"real_probability": 0.5, "fake_probability": 0.5}
    
    def check_entailment(self, claim: str, evidence: str) -> Dict[str, float]:
        
        if self.use_local and self.nli_pipeline:
            # Use local NLI pipeline
            result = self.nli_pipeline(
                evidence, 
                [claim], 
                hypothesis_template="{}",
                multi_label=False
            )
            
            # Extract scores
            labels = result["labels"]
            scores = result["scores"]
            entailment_score = next((score for label, score in zip(labels, scores) 
                                   if "entailment" in label.lower()), 0.33)
            contradiction_score = next((score for label, score in zip(labels, scores) 
                                      if "contradiction" in label.lower()), 0.33)
            neutral_score = next((score for label, score in zip(labels, scores) 
                                if "neutral" in label.lower()), 0.34)
            
            return {
                "entailment": entailment_score,
                "contradiction": contradiction_score,
                "neutral": neutral_score
            }
        else:
            # Use Hugging Face API for NLI
            headers = {"Content-Type": "application/json"}
            if HF_API_TOKEN:
                headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
            
            payload = {
                "inputs": {
                    "premise": evidence,
                    "hypothesis": claim
                },
                "parameters": {
                    "wait_for_model": True
                }
            }
            
            try:
                response = requests.post(self.nli_api_url, headers=headers, json=payload)
                response.raise_for_status()
                results = response.json()
                
                # Extract scores
                if isinstance(results, list) and len(results) > 0:
                    entailment = next((item["score"] for item in results if item["label"] == "entailment"), 0.33)
                    contradiction = next((item["score"] for item in results if item["label"] == "contradiction"), 0.33)
                    neutral = next((item["score"] for item in results if item["label"] == "neutral"), 0.34)
                    
                    return {
                        "entailment": entailment,
                        "contradiction": contradiction, 
                        "neutral": neutral
                    }
                
                # Fallback
                return {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}
                
            except Exception as e:
                print(f"NLI API Error: {e}")
                return {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}
    
    def analyze_articles(self, claim: str, retrieved_articles: List[Dict[str, str]]) -> Dict[str, Any]:
       
        # Classify the claim itself
        claim_classification = self.classify_text(claim)
        print(f"Claim classification: Real={claim_classification['real_probability']:.2f}, " 
              f"Fake={claim_classification['fake_probability']:.2f}")
        
        # Analyze entailment for each article
        article_analyses = []
        for article in retrieved_articles[:3]:  # Analyze top 3 articles
            title = article.get("title", "")
            snippet = article.get("snippet", "")
            source = article.get("source", "")
            
            # Check entailment between claim and article content
            entailment_scores = self.check_entailment(claim, snippet)
            
            # Classify the article content itself
            article_classification = self.classify_text(snippet)
            
            article_analysis = {
                "title": title,
                "source": source,
                "entailment_scores": entailment_scores,
                "classification": article_classification
            }
            article_analyses.append(article_analysis)
            
            print(f"Article '{title[:30]}...' from {source}:")
            print(f"  - Entailment: {entailment_scores['entailment']:.2f}, "
                  f"Contradiction: {entailment_scores['contradiction']:.2f}, "
                  f"Neutral: {entailment_scores['neutral']:.2f}")
            print(f"  - Classification: Real={article_classification['real_probability']:.2f}, "
                  f"Fake={article_classification['fake_probability']:.2f}")
        
        # Aggregate analysis results
        avg_entailment = sum(a["entailment_scores"]["entailment"] for a in article_analyses) / len(article_analyses) if article_analyses else 0
        avg_contradiction = sum(a["entailment_scores"]["contradiction"] for a in article_analyses) / len(article_analyses) if article_analyses else 0
        
        # Make a verdict based on the analyses
        if claim_classification["fake_probability"] > 0.7:
            verdict = "False"
            confidence = int(min(claim_classification["fake_probability"] * 100, 95))
            reason = "The BERT model strongly indicates this claim is fake news."
        elif claim_classification["fake_probability"] < 0.3 and avg_entailment > avg_contradiction:
            verdict = "True"
            confidence = int(min(claim_classification["real_probability"] * 100, 95))
            reason = "The BERT model indicates this claim is likely true, and evidence supports it."
        elif avg_contradiction > avg_entailment:
            verdict = "False"
            confidence = int(min(avg_contradiction * 100, 90))
            reason = "Evidence contradicts the claim."
        elif avg_entailment > 0.6:
            verdict = "True"
            confidence = int(min(avg_entailment * 100, 90))
            reason = "Evidence supports the claim."
        else:
            verdict = "Unverified"
            confidence = 50
            reason = "There is insufficient evidence to make a strong determination."
        
        return {
            "claim_classification": claim_classification,
            "article_analyses": article_analyses,
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "avg_entailment": avg_entailment,
            "avg_contradiction": avg_contradiction
        }
        
    def generate_report(self, claim: str, analysis_results: Dict[str, Any], 
                        sentiment_analysis: Dict[str, Any]) -> str:
       
        verdict = analysis_results["verdict"]
        confidence = analysis_results["confidence"]
        reason = analysis_results["reason"]
        claim_classification = analysis_results["claim_classification"]
        article_analyses = analysis_results["article_analyses"]
        
        # Build a detailed report
        report = f"""
Fact-Check Report
================

Claim: "{claim}"

Verdict: {verdict}
Confidence: {confidence}%

Summary: {reason}

Analysis:
---------
BERT Model Assessment:
- Real news probability: {claim_classification["real_probability"]:.2f}
- Fake news probability: {claim_classification["fake_probability"]:.2f}

Sentiment Analysis:
- Emotional intensity: {sentiment_analysis["emotional_intensity"]:.2f}
- Dominant emotion: {sentiment_analysis["dominant_emotion"]}
- Emotionally charged: {"Yes" if sentiment_analysis["is_emotionally_charged"] else "No"}

Evidence Analysis:
"""

        # Add evidence analysis
        for i, article in enumerate(article_analyses, 1):
            report += f"""
[Evidence {i}]
Source: {article["source"]}
Title: {article["title"]}
Entailment: {article["entailment_scores"]["entailment"]:.2f}
Contradiction: {article["entailment_scores"]["contradiction"]:.2f}
Neutral: {article["entailment_scores"]["neutral"]:.2f}
"""

        # Add conclusion
        report += f"""
Conclusion:
-----------
{reason} 

The model {"detected" if sentiment_analysis["is_emotionally_charged"] else "did not detect"} emotional manipulation in the claim, 
which {"increases" if sentiment_analysis["is_emotionally_charged"] else "doesn't increase"} the likelihood of misinformation.
"""

        return report
    
    def fact_check(self, claim: str, retrieved_articles: List[Dict[str, str]], 
                  sentiment_analysis: Dict[str, Any]) -> Optional[str]:
       
        try:
            # Analyze the claim against articles
            analysis_results = self.analyze_articles(claim, retrieved_articles)
            
            # Generate the report
            report = self.generate_report(claim, analysis_results, sentiment_analysis)
            
            return report
        except Exception as e:
            print(f"Error in fact-checking process: {e}")
            return None


# Sentiment analysis component
class SentimentAnalyzer:
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
       
        return self.analyzer.polarity_scores(text)
    
    def detect_emotional_manipulation(self, text: str) -> Dict[str, Any]:
       
        scores = self.analyze_sentiment(text)
        
        # Calculate emotional intensity (how extreme the sentiment is)
        emotional_intensity = abs(scores["compound"])
        
        # Determine if the text is emotionally charged
        is_emotionally_charged = emotional_intensity > 0.5
        
        # Identify dominant emotion
        if scores["compound"] >= 0.5:
            dominant_emotion = "strongly positive"
        elif scores["compound"] > 0 and scores["compound"] < 0.5:
            dominant_emotion = "mildly positive"
        elif scores["compound"] > -0.5 and scores["compound"] <= 0:
            dominant_emotion = "mildly negative"
        else:
            dominant_emotion = "strongly negative"
        
        return {
            "sentiment_scores": scores,
            "emotional_intensity": emotional_intensity,
            "is_emotionally_charged": is_emotionally_charged,
            "dominant_emotion": dominant_emotion
        }

# Main Fake News Detection System
class FakeNewsDetectionSystem:
    
    
    def __init__(self, model_name: str = "bert-base-uncased-fake-news", use_local: bool = True):
        
        validate_credentials()
        
        # Initialize components
        self.search_engine = WebSearchEngine(GOOGLE_API_KEY, GOOGLE_CX)
        self.embedding_engine = EmbeddingEngine()
        self.faiss_indexer = FAISSIndexer()
        
        if not HF_API_TOKEN and not use_local:
            print("⚠️ Warning: No Hugging Face API token found, but API usage requested.")
            print("⚠️ You may encounter rate limits or errors. Consider getting an API token.")
            
        self.bert_detector = BERTFakeNewsDetector(model_name=model_name, use_local=use_local)
        self.sentiment_analyzer = SentimentAnalyzer()
        
        
        
        actual_mode = "local models" if self.bert_detector.use_local else "Hugging Face API"
        print(f"✓ Fake News Detection System initialized with BERT model: {model_name}")
        print(f"✓ Using {actual_mode} for inference")
    
    def analyze_claim(self, claim: str) -> Dict[str, Any]:
       
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
        
        # Step 6: Fact-check using BERT model
        print("Fact-checking with BERT model...")
        fact_check_result = self.bert_detector.fact_check(claim, retrieved_articles, sentiment_analysis)
        
        if not fact_check_result:
            return {
                "status": "error",
                "message": "Failed to get fact-checking result from BERT model"
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