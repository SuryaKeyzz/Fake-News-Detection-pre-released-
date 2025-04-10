"""
Fake News Detection System using XLM-RoBERTa with RAG
==================================================
This system combines:
1. Web search for retrieving relevant articles
2. Embedding-based similarity for pre-filtering
3. FAISS for efficient similarity search
4. XLM-RoBERTa for fact-checking and verdict generation
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
from datetime import datetime,UTC
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import spacy
import logging
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import re

nltk.download('punkt_tab')
# New imports for XLM-RoBERTa
import torch
from transformers import AlbertTokenizer, AlbertModel, pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("truthlens.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("SpaCy model loaded successfully")
except:
    logger.warning("SpaCy model not found. Downloading...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load environment variables (still keeping this for other credentials)
load_dotenv()

# API Credentials
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CX = os.getenv('GOOGLE_CX')

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

SPARK_API_PASSWORD = os.getenv('SPARK_API_PASSWORD')

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
    if not SPARK_API_PASSWORD:
        print("Warning: Missing Spark API Password. Using default value.")
    else:
        print(f"✓ Spark API credentials are configured with password: {SPARK_API_PASSWORD[:5]}... (length: {len(SPARK_API_PASSWORD)})")
    
    return True

result_cache = {}

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
        cache_key = f"search_{hash(query)}_{num_results}"
        if cache_key in result_cache:
            logger.info(f"Returning cached search results for: {query}")
            return result_cache[cache_key]
        
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": min(num_results, 10)  # Max 10 for free tier
        }
        
        try:
            logger.info(f"Searching for: {query}")
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
            
            # Cache the results
            result_cache[cache_key] = articles
            logger.info(f"Found {len(articles)} articles for: {query}")
            return articles
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error in web search: {e}")
            logger.debug("Full API Response: " + response.text)
            return []
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []
    
    def fetch_article_content(self, url: str) -> Optional[str]:
        """Fetch and extract the main content from an article
        
        Args:
            url: URL of the article
            
        Returns:
            Extracted article content or None if failed
        """
        # Check cache first
        cache_key = f"content_{hash(url)}"
        if cache_key in result_cache:
            return result_cache[cache_key]
        
        try:
            logger.info(f"Fetching content from: {url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Use BeautifulSoup for more sophisticated content extraction
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove non-content elements
            for element in soup(["script", "style", "header", "footer", "nav"]):
                element.decompose()
            
            # Extract paragraphs with reasonable length
            paragraphs = []
            for p in soup.find_all('p'):
                text = p.get_text().strip()
                if len(text) > 40:
                    paragraphs.append(text)
            
            if not paragraphs:
                # Fallback to extracting text from the body
                body_text = soup.body.get_text() if soup.body else ""
                content = " ".join(body_text.split())
            else:
                content = " ".join(paragraphs)
            
            # Truncate to a reasonable length
            content = content[:5000]
            
            # Cache the result
            result_cache[cache_key] = content
            return content
            
        except Exception as e:
            logger.error(f"Error fetching article content: {e}")
            return None


# URL handling component
class URLHandler:
    """Component for handling URL inputs and extracting article content"""
    
    def __init__(self):
        """Initialize the URL handler"""
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def is_url(self, text: str) -> bool:
        """Check if the input text is a URL
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if text is a URL
        """
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ipv4
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(text))
    
    def extract_article(self, url: str) -> Dict[str, Any]:
        """Extract article content and metadata from a URL with improved handling for major news sites
        
        Args:
            url: URL to extract from
            
        Returns:
            Dictionary with article content and metadata
        """
        try:
            logger.info(f"Extracting article from URL: {url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            import trafilatura
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Identify the news source from the URL
            news_source = self._identify_news_source(url)
            extracted_text = None
            
            # Site-specific content extraction
            if news_source == 'cnn':
                article_body = soup.find('div', class_='article__content')
                if article_body:
                    paragraphs = article_body.find_all('p')
                    extracted_text = ' '.join([p.get_text().strip() for p in paragraphs])
                    
            elif news_source == 'bbc':
                article_body = soup.find('article')
                if article_body:
                    paragraphs = article_body.find_all('p')
                    extracted_text = ' '.join([p.get_text().strip() for p in paragraphs])
                
            elif news_source == 'nytimes':
                article_sections = soup.find_all('section', class_='meteredContent')
                if article_sections:
                    paragraphs = []
                    for section in article_sections:
                        paragraphs.extend(section.find_all('p'))
                    extracted_text = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # If site-specific extraction failed or it's not a recognized site, try trafilatura
            if not extracted_text:
                extracted_text = trafilatura.extract(response.text)
            
            # Fallback to BeautifulSoup if trafilatura fails
            if not extracted_text:
                # Remove non-content elements
                for element in soup(["script", "style", "header", "footer", "nav"]):
                    element.decompose()
                
                # Extract paragraphs with reasonable length
                paragraphs = []
                for p in soup.find_all('p'):
                    text = p.get_text().strip()
                    if len(text) > 40:
                        paragraphs.append(text)
                
                if not paragraphs:
                    # Fallback to extracting text from the body
                    body_text = soup.body.get_text() if soup.body else ""
                    extracted_text = " ".join(body_text.split())
                else:
                    extracted_text = " ".join(paragraphs)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url, news_source)
            
            # Build result
            result = {
                "content": extracted_text[:10000],  # Limit content length
                "url": url,
                "title": metadata.get("title", "Unknown Title"),
                "author": metadata.get("author", "Unknown Author"),
                "published_date": metadata.get("published_date", "Unknown Date"),
                "source_domain": self._extract_domain(url),
                "is_url_input": True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting article from URL: {e}")
            return {
                "content": "",
                "url": url,
                "title": "Failed to extract",
                "author": "Unknown",
                "published_date": "Unknown",
                "source_domain": self._extract_domain(url),
                "error": str(e),
                "is_url_input": True
            }

    def _identify_news_source(self, url: str) -> str:
        """Identify the news source from a URL
        
        Args:
            url: The URL to analyze
            
        Returns:
            String identifying the news source
        """
        url_lower = url.lower()
        
        if 'cnn.com' in url_lower:
            return 'cnn'
        elif 'bbc.com' in url_lower or 'bbc.co.uk' in url_lower:
            return 'bbc'
        elif 'nytimes.com' in url_lower:
            return 'nytimes'
        elif 'washingtonpost.com' in url_lower:
            return 'washingtonpost'
        elif 'theguardian.com' in url_lower:
            return 'guardian'
        elif 'reuters.com' in url_lower:
            return 'reuters'
        elif 'apnews.com' in url_lower:
            return 'ap'
        elif 'foxnews.com' in url_lower:
            return 'fox'
        elif 'nbcnews.com' in url_lower:
            return 'nbc'
        elif 'abcnews.go.com' in url_lower:
            return 'abc'
        elif 'cbsnews.com' in url_lower:
            return 'cbs'
        else:
            return 'unknown'
    
   
    def _extract_metadata(self, soup, url: str, news_source: str = 'unknown') -> Dict[str, str]:
        """Extract metadata from HTML with improved handling for major news sites
        
        Args:
            soup: BeautifulSoup object
            url: Original URL
            news_source: Identified news source
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            "title": "Unknown Title",
            "author": "Unknown Author",
            "published_date": "Unknown Date"
        }
        
        # Site-specific title extraction
        if news_source == 'cnn':
            headline_tag = soup.find('h1', class_='headline__text')
            if headline_tag:
                metadata["title"] = headline_tag.get_text().strip()
        
        elif news_source == 'bbc':
            headline_tag = soup.find('h1', id='main-heading')
            if headline_tag:
                metadata["title"] = headline_tag.get_text().strip()
        
        elif news_source == 'nytimes':
            headline_tag = soup.find('h1', class_='css-1l24qy5')
            if headline_tag:
                metadata["title"] = headline_tag.get_text().strip()
        
        elif news_source == 'washingtonpost':
            headline_tag = soup.find('h1', class_='font-md')
            if headline_tag:
                metadata["title"] = headline_tag.get_text().strip()
        
        elif news_source == 'guardian':
            headline_tag = soup.find('h1', class_='dcr-125vfar')
            if headline_tag:
                metadata["title"] = headline_tag.get_text().strip()
        
        # Try OG tags for title (many sites use these)
        if metadata["title"] == "Unknown Title":
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                metadata["title"] = og_title.get('content').strip()
        
        # Fallback to standard title tag
        if metadata["title"] == "Unknown Title" and soup.title:
            metadata["title"] = soup.title.get_text().strip()
        
        # Site-specific author extraction
        if news_source == 'cnn':
            cnn_authors = []
            byline_tags = soup.find_all('div', class_='byline__names')
            for byline in byline_tags:
                author_links = byline.find_all('a', class_='byline__name')
                for author_link in author_links:
                    cnn_authors.append(author_link.get_text().strip())
            
            if cnn_authors:
                metadata["author"] = ", ".join(cnn_authors)
        
        elif news_source == 'bbc':
            byline_tag = soup.find('div', class_='ssrcss-68pt20-Text-TextContributorName')
            if byline_tag:
                metadata["author"] = byline_tag.get_text().strip()
        
        elif news_source == 'nytimes':
            byline_tags = soup.find_all('span', class_='css-1baulvz')
            if byline_tags:
                authors = [tag.get_text().strip() for tag in byline_tags if tag.get_text().strip()]
                if authors:
                    metadata["author"] = ", ".join(authors)
                    
        elif news_source == 'washingtonpost':
            author_tag = soup.find('a', class_='css-nowh2b')
            if author_tag:
                metadata["author"] = author_tag.get_text().strip()
        
        # Try standard meta tags for author
        if metadata["author"] == "Unknown Author":
            author_meta_tags = [
                soup.find('meta', attrs={'name': 'author'}),
                soup.find('meta', attrs={'property': 'article:author'}),
                soup.find('meta', attrs={'property': 'og:author'})
            ]
            
            for tag in author_meta_tags:
                if tag and tag.get('content'):
                    metadata["author"] = tag.get('content').strip()
                    break
        
        # Try common author class/ID patterns
        if metadata["author"] == "Unknown Author":
            author_patterns = [
                soup.find(class_=['author', 'byline', 'byline__name', 'writer', 'creator']),
                soup.find(id=['author', 'byline']),
                soup.find('a', class_=['author', 'byline']),
                soup.find('span', class_=['author', 'byline'])
            ]
            
            for pattern in author_patterns:
                if pattern:
                    author_text = pattern.get_text().strip()
                    if author_text and len(author_text) < 100:  # Avoid getting paragraphs
                        metadata["author"] = author_text
                        break
        
        # Site-specific date extraction
        if news_source == 'cnn':
            date_tag = soup.find('div', class_='timestamp')
            if date_tag:
                metadata["published_date"] = date_tag.get_text().strip()
        
        elif news_source == 'bbc':
            time_tag = soup.find('time')
            if time_tag and time_tag.has_attr('datetime'):
                metadata["published_date"] = time_tag['datetime']
        
        elif news_source == 'nytimes':
            time_tag = soup.find('time')
            if time_tag and time_tag.has_attr('datetime'):
                metadata["published_date"] = time_tag['datetime']
        
        # Try standard meta tags for date
        if metadata["published_date"] == "Unknown Date":
            date_meta_tags = [
                soup.find('meta', attrs={'property': 'article:published_time'}),
                soup.find('meta', attrs={'name': 'publication_date'}),
                soup.find('meta', attrs={'name': 'date'}),
                soup.find('meta', attrs={'property': 'og:published_time'}),
                soup.find('meta', attrs={'itemprop': 'datePublished'})
            ]
            
            for tag in date_meta_tags:
                if tag and tag.get('content'):
                    metadata["published_date"] = tag.get('content').strip()
                    break
        
        # Try common date class/ID patterns
        if metadata["published_date"] == "Unknown Date":
            date_patterns = [
                soup.find(class_=['date', 'published', 'timestamp', 'article-date']),
                soup.find(id=['date', 'published-date', 'publication-date']),
                soup.find('time')
            ]
            
            for pattern in date_patterns:
                if pattern:
                    if pattern.get('datetime'):
                        metadata["published_date"] = pattern.get('datetime')
                    else:
                        date_text = pattern.get_text().strip()
                        if date_text and len(date_text) < 100:  # Avoid getting paragraphs
                            metadata["published_date"] = date_text
                    break
        
        return metadata
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL
        
        Args:
            url: URL
            
        Returns:
            Domain name
        """
        try:
            from urllib.parse import urlparse
            parsed_uri = urlparse(url)
            domain = '{uri.netloc}'.format(uri=parsed_uri)
            return domain
        except:
            return "unknown_domain"



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
        filtered_texts = []
        for text_dict, sim in zip(texts, similarities):
            if sim >= threshold:
                text_dict_copy = text_dict.copy()
                text_dict_copy["similarity"] = float(sim)
                filtered_texts.append(text_dict_copy)
        
        # Sort by similarity score (descending)
        filtered_texts.sort(key=lambda x: x["similarity"], reverse=True)
        return filtered_texts
    
    
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
        distances, indices = self.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(articles):
                article = articles[idx].copy()
                # Convert L2 distance to a similarity score (smaller distance = higher similarity)
                max_distance = 10.0  # Arbitrary max distance for normalization
                similarity = max(0, 1.0 - distances[0][i] / max_distance)
                article["similarity"] = float(similarity)
                results.append(article)
        
        return results


# Named Entity Recognition component
class EntityExtractor:
    """Component for extracting named entities from text"""
    
    def __init__(self):
        """Initialize the entity extractor"""
        self.nlp = nlp
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entity types and their mentions
        """
        doc = self.nlp(text)
        
        entities = {}
        for ent in doc.ents:
            entity_type = ent.label_
            entity_text = ent.text
            
            if entity_type not in entities:
                entities[entity_type] = []
            
            if entity_text not in entities[entity_type]:
                entities[entity_type].append(entity_text)
        
        return entities
    
    def get_key_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract and filter for key entity types
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of key entity types and their mentions
        """
        all_entities = self.extract_entities(text)
        
        # Filter for key entity types
        key_entity_types = ["PERSON", "ORG", "GPE", "DATE", "PERCENT", "MONEY", "QUANTITY"]
        key_entities = {k: v for k, v in all_entities.items() if k in key_entity_types}
        
        return key_entities



# Knowledge Graph Component (simplified version)
class KnowledgeGraph:
    """Component for entity verification using a knowledge graph"""
    
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """Initialize the knowledge graph component
        
        Args:
            uri: Neo4j URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self.available = False
        
        # Only attempt to connect if credentials are provided
        if self.uri and self.user and self.password:
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                self.available = True
                logger.info("Neo4j connection established successfully")
            except Exception as e:
                logger.warning(f"Neo4j connection failed: {e}")
                self.driver = None
        else:
            logger.warning("Neo4j credentials not provided, knowledge graph functionality will be limited")
            self.driver = None
    
    def verify_entity_relationships(self, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Verify relationships between entities
        
        Args:
            entities: Dictionary of entity types and their mentions
            
        Returns:
            Dictionary with verification results
        """
        if not self.available or not self.driver:
            return {"status": "unavailable", "message": "Knowledge graph not available"}
        
        verification_results = {"status": "success", "verified": [], "unverified": []}
        
        try:
            with self.driver.session() as session:
                # For each entity, check if it exists in the graph
                for entity_type, entity_mentions in entities.items():
                    for entity in entity_mentions:
                        # Cypher query to find the entity
                        query = f"""
                        MATCH (e) 
                        WHERE e.name = $entity_name OR e.alias = $entity_name
                        RETURN e
                        """
                        
                        result = session.run(query, entity_name=entity)
                        records = list(result)
                        
                        if records:
                            verification_results["verified"].append({
                                "entity": entity,
                                "type": entity_type,
                                "found_in_kg": True
                            })
                        else:
                            verification_results["unverified"].append({
                                "entity": entity,
                                "type": entity_type,
                                "found_in_kg": False
                            })
                
                return verification_results
                
        except Exception as e:
            logger.error(f"Error verifying entities: {e}")
            return {"status": "error", "message": str(e)}
    
    def close(self):
        """Close the Neo4j connection"""
        if self.available and self.driver:
            self.driver.close()


# Spark LLM component (replacing XLM-RoBERTa)
class SparkLLMFactChecker:
    """Component for fact-checking using Spark LLM API"""
    
    def __init__(self, api_password: str = None):
        """Initialize with Spark API password
        
        Args:
            api_password: Spark API Password (Format: "APIPassword" from console)
        """
        # Load from environment if not provided
        self.api_password = api_password or os.getenv('SPARK_API_PASSWORD', "cmdYpTbmpTgclKqTfhCE:ZOkmMTpDiqULISaijjgd")
        
        # HTTP API URL from the documentation
        self.api_url = "https://spark-api-open.xf-yun.com/v1/chat/completions"
        
        print(f"✓ Spark LLM API configured with password: {self.api_password[:5]}... (length: {len(self.api_password)})")
    
    def preprocess_evidence(self, retrieved_articles: List[Dict[str, str]]) -> str:
        """Preprocess evidence from retrieved articles
        
        Args:
            retrieved_articles: List of retrieved articles
            
        Returns:
            Preprocessed evidence text
        """
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
        return evidence_text
    
    def build_fact_check_prompt(self, claim: str, evidence_text: str) -> str:
        """Build a prompt for fact-checking that avoids content filter triggers
        
        Args:
            claim: The claim to fact-check
            evidence_text: Preprocessed evidence text
            
        Returns:
            Prompt string
        """
        # Build the prompt with neutral examples to avoid content filter
        prompt = f"""
Please help me evaluate the accuracy of the following claim using the provided evidence articles.

Claim to evaluate: "{claim}"

Evidence from reputable sources:
{evidence_text}

Please follow this structured analysis approach:
1) Identify the main factual assertions in the claim
2) Compare these assertions with the evidence provided
3) Note areas where the evidence supports or contradicts the claim
4) Assess if there is sufficient evidence to make a determination
5) Provide a final assessment with confidence level

Format your response as follows:
Verdict: [True/Partially True/False/Unverified]
Confidence: [percentage]
Reasoning:
- Step 1: [Your analysis of factual assertions]
- Step 2: [Your comparison with evidence]
- Step 3: [Support/contradiction analysis]
- Step 4: [Evidence sufficiency assessment]
- Step 5: [Final assessment]

Explanation: [Summary of why the claim is true, partially true, false, or unverified]

Sources: [Which articles support or contradict the claim]
"""
        return prompt
    
    def analyze_claim_evidence_alignment(self, claim: str, evidence: str) -> Dict[str, Any]:
        """Analyze alignment between claim and evidence
        
        Args:
            claim: The claim to analyze
            evidence: Evidence text
            
        Returns:
            Dictionary with alignment analysis
        """
        # Extract entities from claim and evidence using simple method
        def extract_entities(text: str) -> List[str]:
            import re
            from collections import Counter
            
            # Extract capitalized words that might be entities
            words = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
            
            # Count occurrences
            word_counts = Counter(words)
            
            # Return the most common entities (occurring more than once)
            return [word for word, count in word_counts.most_common() if count > 1]
        
        # Extract entities from claim and evidence
        claim_entities = set(extract_entities(claim))
        evidence_entities = set(extract_entities(evidence))
        
        # Check entity overlap
        common_entities = claim_entities.intersection(evidence_entities)
        entity_coverage = len(common_entities) / len(claim_entities) if claim_entities else 0
        
        # Get claim and evidence embeddings for semantic similarity
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Lightweight model for this purpose
        claim_embedding = embedding_model.encode([claim])[0]
        evidence_embedding = embedding_model.encode([evidence])[0]
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            claim_embedding.reshape(1, -1), 
            evidence_embedding.reshape(1, -1)
        )[0][0]
        
        # Determine confidence based on alignment factors
        confidence = (entity_coverage * 0.5) + (similarity * 0.5)
        confidence = min(max(confidence, 0.1), 0.95)  # Ensure reasonable bounds
        
        return {
            "entity_overlap": {
                "claim_entities": list(claim_entities),
                "evidence_entities": list(evidence_entities),
                "common_entities": list(common_entities),
                "entity_coverage": entity_coverage
            },
            "semantic_similarity": float(similarity),
            "confidence": float(confidence)
        }
    
    def fact_check(self, claim: str, retrieved_articles: List[Dict[str, str]], entities: Dict[str, List[str]] = None) -> Optional[str]:
        """Perform fact-checking using Spark LLM
        
        Args:
            claim: The claim to fact-check
            retrieved_articles: List of retrieved articles
            entities: Optional dict of extracted entities
            
        Returns:
            Fact-checking result
        """
        # Check cache first
        cache_key = f"fact_check_{hash(claim)}_{hash(str(retrieved_articles))}"
        if cache_key in result_cache:
            logger.info(f"Returning cached fact-check result for: {claim}")
            return result_cache[cache_key]
        
        # Preprocess evidence
        evidence = self.preprocess_evidence(retrieved_articles)
        
        # Build the prompt
        prompt = self.build_fact_check_prompt(claim, evidence)
        
        # Make API request to Spark LLM
        try:
            logger.info("Sending request to Spark LLM API...")
            
            # Using the format from the documentation
            payload = {
                "model": "4.0Ultra",  # Using Ultra version
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional fact-checker. Your task is to analyze claims and evidence objectively. Provide clear verdicts with confidence levels and detailed reasoning."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,     # Moderate temperature for factual responses
                "top_k": 4,             # From documentation
                "max_tokens": 1500,     # Sufficient for detailed analysis
                "stream": False         # Non-streaming mode
            }
            
            # Simple Bearer token authentication as per docs
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_password}"
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            logger.info(f"API Response Status: {response.status_code}")
            
            response.raise_for_status()
            result = response.json()
            
            # Extract content based on API response structure
            if result.get("code") == 0:  # Success code per documentation
                content = result["choices"][0]["message"]["content"]
                
                # Cache the result
                result_cache[cache_key] = content
                return content
            else:
                error_message = result.get("message", "Unknown error")
                error_code = result.get("code", "Unknown code")
                logger.error(f"Spark API Error {error_code}: {error_message}")
                
                # If API fails, fall back to using alignment analysis for a basic response
                alignment = self.analyze_claim_evidence_alignment(claim, evidence)
                similarity = alignment["semantic_similarity"]
                confidence = int(alignment["confidence"] * 100)
                
                # Determine verdict based on similarity
                if similarity > 0.8:
                    verdict = "True"
                elif similarity > 0.5:
                    verdict = "Partially True"
                elif similarity > 0.3:
                    verdict = "Unverified"
                else:
                    verdict = "False"
                
                # Create a fallback response
                fallback_response = f"""
Verdict: {verdict}
Confidence: {confidence}%
Reasoning:
- Step 1: Entity analysis
  Found entities in claim: {', '.join(alignment["entity_overlap"]["claim_entities"]) or "None"}
  Found entities in evidence: {', '.join(alignment["entity_overlap"]["evidence_entities"][:5]) or "None"}...
  Entity overlap: {alignment["entity_overlap"]["entity_coverage"]:.2f}

- Step 2: Evidence comparison
  Semantic similarity between claim and evidence: {similarity:.2f}
  
- Step 3: Contradiction/confirmation analysis
  The evidence {"supports" if similarity > 0.7 else "partially supports" if similarity > 0.4 else "contradicts"} the claim.
  
- Step 4: Evidence sufficiency
  {"Sufficient evidence exists to make a determination." if len(retrieved_articles) >= 2 else "Limited evidence available, reducing confidence."}

Explanation: {"The claim is well-supported by the evidence, with strong entity overlap and semantic alignment." if verdict == "True" else
              "The claim has partial support in the evidence but contains some unverified elements." if verdict == "Partially True" else
              "The claim contradicts available evidence or lacks sufficient support in reliable sources."}

Sources: {', '.join([article.get("source", "Unknown Source") for article in retrieved_articles[:3]])}
"""
                # Cache the fallback result
                result_cache[cache_key] = fallback_response
                return fallback_response
                
        except Exception as e:
            logger.error(f"Spark API Error: {str(e)}")
            if 'response' in locals():
                logger.error(f"Response text: {response.text}")
            return None
            
    def test_api_connection(self) -> bool:
        """Test the API connection with a simple, safe prompt
        
        Returns:
            True if connection successful, False otherwise
        """
        payload = {
            "model": "4.0Ultra",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, can you tell me about the weather today?"
                }
            ],
            "max_tokens": 50,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_password}"
        }
        
        try:
            logger.info("Testing Spark API connection...")
            response = requests.post(self.api_url, headers=headers, json=payload)
            logger.info(f"Test Response Status: {response.status_code}")
            
            result = response.json()
            if result.get("code") == 0:
                logger.info("✓ Spark API connection successful")
                return True
            else:
                logger.warning(f"× Spark API test failed: {result.get('message', 'Unknown error')}")
                logger.warning(f"Error code: {result.get('code', 'Unknown')}")
                return False
                
        except Exception as e:
            logger.error(f"Spark API test error: {str(e)}")
            return False
        
    
    def detect_title_content_contradiction(self, title: str, content: str) -> Dict[str, Any]:
        """
        Detect potential contradictions between title and content
        
        Args:
            title: Article title
            content: Article content
        
        Returns:
            Dictionary with contradiction analysis details
        """
        # Truncate content to manageable length
        content = content[:2000]  # Limit to prevent excessive API calls
        
        prompt = f"""
        Analyze the following news article for potential contradictions or misleading implications:
        
        Title: "{title}"
        Content: "{content}"
        
        Please evaluate the following aspects:
        1. Do the title and content match in meaning?
        2. Is the title potentially misleading compared to the actual content?
        3. Are there significant discrepancies between the title's implications and the article's substance?
        
        Response format:
        - Contradiction Severity: [Float between 0.0 and 1.0]
        0.0 = No contradiction
        1.0 = Extreme contradiction
        
        - Contradiction Type: 
        [Select from: "No Contradiction", "Slight Mismatch", "Moderate Misleading", "Significant Contradiction"]
        
        - Explanation: [Detailed reasoning for the contradiction assessment]
        """
        
        try:
            # Use Spark LLM to analyze contradiction (similar to fact_check method)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_password}"
            }
            
            payload = {
                "model": "4.0Ultra",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert fact-checker analyzing potential contradictions in news articles."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # Parse the response
            content = result['choices'][0]['message']['content']
            
            # Extract key information using regex or parsing
            contradiction_severity_match = re.search(r'Contradiction Severity: (\d+\.\d+)', content)
            contradiction_type_match = re.search(r'Contradiction Type: (.*?)(\n|$)', content)
            explanation_match = re.search(r'Explanation: (.*)', content, re.DOTALL)
            
            contradiction_severity = float(contradiction_severity_match.group(1)) if contradiction_severity_match else 0.0
            contradiction_type = contradiction_type_match.group(1).strip() if contradiction_type_match else "Unknown"
            explanation = explanation_match.group(1).strip() if explanation_match else "No detailed explanation available"
            
            return {
                "contradiction_severity": contradiction_severity,
                "contradiction_type": contradiction_type,
                "explanation": explanation,
                "is_misleading": contradiction_severity > 0.5
            }
        
        except Exception as e:
            logger.error(f"Error detecting title-content contradiction: {e}")
            return {
                "contradiction_severity": 0.0,
                "contradiction_type": "Analysis Failed",
                "explanation": f"Could not analyze contradiction: {str(e)}",
                "is_misleading": False
            }
    
       

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


# Source credibility analysis component
class CredibilityAnalyzer:
    """Component for analyzing the credibility of news sources and authors"""
    
    def __init__(self):
        """Initialize the credibility analyzer"""
        # Load or define known credible sources (in a real system, this would be a comprehensive database)
        self.credible_news_domains = {
            'reuters.com': 0.95,
            'apnews.com': 0.95,
            'bbc.com': 0.92,
            'bbc.co.uk': 0.92,
            'nytimes.com': 0.90,
            'wsj.com': 0.90,
            'washingtonpost.com': 0.88,
            'economist.com': 0.90,
            'npr.org': 0.88,
            'theguardian.com': 0.85,
            'bloomberg.com': 0.87,
            'cnn.com': 0.80,
            'nbcnews.com': 0.80,
            'abcnews.go.com': 0.80,
            'cbsnews.com': 0.80,
            'politico.com': 0.82,
            'thehill.com': 0.80,
            'usatoday.com': 0.78,
            'latimes.com': 0.82,
            'time.com': 0.85,
            'newsweek.com': 0.75,
            'theatlantic.com': 0.85,
            'newyorker.com': 0.87,
            'huffpost.com': 0.70,
            'vox.com': 0.75,
            'slate.com': 0.72,
            'foxnews.com': 0.70,
            'msnbc.com': 0.70,
        }
    
    def analyze_source_credibility(self, domain: str) -> Dict[str, Any]:
        """Analyze the credibility of a news source by its domain
        
        Args:
            domain: Domain name of the source
            
        Returns:
            Dictionary with credibility metrics
        """
        domain = domain.lower()  # Normalize domain
        
        # Check if domain is in our database of known sources
        base_score = self.credible_news_domains.get(domain)
        
        # If not in database, we do some heuristic analysis
        if base_score is None:
            # Check for educational or government domains
            if domain.endswith('.edu'):
                base_score = 0.85
            elif domain.endswith('.gov'):
                base_score = 0.90
            elif domain.endswith('.org'):
                base_score = 0.75  # .org sites can vary widely in credibility
            else:
                # Default score for unknown domains
                base_score = 0.50
        
        credibility_level = "UNKNOWN"
        explanation = ""
        
        if base_score >= 0.85:
            credibility_level = "HIGH"
            explanation = f"{domain} is recognized as a highly credible news source with established fact-checking processes."
        elif base_score >= 0.70:
            credibility_level = "MEDIUM"
            explanation = f"{domain} is a generally reliable source but may have some bias in reporting."
        elif base_score >= 0.50:
            credibility_level = "LOW_MEDIUM"
            explanation = f"{domain} has moderate credibility but may have significant bias or occasional accuracy issues."
        else:
            credibility_level = "LOW"
            explanation = f"{domain} has limited established credibility or is not a recognized mainstream news source."
            
        return {
            "domain": domain,
            "credibility_score": float(base_score),
            "credibility_level": credibility_level,
            "explanation": explanation
        }
    
    def analyze_author_credibility(self, author: str) -> Dict[str, Any]:
        """Analyze the credibility of an author (basic implementation)
        
        Args:
            author: Author name
            
        Returns:
            Dictionary with author credibility assessment
        """
        # In a real system, this would check against a database of known authors
        # Here we're implementing a basic version that looks for "staff" or unknown authors
        
        if not author or author.lower() in ["unknown", "unknown author", ""]:
            return {
                "author": author,
                "credibility_factor": 0.0,
                "explanation": "Unknown author reduces credibility. Articles without clear attribution are less verifiable."
            }
        
        if any(term in author.lower() for term in ["staff", "editor", "reporter", "correspondent"]):
            return {
                "author": author,
                "credibility_factor": 0.6,
                "explanation": "Generic staff attribution. While from an organization, specific author would increase credibility."
            }
        
        # Basic check for names that look legitimate (first and last name)
        name_parts = [part for part in author.split() if len(part) > 1 and part[0].isupper()]
        if len(name_parts) >= 2:
            return {
                "author": author,
                "credibility_factor": 0.8,
                "explanation": "Named author increases credibility as it provides accountability and verification possibilities."
            }
        
        return {
            "author": author,
            "credibility_factor": 0.5,
            "explanation": "Author information provides some accountability, but may need verification."
        }
        
    def analyze_date_credibility(self, date_str: str, claim_text: str) -> Dict[str, Any]:
        """Analyze the credibility impact of the publication date
        
        Args:
            date_str: Publication date string
            claim_text: Text of the claim being analyzed
            
        Returns:
            Dictionary with date credibility assessment
        """
        if not date_str or date_str.lower() in ["unknown", "unknown date", ""]:
            return {
                "date": "Unknown",
                "credibility_factor": 0.0,
                "recency": "Unknown",
                "explanation": "Missing publication date reduces credibility. Cannot verify timeliness or context."
            }
        
        try:
            # Try to parse date - handle multiple formats
            import dateutil.parser
            from datetime import datetime, timedelta
            
            pub_date = dateutil.parser.parse(date_str)
            current_date = datetime.now()
            
            # Calculate days difference
            days_diff = (current_date - pub_date).days
            
            # Extract years from the claim (important for historical claims)
            year_pattern = r'\b(19\d{2}|20\d{2})\b'
            years_in_claim = re.findall(year_pattern, claim_text)
            
            # If claim mentions specific years and article is from that time, it may be more credible
            if years_in_claim:
                years_mentioned = [int(year) for year in years_in_claim]
                pub_year = pub_date.year
                
                if pub_year in years_mentioned or any(abs(pub_year - year) <= 1 for year in years_mentioned):
                    return {
                        "date": pub_date.strftime("%Y-%m-%d"),
                        "days_ago": days_diff,
                        "credibility_factor": 0.9,
                        "recency": "Contemporaneous with events",
                        "explanation": f"Publication date ({pub_date.strftime('%Y-%m-%d')}) is contemporaneous with events mentioned in the claim, increasing credibility."
                    }
            
            # Assess recency
            if days_diff < 7:
                recency = "Very recent"
                factor = 0.9  # Very recent news may lack full fact-checking
            elif days_diff < 30:
                recency = "Recent"
                factor = 0.95  # Recent but enough time for fact-checking
            elif days_diff < 365:
                recency = "Within past year"
                factor = 0.85  # Relevant but may miss recent developments
            elif days_diff < 365 * 2:
                recency = "1-2 years old"
                factor = 0.7
            else:
                recency = "Older than 2 years"
                factor = 0.5  # May be outdated for current events
                
                # If claim is about current events but article is old, reduce more
                current_event_indicators = ["today", "yesterday", "this week", "this month", "this year", "currently", "now"]
                if any(indicator in claim_text.lower() for indicator in current_event_indicators):
                    factor = 0.3
                    recency = "Outdated for current claim"
            
            return {
                "date": pub_date.strftime("%Y-%m-%d"),
                "days_ago": days_diff,
                "credibility_factor": factor,
                "recency": recency,
                "explanation": f"Publication date: {pub_date.strftime('%Y-%m-%d')} ({recency}). {days_diff} days ago."
            }
            
        except Exception as e:
            logger.error(f"Error parsing date: {e}")
            return {
                "date": date_str,
                "credibility_factor": 0.3,
                "recency": "Unknown format",
                "explanation": f"Unparseable publication date '{date_str}' reduces credibility. Cannot verify timeliness."
            }
    
    def assess_overall_source_credibility(self, domain: str, author: str, date_str: str, claim_text: str) -> Dict[str, Any]:
        """Assess the overall credibility of a source based on multiple factors
        
        Args:
            domain: Source domain
            author: Author name
            date_str: Publication date string
            claim_text: Text of the claim
            
        Returns:
            Dictionary with overall credibility assessment
        """
        # Get individual credibility factors
        source_cred = self.analyze_source_credibility(domain)
        author_cred = self.analyze_author_credibility(author)
        date_cred = self.analyze_date_credibility(date_str, claim_text)
        
        # Calculate weighted overall score
        weights = {
            "source": 0.5,  # Source/domain is most important
            "author": 0.3,  # Author attribution is next
            "date": 0.2     # Date is still important but less weighted
        }
        
        overall_score = (
            source_cred["credibility_score"] * weights["source"] +
            author_cred["credibility_factor"] * weights["author"] +
            date_cred["credibility_factor"] * weights["date"]
        )
        
        # Determine overall credibility level
        if overall_score >= 0.85:
            level = "HIGH"
            explanation = "This is a highly credible source. The information likely underwent editorial review and fact-checking."
        elif overall_score >= 0.70:
            level = "MEDIUM_HIGH"
            explanation = "This is a generally credible source with some minor concerns."
        elif overall_score >= 0.50:
            level = "MEDIUM"
            explanation = "This source has moderate credibility. Verify with additional sources if possible."
        elif overall_score >= 0.30:
            level = "LOW_MEDIUM"
            explanation = "This source has questionable credibility. Treat information with caution."
        else:
            level = "LOW"
            explanation = "This source has low credibility. Information should be verified with more reliable sources."
        
        # Build detailed explanation
        details = [
            f"Source: {source_cred['explanation']}",
            f"Author: {author_cred['explanation']}",
            f"Date: {date_cred['explanation']}"
        ]
        
        return {
            "overall_credibility_score": float(overall_score),
            "credibility_level": level,
            "explanation": explanation,
            "details": details,
            "factors": {
                "source": source_cred,
                "author": author_cred,
                "date": date_cred
            }
        }
    
    def calculate_trust_lens_score(self, source_credibility, factual_match, tone_neutrality, source_transparency):
        """
        Calculate a comprehensive trust score
        
        Weights can be adjusted based on empirical testing
        """
        weights = {
            'source_credibility': 0.4,
            'factual_match': 0.3,
            'tone_neutrality': 0.2,
            'source_transparency': 0.1
        }
        
        trust_score = (
            source_credibility * weights['source_credibility'] +
            factual_match * weights['factual_match'] +
            tone_neutrality * weights['tone_neutrality'] +
            source_transparency * weights['source_transparency']
        )
        
        return min(max(trust_score, 0), 1)  # Ensure score between 0-1




# Main Fake News Detection System
class FakeNewsDetectionSystem:
    """Complete system for fake news detection using XLM-RoBERTa and RAG"""
    
    def __init__(self):
        """Initialize the fake news detection system"""
        validate_credentials()
        
        self.search_engine = WebSearchEngine(GOOGLE_API_KEY, GOOGLE_CX)
        self.embedding_engine = EmbeddingEngine()
        self.faiss_indexer = FAISSIndexer()
        self.spark_llm = SparkLLMFactChecker(SPARK_API_PASSWORD)  # Replace XLM-RoBERTa with Spark LLM
        self.sentiment_analyzer = SentimentAnalyzer()
        self.entity_extractor = EntityExtractor()
        self.url_handler = URLHandler()
        self.credibility_analyzer = CredibilityAnalyzer()

        # Test Spark API connection
        self.spark_llm.test_api_connection()

        # Initialize knowledge graph if credentials are available
        if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
            self.knowledge_graph = KnowledgeGraph()
        else:
            self.knowledge_graph = None
            logger.warning("Knowledge Graph not initialized due to missing credentials")


    def analyze_claim(self, claim: str, use_rag: bool = True, use_kg: bool = True) -> Dict[str, Any]:
        """Analyze a claim for fake news detection
        
        Args:
            claim: The claim to analyze or a URL
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing claim or URL: {claim}")
        start_time = datetime.now()
        
        # Check cache first
        cache_key = f"analysis_{hash(claim)}_{use_rag}_{use_kg}"
        if cache_key in result_cache:
            logger.info(f"Returning cached analysis for: {claim}")
            return result_cache[cache_key]
        
        try:
            # Check if input is a URL
            is_url_input = self.url_handler.is_url(claim)
            article_from_url = None
            
            # If it's a URL, extract the article
            if is_url_input:
                logger.info(f"Input is a URL. Extracting article content...")
                article_from_url = self.url_handler.extract_article(claim)
                claim_for_analysis = article_from_url["content"]  # Use article content as the claim
                
                # If extraction failed, use the URL as the claim
                if not claim_for_analysis:
                    claim_for_analysis = claim
                    logger.warning(f"Failed to extract article content from URL: {claim}")
            else:
                claim_for_analysis = claim
            
            # Step 1: Extract entities
            logger.info("Extracting entities...")
            entities = self.entity_extractor.get_key_entities(claim_for_analysis)
            
            # Step 2: Analyze sentiment and emotional manipulation
            logger.info("Analyzing sentiment...")
            sentiment_analysis = self.sentiment_analyzer.detect_emotional_manipulation(claim_for_analysis)
            
            retrieved_articles = []
            if use_rag:
                # Step 3: Search for relevant articles
                logger.info("Searching for relevant articles...")
                articles = self.search_engine.search(claim_for_analysis)
                
                if not articles:
                    logger.warning("No articles found for the claim")
                else:
                    logger.info(f"Found {len(articles)} articles")
                    
                    # Step 4: Create embeddings
                    logger.info("Creating embeddings...")
                    texts = [article["full_text"] for article in articles]
                    claim_embedding = self.embedding_engine.get_embeddings([claim_for_analysis])
                    article_embeddings = self.embedding_engine.get_embeddings(texts)
                    
                    # Step 5: Calculate similarities and filter
                    similarities = self.embedding_engine.compute_similarity(claim_embedding[0], article_embeddings)
                    retrieved_articles = self.embedding_engine.filter_by_threshold(articles, similarities)
                    
                    # Alternatively, use FAISS for retrieval
                    # self.faiss_indexer.create_index(article_embeddings)
                    # retrieved_articles = self.faiss_indexer.retrieve_docs(claim_embedding, articles)
                    
                    logger.info(f"Retrieved {len(retrieved_articles)} relevant articles")
            
            # Step 6: Verify entities using Knowledge Graph
            entity_verification = {"status": "skipped"}
            if use_kg and self.knowledge_graph and self.knowledge_graph.available and entities:
                logger.info("Verifying entities with Knowledge Graph...")
                entity_verification = self.knowledge_graph.verify_entity_relationships(entities)
            
            # Step 7: Fact-check using XLM RoBERTa
            logger.info("Fact-checking with Spark LLM...")
            fact_check_result = self.spark_llm.fact_check(claim_for_analysis, retrieved_articles, entities)
            
            if not fact_check_result:
                return {
                    "status": "error",
                    "message": "Failed to get fact-checking result from XLM Roberta",
                    "claim": claim,
                    "sentiment_analysis": sentiment_analysis
                }
            
            # New Step: Analyze credibility if URL input
            credibility_analysis = None
            if is_url_input and article_from_url:
                logger.info("Analyzing source credibility...")
                credibility_analysis = self.credibility_analyzer.assess_overall_source_credibility(
                    domain=article_from_url["source_domain"],
                    author=article_from_url["author"],
                    date_str=article_from_url["published_date"],
                    claim_text=claim_for_analysis
                )
            
            # Step 8: Compile and return results
            result = {
                "status": "success",
                "claim": claim,
                "is_url_input": is_url_input,
                "entities": entities,
                "retrieved_articles": retrieved_articles,
                "sentiment_analysis": sentiment_analysis,
                "entity_verification": entity_verification,
                "fact_check": fact_check_result,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Add source metadata and credibility if URL input
            if is_url_input and article_from_url:
                result["source_metadata"] = {
                    "title": article_from_url["title"],
                    "author": article_from_url["author"],
                    "published_date": article_from_url["published_date"],
                    "domain": article_from_url["source_domain"]
                }
                
                contradiction_analysis = self.spark_llm.detect_title_content_contradiction(
                title=article_from_url['title'], 
                content=article_from_url['content'])
                
                
                result["credibility_analysis"] = credibility_analysis
                result['title_content_contradiction'] = contradiction_analysis
            
            
            # Also do credibility analysis on the top retrieved article if available
            elif retrieved_articles and len(retrieved_articles) > 0:
                top_article = retrieved_articles[0]
                domain = top_article.get("source", "unknown")
                
                # Try to extract author and date from retrieved article
                author = "Unknown Author"
                date = "Unknown Date"
                
                # If we have a link, we could potentially fetch the full article to extract more metadata
                # This would be an enhancement point
                
                logger.info("Analyzing top result source credibility...")
                credibility_analysis = self.credibility_analyzer.assess_overall_source_credibility(
                    domain=domain,
                    author=author,
                    date_str=date,
                    claim_text=claim_for_analysis
                )
                result["credibility_analysis"] = credibility_analysis
            
            # Cache the result
            result_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing claim: {e}")
            return {
                "status": "error",
                "message": str(e),
                "claim": claim
            }
    def close(self):
        """Close any open connections"""
        if self.knowledge_graph:
            self.knowledge_graph.close()
            
    

# API Models for Request/Response validation
class AnalysisRequest(BaseModel):
    claim: str = Field(..., description="The claim to analyze or a URL to an article")
    use_rag: bool = Field(True, description="Whether to use RAG for context retrieval")
    use_kg: bool = Field(True, description="Whether to use knowledge graph for entity verification")
    is_url: bool = Field(False, description="Flag to indicate if the claim is a URL (auto-detected if not specified)")

class AnalysisResponse(BaseModel):
    status: str
    claim: str
    is_url_input: Optional[bool] = False
    verdict: Optional[str] = None
    confidence: Optional[int] = None
    explanation: Optional[str] = None
    emotional_manipulation: Optional[dict] = None
    credibility_assessment: Optional[dict] = None  # Add this field
    source_metadata: Optional[dict] = None  # Add this field
    processing_time: Optional[float] = None
    entities: Optional[dict] = None
    message: Optional[str] = None
# API Server
app = FastAPI(
    title="TruthLens API",
    description="Fake news detection API using XLM ROBERTA with RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instance of the TruthLens system
truthlens = None

@app.on_event("startup")
async def startup_event():
    global truthlens
    try:
        truthlens = FakeNewsDetectionSystem()
        logger.info("TruthLens system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize TruthLens system: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global truthlens
    if truthlens:
        truthlens.close()
        logger.info("TruthLens system shut down")

@app.get("/")
async def root():
    return {"message": "Welcome to TruthLens API - Fake News Detection with Spark LLM"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": str(datetime.now())
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_claim(request: AnalysisRequest):
    """
    Analyze a claim or URL for fake news detection.
    """
    global truthlens
    
    if not truthlens:
        raise HTTPException(status_code=503, detail="TruthLens system not initialized")
    
    try:
        result = truthlens.analyze_claim(
            claim=request.claim,
            use_rag=request.use_rag,
            use_kg=request.use_kg
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        # Extract relevant information for the response
        response = {
            "status": result["status"],
            "claim": result["claim"],
            "is_url_input": result.get("is_url_input", False),
            "verdict": None,
            "confidence": None,
            "explanation": None,
            "emotional_manipulation": {
                "score": result["sentiment_analysis"]["manipulation_score"],
                "level": result["sentiment_analysis"]["manipulation_level"],
                "explanation": result["sentiment_analysis"]["explanation"]
            },
            "processing_time": result["processing_time"],
            "entities": result["entities"]
        }
        
       
    
        
        
        # Add source metadata if available
        if "source_metadata" in result:
            response["source_metadata"] = result["source_metadata"]
        
        # Add credibility assessment if available
        if "credibility_analysis" in result:
            response["credibility_assessment"] = {
                "score": result["credibility_analysis"]["overall_credibility_score"],
                "level": result["credibility_analysis"]["credibility_level"],
                "explanation": result["credibility_analysis"]["explanation"],
                "details": result["credibility_analysis"]["details"]
            }
        
        # Parse fact_check result if it's a string
        if isinstance(result["fact_check"], str):
            fact_check_text = result["fact_check"]
            
            # Extract verdict
            verdict_match = re.search(r"Verdict: (.*?)(?:\n|$)", fact_check_text)
            if verdict_match:
                response["verdict"] = verdict_match.group(1).strip()
            
            # Extract confidence
            confidence_match = re.search(r"Confidence: (\d+)%", fact_check_text)
            if confidence_match:
                response["confidence"] = int(confidence_match.group(1))
            
            # Extract explanation
            explanation_match = re.search(r"Explanation: (.*?)(?:\n\n|$)", fact_check_text, re.DOTALL)
            if explanation_match:
                response["explanation"] = explanation_match.group(1).strip()
                
        else:
            # Handle as dictionary if it's not a string
            response["verdict"] = result["fact_check"].get("verdict")
            response["confidence"] = result["fact_check"].get("confidence")
            response["explanation"] = result["fact_check"].get("explanation")
        
        # Calculate Trust Lens Score
        fact_check_confidence = response['confidence'] or 0  # Default to 0 if None
        author_credibility_factor = result.get('credibility_analysis', {}).get('factors', {}).get('author', {}).get('credibility_factor', 0.5)
        sentiment_analysis = result.get('sentiment_analysis', {})
        
        # Pastikan kita memiliki data yang diperlukan untuk menghitung trust_lens_score
        if 'credibility_analysis' in result and 'overall_credibility_score' in result['credibility_analysis']:
            trust_lens_score = truthlens.credibility_analyzer.calculate_trust_lens_score(
                source_credibility=result['credibility_analysis']['overall_credibility_score'],
                factual_match=fact_check_confidence/100,  # Convert to 0-1 scale
                tone_neutrality=1 - sentiment_analysis.get('manipulation_score', 0),
                source_transparency=author_credibility_factor
            )
            response['trust_lens_score'] = trust_lens_score
        
        # Tambahkan title_content_contradiction ke response jika ada
        if 'title_content_contradiction' in result:
            response['title_content_contradiction'] = result['title_content_contradiction']
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing analysis request: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    
    
# Command line interface
def main():
    """Command line interface for TruthLens"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TruthLens: Fake News Detection with Spark LLM")
    parser.add_argument("--claim", type=str, help="Claim to analyze or URL to analyze")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    parser.add_argument("--no-kg", action="store_true", help="Disable Knowledge Graph")
    parser.add_argument("--api", action="store_true", help="Run as API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    
    args = parser.parse_args()
    
    if args.api:
        # Run as API server
        uvicorn.run("Try_train:app", host=args.host, port=args.port, reload=True)
    elif args.claim:
        # Run in CLI mode
        system = FakeNewsDetectionSystem()
        
        try:
            result = system.analyze_claim(
                claim=args.claim,
                use_rag=not args.no_rag,
                use_kg=not args.no_kg
            )
            
            if result["status"] == "success":
                print("\n=== ANALYSIS RESULTS ===")
                print(f"Claim: {result['claim']}")
                
                # Print if input was URL
                if result.get("is_url_input", False):
                    print("\nInput Type: URL")
                    if "source_metadata" in result:
                        print("\nSource Metadata:")
                        print(f"- Title: {result['source_metadata']['title']}")
                        print(f"- Author: {result['source_metadata']['author']}")
                        print(f"- Date: {result['source_metadata']['published_date']}")
                        print(f"- Domain: {result['source_metadata']['domain']}")
                
                # Print credibility if available
                if "credibility_analysis" in result:
                    print("\nSource Credibility Assessment:")
                    print(f"- Score: {result['credibility_analysis']['overall_credibility_score']:.2f}/1.0")
                    print(f"- Level: {result['credibility_analysis']['credibility_level']}")
                    print(f"- Summary: {result['credibility_analysis']['explanation']}")
                    print("\nCredibility Details:")
                    for detail in result['credibility_analysis']['details']:
                        print(f"  {detail}")
                
                print("\nEntities Detected:")
                for entity_type, entities in result['entities'].items():
                    print(f"- {entity_type}: {', '.join(entities)}")
                
                print("\nSentiment Analysis:")
                print(f"Emotional Manipulation: {result['sentiment_analysis']['manipulation_score']:.2f} ({result['sentiment_analysis']['manipulation_level']})")
                print(f"Dominant Emotion: {result['sentiment_analysis']['dominant_emotion']}")
                print(f"Explanation: {result['sentiment_analysis']['explanation']}")
                
                # Extract fact check information
                if isinstance(result["fact_check"], str):
                    fact_check_text = result["fact_check"]
                    
                    # Extract verdict
                    verdict_match = re.search(r"Verdict: (.*?)(?:\n|$)", fact_check_text)
                    verdict = verdict_match.group(1).strip() if verdict_match else "Unknown"
                    
                    # Extract confidence
                    confidence_match = re.search(r"Confidence: (\d+)%", fact_check_text)
                    confidence = confidence_match.group(1) if confidence_match else "Unknown"
                    
                    # Extract explanation
                    explanation_match = re.search(r"Explanation: (.*?)(?:\n\n|$)", fact_check_text, re.DOTALL)
                    explanation = explanation_match.group(1).strip() if explanation_match else "No explanation available"
                    
                    print("\nFact Check Result:")
                    print(f"Verdict: {verdict} (Confidence: {confidence}%)")
                    print(f"Explanation: {explanation}")
                    
                    # Print raw fact check for detailed reasoning
                    print("\nDetailed Reasoning:")
                    print(fact_check_text)
                else:
                    # Handle if it's an object
                    print("\nFact Check Result:")
                    print(f"Verdict: {result['fact_check']['verdict']} (Confidence: {result['fact_check']['confidence']}%)")
                    print(f"Explanation: {result['fact_check']['explanation']}")
                
                print(f"\nProcessing Time: {result['processing_time']:.2f} seconds")
                print("=" * 50)
            else:
                print(f"Error: {result['message']}")
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            system.close()
    else:
        parser.print_help()
    

if __name__ == "__main__":
    main()