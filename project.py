"""
Fake News Detection System using BERT with RAG
===================================================
This system combines:
1. Web search for retrieving relevant articles
2. Embedding-based similarity for pre-filtering
3. FAISS for efficient similarity search
4. BERT for fact-checking and verdict generation
5. Sentiment analysis to detect emotional manipulation
6. Named Entity Recognition to extract key entities
"""
from Try_model import FakeNewsDetectionSystem

# Example usage
if __name__ == "__main__":
    # Initialize the system with the BERT fake news detection model
    # Set use_local=False to use the Hugging Face API instead of local models if needed
    detector = FakeNewsDetectionSystem(model_name="google-bert/bert-base-uncased", use_local=True)
    
    # Example claims to analyze
    claims = [
        "COVID-19 vaccines contain microchips for tracking people",
        "Studies show that drinking coffee every day reduces risk of heart disease",
        "The Earth is flat and NASA is hiding the truth"
    ]
    
    # Analyze each claim
    for claim in claims:
        result = detector.analyze_claim(claim)
        
        if result["status"] == "success":
            print("\n=== ANALYSIS RESULTS ===")
            print(f"Claim: {result['claim']}")
            print("\nSentiment Analysis:")
            print(f"Emotional Intensity: {result['sentiment_analysis']['emotional_intensity']:.2f}")
            print(f"Dominant Emotion: {result['sentiment_analysis']['dominant_emotion']}")
            print(f"Is Emotionally Charged: {result['sentiment_analysis']['is_emotionally_charged']}")
            
            print("\nFact Check Result:")
            print(result['fact_check_result'])
            print("=" * 50)
        else:
            print(f"Error: {result['message']}")