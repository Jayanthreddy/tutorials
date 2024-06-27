import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import textstat
from textblob import TextBlob

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load a small language model and tokenizer (e.g., DistilBERT)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set up the pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Function to evaluate readability
def evaluate_readability(text):
    flesch_score = textstat.flesch_reading_ease(text)
    return flesch_score

# Function to evaluate completeness
def evaluate_completeness(text):
    # Heuristic: Check for presence of key elements in a Jira story
    required_phrases = ["As a", "I want", "so that"]
    completeness_score = sum(phrase in text for phrase in required_phrases) / len(required_phrases) * 10
    return completeness_score

# Function to evaluate clarity
def evaluate_clarity(text):
    # Calculate sentiment polarity using TextBlob (range: -1 to 1, where >0 means positive sentiment)
    blob = TextBlob(text)
    polarity_score = blob.sentiment.polarity
    return polarity_score

# Function to evaluate technical accuracy
def evaluate_technical_accuracy(text):
    # Example: Check for presence of technical terms or acronyms
    technical_terms = ["API", "UI", "SDK"]
    accuracy_score = sum(term in text for term in technical_terms) / len(technical_terms) * 10
    return accuracy_score

# Function to evaluate ease of understanding
def evaluate_understanding(text):
    # Use the language model to classify the text into "easy" or "difficult"
    scores = classifier(text)[0]
    understanding_score = scores[0]['score'] * 10  # Assuming the first label is for "easy"
    return understanding_score

# Function to process a Jira story
def process_jira_story(story):
    readability_score = evaluate_readability(story)
    completeness_score = evaluate_completeness(story)
    clarity_score = evaluate_clarity(story)
    technical_accuracy_score = evaluate_technical_accuracy(story)
    understanding_score = evaluate_understanding(story)
    
    return {
        'readability_score': readability_score,
        'completeness_score': completeness_score,
        'clarity_score': clarity_score,
        'technical_accuracy_score': technical_accuracy_score,
        'understanding_score': understanding_score,
        'average_score': (readability_score + completeness_score + clarity_score + technical_accuracy_score + understanding_score) / 5
    }

# Example Jira story
jira_story = """
As a user, I want to be able to reset my password so that I can regain access to my account if I forget my password.
Acceptance Criteria:
1. The user should receive a password reset link via email.
2. The link should expire after 24 hours.
3. The user should be able to set a new password that meets security requirements.
"""

# Process the Jira story
scores = process_jira_story(jira_story)
print(scores)
