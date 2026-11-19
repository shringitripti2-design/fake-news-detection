import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Safe stopwords loading with robust fallback
try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
        "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
        'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
        'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
        'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
        'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
        "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
        "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
        "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
        'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
        "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
        'won', "won't", 'wouldn', "wouldn't"
    }

stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    """
    Cleans and preprocesses news article text for Machine Learning analysis:
    1. Lowercase conversion
    2. URL and HTML tag removal
    3. Removal of punctuation, numbers, and special symbols
    4. Tokenization and stopword removal
    5. Porter Stemming (keeps tokens with length >= 2 such as pm, ai, us, uk, eu, dr)
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # 4. Remove special characters and digits, keeping only letters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 5. Tokenize into words
    words = text.split()
    
    # 6. Remove stopwords and apply Porter Stemmer (keep tokens length >= 2)
    cleaned_words = [
        stemmer.stem(word) for word in words
        if word not in STOP_WORDS and len(word) >= 2
    ]
    
    return " ".join(cleaned_words)

if __name__ == "__main__":
    sample_text = "Narendra modi pm h india ke"
    print("Original:", sample_text)
    print("Cleaned :", preprocess_text(sample_text))
