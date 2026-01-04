import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load resources once
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Clean and normalize raw text for NLP tasks.
    """
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords and lemmatize
    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in STOP_WORDS
    ]

    return " ".join(tokens)

