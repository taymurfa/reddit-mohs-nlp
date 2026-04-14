import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NONALPHA_PATTERN = re.compile(r"[^a-z\s]")

def ensure_nltk():
    for res, name in [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]:
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(name, quiet=True)

def clean_text(text: str, stop_words, lemmatizer) -> list:
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = NONALPHA_PATTERN.sub(" ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return [t for t in tokens if len(t) > 1]

def preprocess_data(texts: list) -> list:
    """
    Cleans raw text data and tokenizes it.
    Filters out pieces of text that have fewer than 5 tokens.
    """
    if not texts:
        return []
        
    ensure_nltk()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    processed = []
    for doc in texts:
        if isinstance(doc, dict):
            t = doc.get("text", "")
            url = doc.get("url", "")
            doc_type = doc.get("type", "COMMENT")
        else:
            t = doc
            url = ""
            doc_type = "COMMENT"
            
        toks = clean_text(t, stop_words, lemmatizer)
        if len(toks) >= 5:
            processed.append({
                "tokens": toks,
                "url": url,
                "original_text": t,
                "type": doc_type
            })
    return processed
