from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def tokenize_and_stem(text):
    # Getting rid of links
    text = [word for word in text.lower().split() if not 'http' in word]
    text = ' '.join(text)
    
    # Remove HTML Artifacts
    bs = BeautifulSoup(text, features='lxml')
    text = bs.get_text()
    
    # Tokenize clean text by separating out all word characters
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(text)
    
    # Stem the tokens
    p_stemmer = PorterStemmer()
    return [p_stemmer.stem(i) for i in tokens]