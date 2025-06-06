import nltk
import re
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import TreebankWordTokenizer


lemmatizer = WordNetLemmatizer()

# Your full contraction dictionary
contractions_dict = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
    "hasn't": "has not", "haven't": "have not", "he'd": "he had",
    "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have",
    "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
    "how's": "how is", "i'd": "i had", "i'd've": "i would have", "i'll": "i will",
    "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it'd": "it had", "it'd've": "it would have", "it'll": "it will",
    "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not",
    "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
    "she'd": "she had", "she'd've": "she would have", "she'll": "she will",
    "she'll've": "she will have", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
    "so's": "so is", "that'd": "that had", "that'd've": "that would have",
    "that's": "that is", "there'd": "there had", "there'd've": "there would have",
    "there's": "there is", "they'd": "they had", "they'd've": "they would have",
    "they'll": "they will", "they'll've": "they will have", "they're": "they are",
    "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we had",
    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
    "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
    "what'll've": "what will have", "what're": "what are", "what's": "what is",
    "what've": "what have", "when's": "when is", "when've": "when have",
    "where'd": "where did", "where's": "where is", "where've": "where have",
    "who'll": "who will", "who'll've": "who will have", "who's": "who is",
    "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
    "won't": "will not", "won't've": "will not have", "would've": "would have",
    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
    "y'all'd": "you all would", "y'all'd've": "you all would have",
    "y'all're": "you all are", "y'all've": "you all have", "you'd": "you had",
    "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
    "you're": "you are", "you've": "you have"
}

def get_stopwords():
    stop_words = set(nltk.corpus.stopwords.words("english"))
    stop_words.discard("not")
    return stop_words

def expand_contractions(text):
    return ' '.join([contractions_dict.get(word, word) for word in text.split()])

def clean_text(text):
    stop_words = get_stopwords()
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r'\d+', '', text)
    text = ''.join([c for c in text if c not in punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join(text.split())
    return text

def lemmatize(text):
    tokenizer = TreebankWordTokenizer()
    words = tokenizer.tokenize(text)
    return ' '.join([lemmatizer.lemmatize(w) for w in words])

def preprocess_text(text):
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)
    cleaned = [lemmatize(clean_text(sentence)) for sentence in sentences]
    return ' '.join(cleaned), cleaned

def find_keywords_with_context(sentences, keywords):
    results = []
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            for keyword in keywords:
                if keyword in ' '.join(words[i:i+len(keyword.split())]):
                    start = max(i - 5, 0)
                    end = min(i + 5 + len(keyword.split()), len(words))
                    snippet = ' '.join(words[start:end])
                    results.append([snippet])
                    break
    return results
