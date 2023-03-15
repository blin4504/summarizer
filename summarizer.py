import nltk
import requests
from bs4 import BeautifulSoup
from string import punctuation
from nltk.corpus import stopwords
import numpy as np
import re


def scrape(url):
    # request data from URL
    page = requests.get(url).text

    # scrape it off and find all text contents
    soup = BeautifulSoup(page, "html.parser")
    paragraphs = soup.find_all('p')

    # combine all paragraphs into text without <p> tags
    text = ""
    for i in paragraphs:
        text += i.text
    
    text = editted_text = re.sub(r'\[[0-9]*\]', ' ', text)
    editted_text = re.sub(r'\[[0-9]*\]', ' ', text)
    return text, editted_text.lower()

def preprocess(text, editted):
    # split into sentences and tokenizing
    sentences = nltk.sent_tokenize(text)
    word_token = nltk.word_tokenize(editted)
    filtered_words = [word for word in word_token if word not in stopwords.words('english')]
    filtered_words = [word for word in filtered_words if word not in punctuation]
    return (sentences, filtered_words)


def compute_tf_idf(sentences, filtered_words):
    # compute TF-IDF on all the words
    fdist = nltk.FreqDist(filtered_words)
    for i in fdist.keys():
        fdist[i] /= len(filtered_words)

    # compute the sentence scores
    sentence_scores = {}
    for sent in sentences:
        for word in nltk.word_tokenize(sent.lower()):
            if word in fdist:
                if sent not in sentence_scores:
                    sentence_scores[sent] = fdist[word]
                else:
                    sentence_scores[sent] += fdist[word]

    # normalize the scores
    for sent in sentence_scores:
        sentence_scores[sent] /= len(sent)
    return sentence_scores


# get summary
def get_summary(sentence_scores):
    threshhold = np.mean(list(sentence_scores.values()))
    summary = ""
    for sent in sentence_scores:
        if sentence_scores[sent] >= threshhold * 1.5:
            summary += sent
    return summary

def summarize(url):
    text, editted = scrape(url)
    sentences, filtered_words = preprocess(text, editted)
    scores = compute_tf_idf(sentences, filtered_words)
    return get_summary(scores)


