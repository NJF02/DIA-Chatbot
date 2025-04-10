import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import contextualSpellCheck

nlp = spacy.load('en_core_web_sm')
contextualSpellCheck.add_to_pipe(nlp)
nlp.add_pipe("spacytextblob")

def tokenise_lemmatise(sentence):
    ignore_words = [',', '.', '?', '!', "'", "-"]
    # tokenise each word in the sentence
    doc = nlp(sentence.lower())
    # lemmatise and lower each word
    w = [token.lemma_.lower() for token in doc if token.text not in ignore_words]
    return w

def bag_of_words(sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0   ]
    """
    # initialise bag with 0 for each word
    bag = np.zeros(len(words), dtype = np.float32)
    for index, word in enumerate(words):
        if word in sentence: 
            bag[index] = 1

    return bag

def check_typo(sentence):
    doc = nlp(sentence.lower())
    sentence = doc._.outcome_spellCheck if doc._.performed_spellCheck else sentence
    return sentence

def analyse_sentiment(sentence):
    doc = nlp(sentence)
    polarity = doc._.blob.polarity
    sentiment = 'Positive' if polarity >= 0.05 else 'Negative' if polarity <= -0.05 else 'Neutral'
    return sentiment
    