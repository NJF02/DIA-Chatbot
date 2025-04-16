import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy.pipeline import EntityRuler
import contextualSpellCheck
import dateparser

nlp = spacy.load('en_core_web_sm')
ruler = EntityRuler(nlp)
patterns = [
    {"label" : "FOOD", "pattern" : "banana"},
    {"label" : "FOOD", "pattern" : "apple"},
    {"label" : "FOOD", "pattern" : "guava"},
    {"label" : "FOOD", "pattern" : "watermelon"},
    {"label" : "FOOD", "pattern" : "orange"}
]

ruler = nlp.add_pipe("entity_ruler")
ruler.add_patterns(patterns)

contextualSpellCheck.add_to_pipe(nlp)
nlp.add_pipe("spacytextblob")

from word2number import w2n

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

def parse_number(sentence):
    doc = nlp(sentence.lower())
    numbers = [ent.text for ent in doc.ents if ent.label_ == "CARDINAL"]
    parsed_number = None if numbers == [] else w2n.word_to_num(numbers[0])
    
    return parsed_number

def parse_time(sentence):
    doc = nlp(sentence.lower())
    time_phrases = [ent.text for ent in doc.ents if ent.label_ == "TIME"]
    parsed_time = None if time_phrases == [] else dateparser.parse(time_phrases[0])
    
    return parsed_time

def parse_food(sentence):
    doc = nlp(sentence.lower())
    food_phrases = [ent.text for ent in doc.ents if ent.label_ == "FOOD"]
    
    return food_phrases
            
def analyse_sentiment(sentence):
    doc = nlp(sentence)
    polarity = doc._.blob.polarity
    sentiment = 'Positive' if polarity >= 0.05 else 'Negative' if polarity <= -0.05 else 'Neutral'
    return sentiment
    