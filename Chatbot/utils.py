import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import spacy
from date_spacy import find_dates
from spacy.util import filter_spans
from spacy.pipeline import EntityRuler
import dateparser
import logging
import stanza
from word2number import w2n

# Import files from the same module
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from Chatbot.settings import train_path
menu = pd.read_csv(train_path + "cafe_menu.csv")
menu["Item"] = menu["Item"].str.lower()

# Setup nlp with the necessary pipes
def setup_nlp():
    # Initialise nlp instance
    nlp_general = spacy.load("en_core_web_sm")

    # For recognising dates in sentences
    nlp_date = spacy.blank("en")
    nlp_date.add_pipe("find_dates", last = True)

    # For recognising food terms in sentences
    ruler = EntityRuler(nlp_general)
    food_patterns = []
    for index, row in menu.iterrows():
        food_patterns.append({"label": "FOOD", "pattern": row["Item"]})

    ruler = nlp_general.add_pipe("entity_ruler")
    ruler.add_patterns(food_patterns)

    # Initialise a new nlp instance for sentiment analysis
    logging.getLogger("stanza").setLevel(logging.ERROR)
    nlp_sentiment = stanza.Pipeline(lang = "en", processors = "tokenize, sentiment", verbose = False)
    
    return nlp_general, nlp_date, nlp_sentiment

nlp_general, nlp_date, nlp_sentiment = setup_nlp()

# Tokenise and lemmatise each word in the sentence
def tokenise_lemmatise(sentence):
    ignore_words = [",", ".", "?", "!", "'", "-"]
    # Tokenise each word in the sentence
    doc = nlp_general(sentence.lower())
    # Lemmatise and lower each word
    w = [token.lemma_.lower() for token in doc if token.text not in ignore_words]
    return w

# Formulate bag of words array to allocate 1 for each know word that exists in the sentence or 0 otherwise
def bag_of_words(sentence, words):
    # Initialise bag with 0 for each word
    bag = np.zeros(len(words), dtype = np.float32)
    for index, word in enumerate(words):
        if word in sentence: 
            bag[index] = 1

    return bag

# Look for numbers in a sentence
def parse_number(sentence):
    doc = nlp_general(sentence.lower())
    numbers = [ent.text for ent in doc.ents if ent.label_ == "CARDINAL"]
    parsed_number = None if numbers == [] else w2n.word_to_num(numbers[0])
    
    return parsed_number

# Look for time in a sentence
def parse_date_time(sentence):
    doc1 = nlp_date(sentence.lower())
    date_phrases = [ent.text for ent in doc1.ents if ent.label_ == "DATE"]
    for date_phrase in date_phrases:
        sentence = sentence.replace(date_phrase, "")
        
    doc2 = nlp_general(sentence.lower())
    time_phrases = [ent.text for ent in doc2.ents if ent.label_ == "TIME"]
    
    if(time_phrases == []):
        return None
    
    elif(date_phrases == []):
        return dateparser.parse(time_phrases[0])
    
    else:
        return dateparser.parse(date_phrases[0] + " " + time_phrases[0])

# Look for any food given in the menu in a sentence
def parse_food(sentence):
    doc = nlp_general(sentence.lower())
    food_phrases = [ent.text for ent in doc.ents if ent.label_ == "FOOD"]
    
    return food_phrases

# Determine sentiment of a message 
def analyse_sentiment(msg):
    doc = nlp_sentiment(msg)
    polarity = ((sum(sentence.sentiment for sentence in doc.sentences) / len(doc.sentences)) - 1) if len(doc.sentences) > 0 else 0
    sentiment = "positive" if polarity >= 0.25 else ("negative" if polarity <= -0.25 else "neutral")
    
    return sentiment
    