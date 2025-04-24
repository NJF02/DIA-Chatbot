# Import data for model from a different module
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from Chatbot.settings import categories
from Chatbot.bot_config import get_response

# Read csv file for test data
dir = str(Path(__file__).parent)

def evaluate_intent_accuracy():
    intent_test_data = pd.read_csv(dir + "\\Intent Test Data.csv", names = ["msg", "category", "tag"])
    intent_matrix = {"general": {"category": 0, "tag": 0, "total": 0}, 
                     "menu_cuisine": {"category": 0, "tag": 0, "total": 0}, 
                     "reservation": {"category": 0, "tag": 0, "total": 0}, 
                     "delivery": {"category": 0, "tag": 0, "total": 0}}
    intent_accuracy = {"general": {"main": 0, "sub": 0}, 
                       "menu_cuisine": {"main": 0, "sub": 0},
                       "reservation": {"main": 0, "sub": 0},
                       "delivery": {"main": 0, "sub": 0}}
    intent_overall_accuracy = 0
    
    for index, row in intent_test_data.iterrows():
        tags = get_response(row["msg"])["tags"]
        category, tag = tags["category"], tags["tag"]
        # If tag is correct
        if(tag == row["tag"]):
            intent_matrix[row["category"]]["category"] += 1
            intent_matrix[row["category"]]["tag"] += 1
            intent_matrix[row["category"]]["total"] += 1
            
        # If category is correct but tag is incorrect
        elif(category == row["category"]):
            intent_matrix[row["category"]]["category"] += 1
            intent_matrix[row["category"]]["total"] += 1
            
        # If category is incorrect
        else:
            intent_matrix[row["category"]]["total"] += 1
            
    for category in categories:
        intent_accuracy[category]["main"] = intent_matrix[category]["category"] / intent_matrix[category]["total"]
        intent_accuracy[category]["sub"] = intent_matrix[category]["tag"] / intent_matrix[category]["total"]
        intent_overall_accuracy += intent_matrix[category]["tag"]
        
    intent_overall_accuracy /= len(intent_test_data)
        
    return intent_accuracy, intent_overall_accuracy, intent_matrix

def evaluate_sentiment_accuracy():
    sentiment_test_data = pd.read_csv(dir + "\\Sentiment Test Data.csv", names = ["msg", "sentiment"])
    sentiment_accuracy = 0
    sentiment_confusion_matrix = {"positive": {"positive": 0, "negative": 0, "neutral": 0},
                                  "negative": {"positive": 0, "negative": 0, "neutral": 0},
                                  "neutral": {"positive": 0, "negative": 0, "neutral": 0}}
    for index, row in sentiment_test_data.iterrows():
        sentiment = get_response(row["msg"])["sentiment"]
        sentiment_confusion_matrix[row["sentiment"]][sentiment] += 1
        if(sentiment == row["sentiment"]):
            sentiment_accuracy += 1
            
    sentiment_accuracy /= len(sentiment_test_data)
    return sentiment_accuracy, sentiment_confusion_matrix

intent_accuracy, intent_overall_accuracy, intent_matrix = evaluate_intent_accuracy()
print("Intent Overall Accuracy: " + str(intent_overall_accuracy))
# print(intent_accuracy)
# print(intent_matrix)

sentiment_accuracy, sentiment_confusion_matrix = evaluate_sentiment_accuracy()
print("Sentiment Accuracy: " + str(sentiment_accuracy))
# print(sentiment_confusion_matrix)
