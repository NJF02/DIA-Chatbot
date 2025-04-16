from pathlib import Path
import pandas as pd

# Import data for model from a different module
import sys
sys.path.append(str(Path(__file__).parent.parent))

from Chatbot.bot_config import get_response

# Read csv file for test data
dir = str(Path(__file__).parent)

def evaluate_intent_accuracy():
    intent_test_data = pd.read_csv(dir + '\\Intent Test Data.csv', names = ["msg", "tag"])
    intent_accuracy = 0
    for index, row in intent_test_data.iterrows():
        if(get_response(row['msg'])['tags']['tag'] == row['tag']):
            intent_accuracy += 1
        
    intent_accuracy /= len(intent_test_data)
    return intent_accuracy

def evaluate_sentiment_accuracy():
    sentiment_test_data = pd.read_csv(dir + '\\Sentiment Test Data.csv', names = ["msg", "sentiment"])
    sentiment_accuracy = 0
    for index, row in sentiment_test_data.iterrows():
        if(get_response(row['msg'])['sentiment'] == row['sentiment']):
            sentiment_accuracy += 1
            
    sentiment_accuracy /= len(sentiment_test_data)
    return sentiment_accuracy

print(evaluate_sentiment_accuracy())
