import random
import json
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import tokenise_lemmatise, bag_of_words, check_typo, analyse_sentiment
from model_framework import NeuralNet
from settings import train_path, data_path, sub_categories

# initialise model parameters
intents = {}
input_size = {}
hidden_size = {}
output_size = {}
tags = {}
all_words = {}
model_state = {}
model = {}

# access master model parameters
json_file = train_path + "master.json"
data_file = data_path + "master.pth"
with open(json_file, 'r', encoding = 'utf-8') as json_data:
    intents['master'] = json.load(json_data)

data = torch.load(data_file)

# obtain model hyperparameters
input_size['master'] = data['input_size']
hidden_size['master'] = data['hidden_size']
output_size['master'] = data['output_size']
tags['master'] = data['tags']
all_words['master'] = data['all_words']
model_state['master'] = data['model_state']

# initialise trained model
model['master'] = NeuralNet(input_size['master'], hidden_size['master'], output_size['master']).to(device)
model['master'].load_state_dict(model_state['master'])
model['master'].eval()

# access sub category models parameters
for sub_category in sub_categories:
    json_file = train_path + sub_category + ".json"
    data_file = data_path + sub_category + ".pth"
    with open(json_file, 'r', encoding = 'utf-8') as json_data:
        intents[sub_category] = json.load(json_data)

    data = torch.load(data_file)

    # obtain model hyperparameters
    input_size[sub_category] = data['input_size']
    hidden_size[sub_category] = data['hidden_size']
    output_size[sub_category] = data['output_size']
    tags[sub_category] = data['tags']
    all_words[sub_category] = data['all_words']
    model_state[sub_category] = data['model_state']

    # initialise trained model
    model[sub_category] = NeuralNet(input_size[sub_category], hidden_size[sub_category], output_size[sub_category]).to(device)
    model[sub_category].load_state_dict(model_state[sub_category])
    model[sub_category].eval()

bot_name = "Bot"
all_tags = []

def transform_input(category, msg):
    # tokenise and lemmatise message and reform it to input into model
    w = tokenise_lemmatise(msg)
    X = bag_of_words(w, all_words[category])
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    return X

def determine_tag(category, X):
    # input into model to determine the tag for the message
    output = model[category](X)
    # applied softmax function to find probabilities for each tag
    probs = torch.softmax(output, dim = 1)
    # determine tag with highest probability
    prob, predicted = torch.max(probs, dim = 1)
    if prob.item() > 0.75:
        tag = tags[category][predicted.item()]
        for intent in intents[category]['intents']:
            if tag == intent['tag']:
                return tag, intent
    else:
        return None, None

unclear_responses = [
    "I didn't understand your message. You could try asking about our location, viewing the menu, or checking our opening hours.",
    "Hmm, I'm not sure how to help with that. Maybe try asking about reservations, the menu, or the restaurant's location.",
    "I'm sorry, I didn't quite catch that. Perhaps you could rephrase it, or ask about things like reservations, the menu, or our business hours.",
    "I'm not sure I understood that. You might want to ask about our menu, how to make a reservation, or what time we're open.",
    "Oops! That's outside what I can understand right now. You could try asking about our menu, opening hours, how to make a reservation, or where to find our website."
    ]
    
negative_responses = [
    "Apologies for the inconvenience.",
    "I truly apologise for the trouble.",
    "I understand your frustration, and I'm sorry.",
    "I'm sorry, and I appreciate your patience.",
    "We're sorry things didn't go as expected."
    ]
    
def get_response(msg):
    # check for typos
    # msg = check_typo(msg)
    # print(msg)
    # check for master tag
    X = transform_input('master', msg)
    sub_category, _ = determine_tag('master', X)
    # check for sentiment
    sentiment = analyse_sentiment(msg)
    sentiment_msg = random.choice(negative_responses) + "\n" if sentiment == 'Negative' else ''
    
    if sub_category != None:
        # check for sub category tag
        X = transform_input(sub_category, msg)
        tag, intent = determine_tag(sub_category, X)
        print((sub_category, tag))
        all_tags.extend((sub_category, tag))
        
        if(tag != None):
            return sentiment_msg + random.choice(intent['responses'])
    else:
        print((None, None))
        all_tags.extend((None, None))
        
    return sentiment_msg + random.choice(unclear_responses)  

def analyse_text(msg):
    sentiment = analyse_sentiment(msg)
    return f"Sentiment: {sentiment}"





