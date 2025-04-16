import random
import json
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import files from the same module
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from Chatbot.utils import tokenise_lemmatise, bag_of_words, analyse_sentiment, parse_number, parse_time, parse_food
from Chatbot.model_framework import NeuralNet
from Chatbot.settings import train_path, data_path, categories

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

# access category models parameters
for category in categories:
    json_file = train_path + category + ".json"
    data_file = data_path + category + ".pth"
    with open(json_file, 'r', encoding = 'utf-8') as json_data:
        intents[category] = json.load(json_data)

    data = torch.load(data_file)

    # obtain model hyperparameters
    input_size[category] = data['input_size']
    hidden_size[category] = data['hidden_size']
    output_size[category] = data['output_size']
    tags[category] = data['tags']
    all_words[category] = data['all_words']
    model_state[category] = data['model_state']

    # initialise trained model
    model[category] = NeuralNet(input_size[category], hidden_size[category], output_size[category]).to(device)
    model[category].load_state_dict(model_state[category])
    model[category].eval()

bot_name = "Bot"
all_tags = []

def check_yes_no(msg):
    w = tokenise_lemmatise(msg)
    return 'y' if 'yes' in w else 'n' if 'no' in w else None

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

# return a message to prompt user to ask specific questions whenever user intent is unclear
unclear_responses = [
    "I didn't understand your message. You could try asking about our location, viewing the menu, or checking our opening hours.",
    "Hmm, I'm not sure how to help with that. Maybe try asking about reservations, the menu, or the restaurant's location.",
    "I'm sorry, I didn't quite catch that. Perhaps you could rephrase it, or ask about things like reservations, the menu, or our business hours.",
    "I'm not sure I understood that. You might want to ask about our menu, how to make a reservation, or what time we're open.",
    "Oops! That's outside what I can understand right now. You could try asking about our menu, opening hours, how to make a reservation, or where to find our website."
]

# return a message after user says no to a bot question
prompt_responses = [
    "Alright, you could try asking about our location, viewing the menu, or checking our opening hours.",
    "Alright, you could try asking about our menu, business hours, or where to find our website."
]    

# return an apology message whenever negative sentiment is detected
negative_responses = [
    "Apologies for the inconvenience.",
    "I truly apologise for the trouble.",
    "I understand your frustration, and I'm sorry.",
    "I'm sorry, and I appreciate your patience.",
    "We're sorry things didn't go as expected."
]
    
# keep track of user responses to create follow-up queries
user_responses = {
    "reservation": {
        "query": False,
        "process": False,
        "made": False,
        "pax": None,
        "time": None,
        "cancel": False
    },
    "delivery": {
        "query": False,
        "process": False,
        "made": False,
        "food": [],
        "address": None,
        "cancel": False
    }
}
    
def update_status(msg):
    # if user wants to make a reservation or place a delivery, given that the user asked about it beforehand
    if(user_responses['reservation']['query']):
        user_responses['reservation']['query'] = False
        if(check_yes_no(msg) == 'y'):
            user_responses['reservation']['process'] = True
            return "When would you like your reservation to be and for how many people?"
            
        elif(check_yes_no(msg) == 'n'):
            return random.choice(prompt_responses)
        
        else:
            return None
        
    elif(user_responses['delivery']['query']):
        user_responses['delivery']['query'] = False
        if(check_yes_no(msg) == 'y'):
            user_responses['delivery']['process'] = True
            return "What would you like to order?"
            
        elif(check_yes_no(msg) == 'n'):
            return random.choice(prompt_responses)
        
        else:
            return None
        
    # if user provides details for reservation or delivery, given that the user agreed to make a reservation or place a delivery
    if(user_responses['reservation']['process']):
        user_responses['reservation']['process'] = False
        pax = parse_number(msg)
        time = parse_time(msg)
        if(pax != None and time != None):
            user_responses['reservation']['made'] = True
            user_responses['reservation']['pax'] = pax
            user_responses['reservation']['time'] = time
            return "Reservation has been made!"
            
        else:
            return "It seems that the details you have provided do not seem relevant or are incomplete, please retry."
        
    elif(user_responses['delivery']['process']):
        user_responses['delivery']['process'] = False
        food = parse_food(msg)
        if(food != []):
            user_responses['delivery']['made'] = True
            user_responses['delivery']['food'] += [food]
            return "Order for delivery has been placed!"
            
        else:
            return "It seems that we do not serve what you have ordered. Please find our menu at www.cafebot.com and retry."
        
    # if user wants to cancel reservation or delivery, given that the user has made a reservation or placed a delivery
    if(user_responses['reservation']['cancel']):
        user_responses['reservation']['cancel'] = False
        if(check_yes_no(msg) == 'y'):
            if(user_responses['reservation']['made']):
                user_responses['reservation']['made'] = False
                user_responses['reservation']['pax'] = None
                user_responses['reservation']['date'] = None
                user_responses['reservation']['time'] = None
                return "Reservation has been cancelled!"
            
            else:
                return "No active reservaiton to cancel."
        
        elif(check_yes_no(msg) == 'n'):
            return random.choice(prompt_responses)
        
        else:
            return None
        
    elif(user_responses['delivery']['cancel']):
        user_responses['delivery']['cancel'] = False
        if(check_yes_no(msg) == 'y'):
            if(user_responses['delivery']['made']):
                user_responses['delivery']['made'] = False
                user_responses['delivery']['food'] = []
                return "Delivery has been cancelled!"
            
            else:
                return "No active delivery to cancel."
        
        elif(check_yes_no(msg) == 'n'):
            return random.choice(prompt_responses)
        
        else:
            return None
    
def get_response(msg):
    # check for sentiment
    sentiment = analyse_sentiment(msg)
    sentiment_msg = random.choice(negative_responses) + "\n" if sentiment == 'Negative' else ''
    
    # multi-step reasoning and/or contextual tracking
    bot_response = update_status(msg)
    if(bot_response != None):
        return {"tags": {"category": None, "tag": None}, 
                "sentiment": sentiment, "response": sentiment_msg + bot_response}
        return ((None, None), sentiment, sentiment_msg + bot_response)
    
    # check for master tag
    X = transform_input('master', msg)
    category, _ = determine_tag('master', X)
    
    if category != None:
        # check for tag
        X = transform_input(category, msg)
        tag, intent = determine_tag(category, X)
        print((category, tag))
        all_tags.append((category, tag))
        
        if(tag != None):
            # if user asks for reservation or delivery
            if(tag == 'make_reservation'):
                user_responses['reservation']['query'] = True
            
            elif(tag == 'order_delivery'):
                user_responses['delivery']['query'] = True
                
            elif(tag == 'cancel_reservation'):
                user_responses['reservation']['cancel'] = True
                
            elif(tag == 'cancel_delivery'):
                user_responses['delivery']['cancel'] = True
            
            return {"tags": {"category": category, "tag": tag}, 
                    "sentiment": sentiment, "response": sentiment_msg + random.choice(intent["responses"])}
            return ((category, tag), sentiment, sentiment_msg + random.choice(intent['responses']))
    else:
        print((None, None))
        all_tags.append((None, None))
        
    return {"tags": {"category": None, "tag": None}, 
            "sentiment": sentiment, "response": sentiment_msg + random.choice(unclear_responses)}
    return ((None, None), sentiment, sentiment_msg + random.choice(unclear_responses))

def analyse_text(msg):
    sentiment = analyse_sentiment(msg)
    return f"Sentiment: {sentiment}"





