import random
import json
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import files from the same module
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from Chatbot.utils import menu, tokenise_lemmatise, bag_of_words, analyse_sentiment, parse_number, parse_date_time, parse_food
from Chatbot.model_framework import NeuralNet
from Chatbot.settings import train_path, data_path, categories

# Initialise model parameters
intents = {}
input_size = {}
hidden_size = {}
output_size = {}
tags = {}
all_words = {}
model_state = {}
model = {}

# Access master model parameters
json_file = train_path + "master.json"
data_file = data_path + "master.pth"
with open(json_file, "r", encoding = "utf-8") as json_data:
    intents["master"] = json.load(json_data)

data = torch.load(data_file)

# Obtain model hyperparameters
input_size["master"] = data["input_size"]
hidden_size["master"] = data["hidden_size"]
output_size["master"] = data["output_size"]
tags["master"] = data["tags"]
all_words["master"] = data["all_words"]
model_state["master"] = data["model_state"]

# Initialise trained model
model["master"] = NeuralNet(input_size["master"], hidden_size["master"], output_size["master"]).to(device)
model["master"].load_state_dict(model_state["master"])
model["master"].eval()

# Access category models parameters
for category in categories:
    json_file = train_path + category + ".json"
    data_file = data_path + category + ".pth"
    with open(json_file, "r", encoding = "utf-8") as json_data:
        intents[category] = json.load(json_data)

    data = torch.load(data_file)

    # Obtain model hyperparameters
    input_size[category] = data["input_size"]
    hidden_size[category] = data["hidden_size"]
    output_size[category] = data["output_size"]
    tags[category] = data["tags"]
    all_words[category] = data["all_words"]
    model_state[category] = data["model_state"]

    # Initialise trained model
    model[category] = NeuralNet(input_size[category], hidden_size[category], output_size[category]).to(device)
    model[category].load_state_dict(model_state[category])
    model[category].eval()

bot_name = "Cafe Bot"
bot_first_msg = """Welcome to Bot Cafe! ☕🤖
I'm your virtual barista — ready to help you reserve a table or get your favorite drinks delivered!

Just type:
📅 "reservation" - to book a table
🚚 "delivery" - to order food or drinks to your door
👋 "goodbye" - to get a summary before you leave

How can I help you today?"""
all_tags = []

def check_yes_no(msg):
    w = tokenise_lemmatise(msg)
    return "y" if "yes" in w else "n" if "no" in w else None

def transform_input(category, msg):
    # Tokenise and lemmatise message and reform it to input into model
    w = tokenise_lemmatise(msg)
    X = bag_of_words(w, all_words[category])
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    return X

def determine_tag(category, X):
    # Input into model to determine the tag for the message
    output = model[category](X)
    # Apply softmax function to find probabilities for each tag
    probs = torch.softmax(output, dim = 1)
    # Determine tag with highest probability
    prob, predicted = torch.max(probs, dim = 1)
    if prob.item() > 0.75:
        tag = tags[category][predicted.item()]
        for intent in intents[category]["intents"]:
            if tag == intent["tag"]:
                return tag, intent
    else:
        return None, None

# Return a message to prompt user to ask specific questions whenever user intent is unclear
unclear_responses = [
    "I didn't understand your message. You could try asking about our location, viewing the menu, or checking our opening hours.",
    "Hmm, I'm not sure how to help with that. Maybe try asking about reservations, the menu, or the restaurant's location.",
    "I'm sorry, I didn't quite catch that. Perhaps you could rephrase it, or ask about things like reservations, the menu, or our business hours.",
    "I'm not sure I understood that. You might want to ask about our menu, how to make a reservation, or what time we're open.",
    "Oops! That's outside what I can understand right now. You could try asking about our menu, opening hours, how to make a reservation, or where to find our website."
]

# Return a message after user says no to a bot question
prompt_responses = [
    "Alright, you could try asking about our location, viewing the menu, or checking our opening hours.",
    "Alright, you could try asking about our menu, business hours, or where to find our website."
]    

# Return an apology message whenever negative sentiment is detected
negative_responses = [
    "Apologies for the inconvenience.",
    "I truly apologise for the trouble.",
    "I understand your frustration, and I'm sorry.",
    "I'm sorry, and I appreciate your patience.",
    "We're sorry things didn't go as expected."
]
    
# Keep track of user responses to create follow-up queries
user_responses = {
    "reservation": {
        "query": False,
        "process": False,
        "made": False,
        "pax": None,
        "date_time": None,
        "msg": None,
        "cancel": False
    },
    "delivery": {
        "query": False,
        "process": False,
        "made": False,
        "food": [],
        "cost": 0,
        "calorie": 0,
        "msg": None,
        "cancel": False
    }
}
    
# Update reservation and delivery status
def update_status(msg):
    # If user wants to make a reservation or place a delivery, given that the user asked about it beforehand
    if(user_responses["reservation"]["query"]):
        user_responses["reservation"]["query"] = False
        if(check_yes_no(msg) == "y"):
            user_responses["reservation"]["process"] = True
            return "Could you please let me know the date, time, and number of people for your reservation?"
            
        elif(check_yes_no(msg) == "n"):
            return random.choice(prompt_responses)
        
        else:
            return None
        
    elif(user_responses["delivery"]["query"]):
        user_responses["delivery"]["query"] = False
        if(check_yes_no(msg) == "y"):
            user_responses["delivery"]["process"] = True
            return "What would you like to order? We have yogurt parfait, falafel wrap and many more which you can find on www.cafebot.com."
            
        elif(check_yes_no(msg) == "n"):
            return random.choice(prompt_responses)
        
        else:
            return None
        
    # If user provides details for reservation or delivery, given that the user agreed to make a reservation or place a delivery
    if(user_responses["reservation"]["process"]):
        user_responses["reservation"]["process"] = False
        pax = parse_number(msg)
        date_time = parse_date_time(msg)
        if(pax != None and date_time != None):
            user_responses["reservation"]["made"] = True
            user_responses["reservation"]["pax"] = pax
            user_responses["reservation"]["date_time"] = date_time
            user_responses["reservation"]["msg"] = f"Reservation has been made on {date_time} for {pax}!"
            
            return user_responses["reservation"]["msg"]
            
        else:
            return "It seems that the details you have provided do not seem relevant or are incomplete, please retry."
        
    elif(user_responses["delivery"]["process"]):
        user_responses["delivery"]["process"] = False
        all_food = parse_food(msg)
        if(all_food != []):
            user_responses["delivery"]["made"] = True
            user_responses["delivery"]["food"] += all_food
            for food in all_food:
                user_responses["delivery"]["cost"] += int(menu.loc[menu["Item"] == food, "Price (RM)"].iloc[0])
                user_responses["delivery"]["calorie"] += int(menu.loc[menu["Item"] == food, "Calories (kcal)"].iloc[0])
            
            all_food = user_responses["delivery"]["food"]
            msg = "Order for delivery has been placed for "
            
            # Formulate message to confirm the food ordered for delivery
            if(len(all_food) == 1):
                msg += f"{all_food[0]}!"
            
            else:
                for i in range(len(all_food) - 1):
                    msg += f"{all_food[i]}, "
                    
                msg += f"and {all_food[-1]}!"
                
            user_responses["delivery"]["msg"] = msg
            
            return user_responses["delivery"]["msg"]
            
        else:
            return "It seems that we do not serve what you have ordered. Please find our menu at www.cafebot.com and retry."
        
    # If user wants to cancel reservation or delivery, given that the user has made a reservation or placed a delivery
    if(user_responses["reservation"]["cancel"]):
        user_responses["reservation"]["cancel"] = False
        if(check_yes_no(msg) == "y"):
            if(user_responses["reservation"]["made"]):
                user_responses["reservation"]["made"] = False
                user_responses["reservation"]["pax"] = None
                user_responses["reservation"]["date"] = None
                user_responses["reservation"]["time"] = None
                user_responses["reservation"]["msg"] = None
                return "Reservation has been cancelled!"
            
            else:
                return "No active reservaiton to cancel."
        
        elif(check_yes_no(msg) == "n"):
            return random.choice(prompt_responses)
        
        else:
            return None
        
    elif(user_responses["delivery"]["cancel"]):
        user_responses["delivery"]["cancel"] = False
        if(check_yes_no(msg) == "y"):
            if(user_responses["delivery"]["made"]):
                user_responses["delivery"]["made"] = False
                user_responses["delivery"]["food"] = []
                user_responses["delivery"]["cost"] = 0
                user_responses["delivery"]["calorie"] = 0
                user_responses["delivery"]["msg"] = None
                return "Delivery has been cancelled!"
            
            else:
                return "No active delivery to cancel."
        
        elif(check_yes_no(msg) == "n"):
            return random.choice(prompt_responses)
        
        else:
            return None
    
# Formulate a response based on intent, sentiment and context
def get_response(msg):
    # Check for sentiment
    sentiment = analyse_sentiment(msg)
    sentiment_msg = random.choice(negative_responses) + "\n" if sentiment == "negative" else ""
    
    # Multi-step reasoning and/or contextual tracking
    bot_response = update_status(msg)
    if(bot_response != None):
        return {"tags": {"category": None, "tag": None}, 
                "sentiment": sentiment, "response": sentiment_msg + bot_response}
    
    # Check for master tag
    X = transform_input("master", msg)
    category, _ = determine_tag("master", X)
    
    if(category != None):
        # Check for tag
        X = transform_input(category, msg)
        tag, intent = determine_tag(category, X)
        all_tags.append((category, tag))
        
        # If user asks for the cost or calorie of a certain food or drink in the menu
        if(tag == "cost" or tag == "calorie"):
            all_food = parse_food(msg)
            response = ""
            if(all_food != []):
                if(tag == "cost"):
                    for food in all_food:
                        cost = str(menu.loc[menu["Item"] == food, "Price (RM)"].iloc[0])
                        response += food.capitalize() + " costs RM" + cost + ". "
                        
                else:
                    for food in all_food:
                        calorie = str(menu.loc[menu["Item"] == food, "Calories (kcal)"].iloc[0])
                        response += food.capitalize() + " has " + calorie + "kcal calories. "
                    
            else:
                response = "It seems that the food you have listed is not in our menu. Please find our menu at www.cafebot.com and retry."
                
            return {"tags": {"category": category, "tag": tag},
                    "sentiment": sentiment, "response": sentiment_msg + response}
        
        if(tag != None):
            # if user asks for reservation or delivery
            if(tag == "make_reservation"):
                user_responses["reservation"]["query"] = True
            
            elif(tag == "order_delivery"):
                user_responses["delivery"]["query"] = True
                
            elif(tag == "cancel_reservation"):
                user_responses["reservation"]["cancel"] = True
                
            elif(tag == "cancel_delivery"):
                user_responses["delivery"]["cancel"] = True
            
            return {"tags": {"category": category, "tag": tag}, 
                    "sentiment": sentiment, "response": sentiment_msg + random.choice(intent["responses"])}
    else:
        all_tags.append((None, None))
        
    return {"tags": {"category": None, "tag": None}, 
            "sentiment": sentiment, "response": sentiment_msg + random.choice(unclear_responses)}

# Provide a summary of reservation and/or delivery details if made
def get_summary():
    summary = ""
    
    if(user_responses["reservation"]["made"] == True):
        summary += "- " + user_responses["reservation"]["msg"] + "\n\n"

    else:
        summary += "- No reservation has been made.\n\n"

    if(user_responses["delivery"]["made"] == True):
        summary += "- " + user_responses["delivery"]["msg"] + "\n"
        summary += "- Total cost is RM" + str(user_responses["delivery"]["cost"]) + ".\n"
        summary += "- Total calorie is " + str(user_responses["delivery"]["calorie"]) + "kcal.\n\n"
        
    else:
        summary += "- No delivery has been ordered.\n\n"

    return summary