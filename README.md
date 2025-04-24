# Designing Intelligent Agents (COMP3071 UNMC) Coursework
## Bot Cafe Chatbot
Bot Cafe Chatbot is a Python-based virtual assistant built to help customers with table reservations, delivery orders, and chat summaries. It's designed to create a smooth and friendly user experience through natural conversation.

## Basic Features
Reservation: Book a table easily through chat<br>
Delivery: Place orders for delivery<br>
Goodbye: End the chat and receive a summary

## Usage Instructions
### Train the Chatbot
To prepare the chatbot with necessary training data:
1. Navigate to Chatbot/train.py and run the Python file.
2. The trained model and data files will be saved in the Chatbot/Data directory.

### Run the Chatbot
To launch the chatbot and begin interaction:
1. Open Chatbot/chatbot.py.
2. Run the Python file to start chatting with Cafe Bot.

### Evaluate the Chatbot
To assess the chatbot’s performance:
1. Open and run Evaluation/evaluation.py.
2. Performance metrics and evaluation results will be displayed in the console.

## Notes and Configurations
1. Prior to training the chatbot, hyper-parameters for the neural networks can be modified.
2. Make sure to train the chatbot first being running it.
3. Menu can be found in Chatbot/Train/cafe_menu.csv (only food and drinks listed in the menu can be ordered for delivery).

## Dependencies
Make sure the required Python packages are installed before running the chatbot. The list of Python packages can be found in requirements.txt.<br>
You can install all dependencies at once using:
```
pip install -r requirements.txt
```

## Troubleshooting
Here are some common issues and how to fix them:

❌ Issue: ModuleNotFoundError<br>
✅ Solution:<br>
Make sure all dependencies are installed. Run:

```
pip install -r requirements.txt
```

❌ Issue: NLTK-related errors (e.g., missing corpora)<br>
✅ Solution:<br>
Some NLTK resources may need to be downloaded manually. Open a new temporary Python file, run:

```
import nltk
nltk.download('punkt')
```

❌ Issue: Encoding errors when reading files<br>
✅ Solution:<br>
Ensure you're using the correct encoding when reading or writing files:

```
open('file.txt', 'r', encoding='utf-8')
```

❌ Issue: Chatbot not responding as expected<br>
✅ Solution:<br>
1. Confirm that training completed successfully and the model/data files were saved in Chatbot/Data.
2. Double-check that you're running the correct script: chatbot.py.
3. Print or log user inputs and bot responses for debugging.
4. Close the program and rerun to reset the chatbot.

## Lessons Learnt
Throughout the development of the Bot Cafe Chatbot, several key lessons and insights were gained:<br>
1. Data Handling: Preprocessing and organising data correctly was essential for accurate training and evaluation. Small inconsistencies in data format can significantly affect performance.
2. Modular Code Structure: Keeping training, chatting, and evaluation logic in separate folders or files made the project more manageable and easier to debug or extend.
3. Importance of Clear User Prompts: Simple, keyword-based inputs helped streamline user interactions. Carefully designed messages improved the chatbot's usability and practicality.

## Contact Information
If you have any questions, suggestions, or would like to collaborate, feel free to reach out:
* Leong Cheng Hou - szycl3@nottingham.edu.my
* Ng Jun Fei - szyjn1@nottingham.edu.my
