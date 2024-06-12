from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors module
import pandas as pd
import json
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes by adding this line

# Your remaining Flask app code goes here...


# Load training data
with open('./training_sample.json', 'r') as f:
    training_data = json.load(f)

# Define answers dictionary
answers = {
    "customer_service.say_hello": "Hello! Welcome to our customer service. How can I assist you today?",
    "customer_service.introduction": "Certainly! We offer a wide range of products, including computer accessories, electronics, and home and kitchen appliances. Our selection includes items such as USB cables for computers, HDMI cables for home theater and TV setups, and steam irons for household chores"
}

# Compile punctuation regex
punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))

class MyChatbotData:
    def __init__(self, json_obj, text_fld, answers):
        dfs = []
        for intent, data in json_obj.items():
            patterns = data[text_fld].copy()
            for i, p in enumerate(patterns):
                p = p.lower()
                p = self.remove_punctuation(p)
                patterns[i] = p
            df = pd.DataFrame(list(zip([intent]*len(patterns), patterns, [answers[intent]]*len(patterns))), \
                              columns=['intent', 'phrase', 'answer'])
            dfs.append(df)
        self.df = pd.concat(dfs)

    def get_answer(self, intent):
        return pd.unique(self.df[self.df['intent'] == intent]['answer'])[0]

    def remove_punctuation(self, text):
        return punct_re_escape.sub('', text)

    def get_phrases(self, intent):
        return list(self.df[self.df['intent'] == intent]['phrase'])

    def get_intents(self):
        return list(pd.unique(self.df['intent']))

    def show_batch(self, size=5):
        return self.df.head(size)

    def __len__(self):
        return len(self.df)

chatbot_data = MyChatbotData(training_data, 'patterns', answers)
UNK = "I don't know"

def exact_match(query):
    intents = chatbot_data.get_intents()
    for intent in intents:
        phrases = chatbot_data.get_phrases(intent)
        if query in phrases:
            return chatbot_data.get_answer(intent)
    return UNK

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = exact_match(user_message.lower())
    return jsonify([{"text": response}])

# Run the server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
