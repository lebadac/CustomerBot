from MyChatbotData import MyChatbotData
from emoji import emoji_normalize, emoji_isolate, ascii_normalize
import re
import json  # Add this import statement for json

# Load training data
with open('./training_sample.json', 'r') as f:
    training_data = json.load(f)

# Define answers dictionary
answers = {
    "customer_service.say_hello": "Hello! Welcome to our customer service. How can I assist you today?",
    "customer_service.introduction": "Certainly! We offer a wide range of products, including computer accessories, electronics, and home and kitchen appliances. Our selection includes items such as USB cables for computers, HDMI cables for home theater and TV setups, and steam irons for household chores.",
    "customer_service.product_details": "I will provide you with detailed information about the specifications, features, materials, dimensions, and weight of the product. You will also be informed about the product's warranty policy.",
    "customer_service.shipping": "We offer a variety of shipping options, including expedited and free shipping. You will be notified about the expected delivery time and can track the status of your order.",
    "customer_service.returns": "We have a flexible returns policy. You can return the product within a certain timeframe and receive a full refund or exchange it for a different item.",
    "customer_service.support": "Our dedicated customer service team is available to assist you in resolving any issues related to your order or the product. You can reach out to us through various channels.",
    "customer_service.say_goodbye": "Thank you for your inquiry. It was a pleasure assisting you today. I hope I was able to provide the information you needed. Please feel free to reach out to us again if you have any other questions. Take care and have a wonderful rest of your day!"
}
# Compile punctuation regex
punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))

chatbot_data = MyChatbotData(training_data, 'patterns', answers)
UNK = "I don't know"

def exact_match(query):
    intents = chatbot_data.get_intents()
    for intent in intents:
        phrases = chatbot_data.get_phrases(intent)
        if query in phrases:
            return chatbot_data.get_answer(intent)
    return UNK

def preprocess(text):
    text = ascii_normalize(text) or text
    text = emoji_normalize(text) or text
    text = emoji_isolate(text) or text
    text = chatbot_data.remove_punctuation(text) or text  # Modify here
    return text

def rule_based_classifier(query):
    preprocessed_query = preprocess(query.lower())
    return exact_match(preprocessed_query)
