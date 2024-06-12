import re
import json  # Add this import statement for json
from MyChatbotData import MyChatbotData
from emoji import emoji_normalize, emoji_isolate, ascii_normalize


# Initialize punct_re_escape globally
punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))

# Load training data
with open('./training_sample.json', 'r') as f:
    training_data = json.load(f)

# Define answers dictionary
answers = {
    "customer_service.say_hello": {
        "text": "👋 Hello! Welcome to our customer service. I'm happy to assist you today, how can I help you?",
        "image": "https://st.depositphotos.com/8684932/56984/v/1600/depositphotos_569844582-stock-illustration-cartoon-mascot-cereal-bowl-customer.jpg",
        "suggestions": ["What's problem with you?", "shipping", "product details", "return", "support"]
    },
    "customer_service.introduction": {
        "text": "🛒 We have a great selection of products like computer accessories, electronics, and home appliances. You can find items such as USB cables, HDMI cables, and steam irons in our product catalog."
    },
    "customer_service.product_details": {
        "text": "🔍 I'll provide you with detailed information about the product's technical specifications, features, materials, dimensions, and weight. You'll also be informed about our warranty policy."
    },
    "customer_service.shipping": {
        "text": "🚚 We offer various shipping options, including fast and free delivery. You'll be notified about the estimated delivery time and can track the status of your order."
    },
    "customer_service.returns": {
        "text": "We have a flexible returns policy. You can return the product within a certain timeframe and receive a full refund or exchange it for a different item."
    },
    "customer_service.support": {
        "text": "🔄 We have a flexible return policy. You can return the product within a certain timeframe and receive a full refund or exchange it for a different item."
    },
    "customer_service.say_goodbye": {
        "text": "T📞 Our customer support team is always ready to assist you in resolving any issues related to your order or product. You can contact us through multiple channels."
    }
}

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
    text = remove_punctuation(text) or text
    return text

def remove_punctuation(text):
    return punct_re_escape.sub('', text)

def rule_based_classifier(query):
    preprocessed_query = preprocess(query.lower())
    return exact_match(preprocessed_query)
