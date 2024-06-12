from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
from rule_based import rule_based_classifier
from MyChatbotData import MyChatbotData
from models.your_ml_model import MLClassifier

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes by adding this line

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

# Initialize Machine Learning classifier
ml_classifier = MLClassifier(chatbot_data)
ml_classifier.train()

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    # Rule-based classification
    response = rule_based_classifier(user_message)
    
    # If rule-based classifier can't classify, use ML classifier
    if response == "I don't know":
        # Use ML classifier to predict intent
        ml_intent = ml_classifier.predict(user_message)
        
        # Get the answer corresponding to the predicted intent
        ml_response = answers.get(ml_intent, {"text": "Sorry, I don't understand."})
        response = ml_response
    
    return jsonify([response])  # Make sure the response is wrapped in a list

# Run the server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
