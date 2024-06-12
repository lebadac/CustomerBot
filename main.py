from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors module
import json  # Add this import statement for json
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
    "customer_service.say_hello": "Hello! Welcome to our customer service. How can I assist you today?",
    "customer_service.introduction": "Certainly! We offer a wide range of products, including computer accessories, electronics, and home and kitchen appliances. Our selection includes items such as USB cables for computers, HDMI cables for home theater and TV setups, and steam irons for household chores",
    "customer_service.product_details": "I will provide you with detailed information about the specifications, features, materials, dimensions, and weight of the product. You will also be informed about the product's warranty policy.",
    "customer_service.shipping": "We offer a variety of shipping options, including expedited and free shipping. You will be notified about the expected delivery time and can track the status of your order.",
    "customer_service.returns": "We have a flexible returns policy. You can return the product within a certain timeframe and receive a full refund or exchange it for a different item.",
    "customer_service.support": "Our dedicated customer service team is available to assist you in resolving any issues related to your order or the product. You can reach out to us through various channels.",
    "customer_service.say_goodbye": "Thank you for your inquiry. It was a pleasure assisting you today. I hope I was able to provide the information you needed. Please feel free to reach out to us again if you have any other questions. Take care and have a wonderful rest of your day!😀 "
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
        ml_response = answers.get(ml_intent, "Sorry, I don't understand.")
        response = ml_response
    
    return jsonify([{"text": response}])


# Run the server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
