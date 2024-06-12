import json
import csv
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from rule_based import rule_based_classifier
from MyChatbotData import MyChatbotData
from models.your_ml_model import MLClassifier

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes by adding this line

# Load training data from JSON file
def load_training_data_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load training data from CSV file
def load_training_data_from_csv(file_path):
    training_data = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            intent, pattern, category = row
            if intent in training_data:
                training_data[intent]['patterns'].append(pattern.strip())
            else:
                training_data[intent] = {
                    'intent': intent.strip(),
                    'patterns': [pattern.strip()],
                    'category': category.strip()
                }
    return training_data




# Paths to your files
json_file_path = './training_sample.json'
csv_file_path = './training_data.csv'

# Load training data from JSON and CSV files

json_data = load_training_data_from_json(json_file_path)
# print("Json_format:", json_data)  # Print the number of columns in CSV for verification
csv_data = load_training_data_from_csv(csv_file_path)
# print("csv_format:", csv_data)  # Print the number of rows in CSV for verification
# Kết hợp dữ liệu từ tệp JSON và tệp CSV
training_data = {**json_data, **csv_data}
# print("csv_format:", training_data) 
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
        "text": "🔍 I'll provide you with detailed information about the product's technical specifications, pictures, prices. You'll also be informed about our warranty policy."
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
        "text": "📞 Our customer support team is always ready to assist you in resolving any issues related to your order or product. You can contact us through multiple channels."
    },
    "cancel_order": {
        "text": "🙋‍♀️ I'd be happy to help you with cancelling your order. To cancel your last order, you can visit your account page on our website and look for the 'Cancel Order' or 'Order History' section. There, you should be able to find the order you want to cancel and select the option to cancel it. If you have any trouble finding this or need further assistance, please feel free to contact our customer support team, and they'll be glad to walk you through the cancellation process. We're here to help ensure you have a smooth shopping experience."
    },
    "change_order": {
        "text": "🔄 I apologize, but it seems like you are having trouble changing or modifying your order. Our order management system does allow you to make changes to your order, such as adding, removing, or substituting items, before it is shipped. To make changes to your existing order, please log into your account on our website and look for the 'Manage Order' or 'Edit Order' functionality. There, you should be able to view the details of your order and make any necessary modifications. If you are still having trouble, feel free to reach out to our customer service team and they will be happy to assist you with updating your order. We want to ensure you get exactly what you need, so please don't hesitate to let us know how we can help."
    },
    "change_shipping_address": {
        "text": "🚚 I understand you're having trouble updating your shipping address for your order. Our order management system does allow you to change your shipping address before the order is shipped. To do this, simply log into your account on our website and look for the 'Edit Order' or 'Change Shipping Address' option. There, you should be able to update the address details. If you're having any trouble finding this or need further assistance, please don't hesitate to reach out to our customer service team. They'll be happy to walk you through the steps to successfully change your shipping address. We want to make sure your order gets delivered to the right place, so please let us know how we can help."
    },
    "check_cancellation_fee": {
        "text": "🔍 To check the cancellation or termination fees associated with your order, I would suggest reviewing the terms and conditions or contacting our customer service team. The specific fees can vary depending on the product or service, and the stage of the order. Our representatives will be able to provide you with the exact cancellation charges that would apply to your order if you decide to cancel. Please feel free to reach out to them, and they'll be happy to look up the relevant details and walk you through the process. We want to ensure you have all the information you need to make an informed decision."
    },
    "check_invoice": {
        "text": "📋 To access your past invoices, you can typically log into your account on our website and look for the 'Invoices' or 'Order History' section. There, you should be able to view and download PDF copies of your previous orders and their associated invoices. If you're having trouble locating a specific invoice, our customer service team would be happy to assist you. Just provide them with details like the approximate purchase date or order number, and they can look up the invoice for you. Please let me know if you need any other help accessing your invoice information."
    },
    "check_payment_methods": {
        "text": "💳 You can find our accepted payment methods by checking the 'Payment' or 'Checkout' section of our website. There, you should see a list of all the payment options we currently support, such as credit/debit cards, digital wallets, bank transfers, and more. If you're having trouble locating this information, our customer service team will be happy to provide you with the details on our accepted payment methods. Please let us know if you need any assistance checking what payment options are available."
    },
    "check_refund_policy": {
        "text": "💰 To review our refund policy and understand the circumstances under which you may be eligible for a refund, I recommend checking the 'Returns' or 'Refunds' section of our website. There, you should find detailed information on our money-back guarantee, the timeframe for processing refunds, and the specific criteria that would qualify an order for a refund. If you're unable to locate this information online, our customer service team would be happy to provide you with the details of our refund policy. Please let us know if you have any other questions about getting a refund."
    },
    "complaint": {
        "text": "🤔 I apologize that you've had an unsatisfactory experience with our business. Our goal is always to provide excellent service and products to our customers. If you would like to file an official complaint, I recommend contacting our customer service department either by phone, email, or through the contact form on our website. They will be able to document your concerns and work with you to try to resolve the issue. Please let me know if there is any other way I can assist you."
    },
    "contact_customer_service": {
        "text": "📞 To find the available hours for our customer service team, I recommend checking the 'Contact Us' or 'Customer Support' section of our website. There, you should be able to see our customer service hours, as well as the various ways you can reach out to us, such as by phone, email, or live chat. If you're unable to locate this information online, you're welcome to call our main customer service number and one of our representatives will be happy to provide you with the current operating hours. Please let me know if you need any other assistance contacting our customer support."
    },
    "contact_human_agent": {
        "text": "🤖 To speak with one of our customer service agents, the quickest way is to call our main customer service phone number. Our agents are available during our standard business hours, which you can find listed on our website. Alternatively, you can try reaching out via our live chat feature, also available on our website during business hours. If you're having trouble locating the right contact information, please let me know and I'll do my best to guide you to the appropriate channel for speaking with a live agent. I'm happy to assist further, but cannot directly transfer you to a human agent from this interface."
    },
    "create_account": {
        "text": "🙂 To create a new account with our service, simply visit our website and look for the 'Sign Up' or 'Create Account' button or link. This will guide you through the registration process, which typically involves providing some basic personal and contact information. If you have any trouble or need assistance during the sign-up, feel free to reach out to our customer support team and they'll be happy to walk you through it step-by-step. Let me know if you have any other questions!"
    },
    "delete_account": {
        "text": "📝 To delete your account with our service, you can typically find the option to do so in your account settings or profile page. If you're unable to locate it there, you're welcome to reach out to our customer support team who can guide you through the account deletion process. They'll be able to confirm the necessary steps and ensure your account is properly closed. Please let me know if you need any assistance navigating the account deletion procedure."
    },
    "delivery_options": {
        "text": "🚚 To view the available delivery options for your order, simply go to the checkout page on our website. There you'll see a section that lists all the different shipping methods and timeframes we offer, along with the associated costs. Feel free to review the options and select the one that best suits your needs. If you have any trouble finding or understanding the delivery choices, our customer service team would be happy to provide more information and guidance. Just let me know if you need any assistance navigating the delivery options."
    },
    "delivery_period": {
        "text": "📦 To check the expected delivery timeline for your order, you can usually find that information on the order confirmation page or in your account's order history. There you'll see the estimated delivery date range based on the shipping method you selected. If you can't locate that detail, you're welcome to reach out to our customer support team and they'll be happy to look up the projected delivery period for your specific order. Just let us know the order number or other relevant details, and we'll provide that delivery timeline information."
    },
    "edit_account": {
        "text": "🔧 To edit the information associated with your account, simply log into your profile and look for the section to update your personal details. This is typically found in your account settings or under a 'My Account' or 'Edit Profile' area. From there, you can make any necessary changes to your name, contact information, password, or other relevant details. Once you've updated the information, be sure to save the changes. If you have any trouble navigating the account editing process, don't hesitate to reach out to our customer support team for assistance. They'll be glad to walk you through the steps to successfully modify the details on your user profile."
    },
    "get_invoice": {
        "text": "📋 To retrieve your past order invoices, you can typically access them through your online account. Look for a section labeled 'Orders' or 'Purchase History' where you should be able to view and download PDF copies of your invoices, even for purchases made months or years ago. If you're unable to locate your invoices there, you're welcome to reach out to our customer service team. Provide them with details like the approximate purchase date or order number, and they'll be happy to research your account and send over the requested invoice(s). Let me know if you need any other assistance accessing your past order documentation."
    },
    "get_refund": {
        "text": "💰 To initiate a refund, I'd recommend first reviewing the return/refund policy associated with your purchase. This information is typically provided during the checkout process or available on the company's website. If you're eligible for a refund, you can usually start the process by contacting the company's customer service team, either by phone, email, or through an online form. Be prepared to provide details like your order number, date of purchase, and reason for the refund request. The customer service representatives will then guide you through the necessary steps to successfully process the refund back to your original payment method. Let me know if you have any other questions about getting a refund - I'm happy to provide additional assistance."
    },
    "newsletter_subscription": {
        "text": "📰 To subscribe to our company newsletter, you can typically do so by visiting our website and looking for a 'Newsletter' or 'Sign Up' section, where you can enter your email address. Once subscribed, you'll begin receiving our latest updates and content.\n\nTo unsubscribe, there is usually an 'Unsubscribe' link included at the bottom of each newsletter email that will allow you to manage your subscription preferences and opt out. Alternatively, you can reach out to our customer service team for assistance with updating or canceling your newsletter subscription.\n\nPlease let me know if you need any further help accessing or modifying your newsletter subscription status. I'm happy to provide additional guidance."
    },
    "payment_issue": {
        "text": "💳 If you're experiencing any issues with your payments, I'm happy to help. The best way to report and resolve a payment problem is typically to reach out directly to our customer service team. You can contact them by phone, email, or through an online support form on our website. Provide them with details about the specific issue you're facing, such as the transaction date, order number, payment method used, and any error messages or problems you encountered. Our team will then review the details and work with you to identify the root cause and get the issue corrected. Please don't hesitate to get in touch - we're committed to ensuring a smooth payment experience and will do our best to resolve any problems you may be having.."
    },
    "place_order": {
        "text": "🛒 To make a purchase on our website, simply browse our selection of items, add the ones you'd like to buy to your cart, and then proceed to checkout. During checkout, you'll be prompted to provide your shipping information and choose a payment method. We accept a variety of payment options, including credit/debit cards, PayPal, and digital wallets. Once you complete the checkout process, your order will be processed and shipped to the address you provided. If you need any assistance navigating our website or completing your purchase, don't hesitate to reach out to our customer service team - they'll be happy to guide you through the ordering process step-by-step. Let me know if you have any other questions!"
    },
    "recover_password": {
        "text": "🔑 If you've forgotten your password, the best way to recover access to your account is to use our password reset functionality. On our website, look for a 'Forgot Password' or 'Reset Password' link, typically located near the login area. Click on that, and you'll be prompted to enter the email address associated with your account. We'll then send you instructions on how to reset your password securely. This usually involves creating a new password that you can then use to log back into your account. If you have any trouble with the password reset process, don't hesitate to reach out to our customer support team for assistance. They'll be glad to help you regain access to your account."
    },
    "registration_problems": {
        "text": "📝 I apologize for any difficulties you've encountered with the registration process. If you're experiencing issues signing up or notice any errors during registration, the best thing to do is to reach out to our customer support team. You can contact them via email, phone, or by submitting a support ticket on our website. Provide as much detail as possible about the specific problem you're facing, such as any error messages you're seeing, the steps you took leading up to the issue, and your contact information. Our support staff will review the details and work quickly to investigate and resolve the registration problem. We're committed to ensuring a smooth sign-up experience, so please don't hesitate to report any issues you run into. We're here to help get you successfully registered."
    },
    "review": {
        "text": "📋 We greatly appreciate you taking the time to provide feedback about our services. The best way to leave a review is to visit our website and look for a 'Leave a Review' or 'Submit Feedback' section, often found in the footer or customer support area. There you can fill out a short form to share your thoughts, comments, or rating of your experience with us. We carefully read and consider all feedback received, as it helps us continue improving our products and services for customers like yourself. Thank you in advance for your review - we value your input and look forward to hearing from you!"
    },
    "set_up_shipping_address": {
        "text": "📦 To set up a new or different shipping address, you can typically do so through your account settings on our website. Look for a section labeled 'Shipping Addresses' or something similar, where you'll be able to add, edit, or select a new address. Simply fill out the required fields with the new address details, and save the changes. If you're having trouble getting the new address to be accepted, double-check that you've entered all the information correctly, including the full street address, city, state/province, postal code, and country. If you continue to run into issues, don't hesitate to reach out to our customer support team - they'll be happy to assist you in successfully updating your shipping address in our system."
    },
    "switch_account": {
        "text": "🔒 To switch to a different user account, typically you would look for a 'Switch Account' or 'Change User' option, often located in the top right corner of the interface or under account settings. This will allow you to log out of the current account and log into a new one. Make sure you have the credentials (username and password) for the account you want to switch to. If you're having trouble finding where to switch accounts or are unable to access your desired account, I'd recommend reaching out to our customer support team for assistance. They can help guide you through the account switching process to get you logged into the correct profile."
    },
    "track_order": {
        "text": "📦 To track the status of your order, you can typically do so by logging into your account on our website and looking for an 'Order History' or 'Track Order' section. There, you should be able to enter your order number or other identifying details to view the current status and estimated delivery date of your shipment. If you're unable to locate this information in your account, you can also contact our customer service team and provide them with your order details - they'll be happy to look up the status of your order and give you an update on the expected arrival time. Just let me know if you need any assistance tracking a specific order."
    },
     "track_refund": {
        "text": "💳 To check the status of your refund, you can typically log into your account on our website and look for a section related to orders, payments, or refunds. There, you should be able to view the details of your refund request, including the date it was processed and the status. If you're unable to locate this information online, you can also contact our customer service team directly. Provide them with the details of your original order and refund request, and they'll be happy to look into the status of your refund and provide you with an update. Just let me know if you have any trouble accessing your refund information, and I'll do my best to assist you further."
    }, 
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
