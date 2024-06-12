import numpy as np
import pandas as pd
import re
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_amazon_data(file_path):
    # Load Amazon data from CSV file
    amazon_data = pd.read_csv(file_path, delimiter=',')
    
    # Add an incremental index column if it doesn't exist
    if 'index' not in amazon_data.columns:
        amazon_data['index'] = range(1, len(amazon_data) + 1)
    
    return amazon_data

def preprocess(text):
    # Function to preprocess text
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    return text

def recommend_product(user_message, amazon_data_path):
    # Load Amazon data
    amazon_data = load_amazon_data(amazon_data_path)
    
    # Selecting the relevant features for recommendation
    selected_features = ['category', 'about_product']
    
    # Replacing the null values with empty strings
    for feature in selected_features:
        amazon_data[feature] = amazon_data[feature].fillna('')
    
    # Combining all the selected features
    combined_features = amazon_data['category'] + ' ' + amazon_data['about_product']
    
    # Converting the text data to feature vectors
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    
    # Getting the similarity scores using cosine similarity
    similarity = cosine_similarity(feature_vectors)
    
    # Preprocess the product names in the dataset
    amazon_data['processed_product_name'] = amazon_data['product_name'].apply(preprocess)
    
    # Preprocess the user message
    user_message_processed = preprocess(user_message)
    
    # Create a list with all the processed product names given in the dataset
    list_of_all_titles = amazon_data['processed_product_name'].tolist()
    
    # Finding the close match for the product name given by the user using fuzzy matching
    close_match = process.extractOne(user_message_processed, list_of_all_titles)
    
    if close_match is None:
        return "No close match found."
    else:
        close_match_title = close_match[0]
        
        # Finding the index of the product with the title
        product_indices = amazon_data[amazon_data.processed_product_name == close_match_title]['index'].values
        if len(product_indices) == 0:
            return "No matching product index found."

        index_of_the_product = product_indices[0]
        
        # Getting a list of similar products
        similarity_score = list(enumerate(similarity[index_of_the_product]))
        print ('similarity_score ', )
        
        # Sorting the products based on their similarity score
        sorted_similar_products = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        # Prepare the response with product recommendations
        response_text = 'Products suggested for you: \n'
        for i, product in enumerate(sorted_similar_products[:2]):
            index = product[0]
            product_titles = amazon_data[amazon_data['index'] == index]['product_name'].values
            if len(product_titles) == 0:
                continue
            title_from_index = product_titles[0]
            response_text += f"{i + 1}. {title_from_index}\n"

        # Ensure the second product recommendation is on a new line
        response_text += "\n"

        return response_text

# Example usage:
# Ensure the correct file path is provided for your amazon data CSV
# print(recommend_product("Your favorite product name", "./path/to/amazon.csv"))
