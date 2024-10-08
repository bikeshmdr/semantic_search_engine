import streamlit as st
import pandas as pd
import spacy 

from transformers import BertTokenizer, BertModel
from functions import word_level_search, preprocess_query

# Set the page configuration
#st.set_page_config(page_title="Your App Title", layout="wide")

import os
import pandas as pd

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the pickle file inside the 'streamlit' folder
file_path = os.path.join(current_dir, 'preprocessed_data.pkl')

# Now use this path to load the pickle file
df = pd.read_pickle(file_path)

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Try loading the model, if not found, download it
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not found, download it
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def display_results(query, df, tokenizer, model):
    # Preprocess query as per your model requirements
    preprocessed_query = preprocess_query(query, nlp)

    # Call the word_level_search function
    output = word_level_search(preprocessed_query, df['tokens'], df['embeddings'], tokenizer, model)

    # Iterate over the sorted top documents returned by process_and_sort_results
    for result in output:
        doc_index = result['document_index']  # Get the document index from the result
        avg_similarity = result['avg_similarity']  # Get the average similarity score (optional, for display)
        
        # Extract columns 0 to 6 from the DataFrame row using iloc
        document_info = df.iloc[doc_index, 1:7]  # Get the specific columns (1 to 6)
        
        # Display the document information in Streamlit
        st.markdown(f"**Document Index:** {doc_index}, **Average Similarity:** {avg_similarity:.3f}")
        st.table(document_info)  # Display the document information
        url = df.iloc[doc_index, 0]  # Extract the URL from the DataFrame
        # Display a clickable link with alias "Amazon"
        st.markdown(f"[Click the following link to visit the website: **Amazon**]({url})")
        st.write("---")  # Add a separator between results

# Streamlit app layout
st.title("Semantic Search Engine")

# Take text input from the user
user_input = st.text_input("Enter your search query:")

# Display results when a query is entered
if user_input:
    st.write(f"### Search Results for: {user_input}")
    display_results(user_input, df, tokenizer, model)
