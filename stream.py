# Import necessary libraries
import streamlit as st
import pandas as pd
#import PyPDF2
import numpy as np
import re
from similarityfunction import *

# Example usage:
glove_file_path = 'glove.6B.100d.txt'  # Replace with the correct file path
glove_embeddings = load_glove_model(glove_file_path)


# Command-line user input function
def get_user_input():
    user_input = input("Enter a search term: ")
    user_input = preprocess_text(user_input)
    return user_input

# Display function for search results
def display_results(results):
    if not results:
        print("No results found.")
    else:
        for result in results:
            print(result)


# Define function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Define function to preprocess DataFrame
def preprocess_dataframe(df):
    df['Sentence'] = df['Sentence'].apply(preprocess_text)
    return df

# Define function to display results in a table
def display_table(results, title):
    st.subheader(title)
    if results:
        st.table(pd.DataFrame(results))
    else:
        st.write("No results found.")

# Define the main Streamlit app
def main():
    st.title("Similarity Search, Document Search and Comparison")
    uploaded_file = st.file_uploader("Upload a text file or PDF", type=['txt', 'pdf'])

    if uploaded_file is not None:
        file_contents = None
        if uploaded_file.type == 'text/plain':
            file_contents = uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.type == 'application/pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            file_contents = ""
            for page in pdf_reader.pages:
                file_contents += page.extract_text()

        if file_contents:
            #st.write("File contents:")
            #st.write(file_contents)

            user_input = st.text_input("Enter a search term:")
            if st.button("Search"):
                df = pd.DataFrame({'Sentence': file_contents.split('\n')})
                df['Index'] = df.index

                user_input = preprocess_text(user_input)
                df = preprocess_dataframe(df)

                character_results = character_search(user_input, df)
                keyword_results = keyword_search(user_input, df)
                context_results = context_search(user_input, df, glove_embeddings)
                cosine_results = calculate_cosine_similarity(user_input, df)
                edit_distance_results = calculate_edit_distance_advanced(user_input, df)
                jaccard_results = calculate_jaccard_similarity_advanced(user_input, df)
                ngram_results = calculate_ngram_similarity_advanced(user_input, df)
                word_em = calculate_word_embeddings_similarity(user_input, df, glove_embeddings)

                display_table(character_results, "Character Search Results")
                display_table(keyword_results, "Keyword Search Results")
                display_table(context_results, "Context Search Results")
                display_table(cosine_results, "Cosine Similarity Results")
                display_table(edit_distance_results, "Edit Distance Results")
                display_table(jaccard_results, "Jaccard Similarity Results")
                display_table(ngram_results, "N-gram Similarity Results")
                display_table(word_em, "Word embeddings Results")

if __name__ == "__main__":
    main()
