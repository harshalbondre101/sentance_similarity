from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from scipy.spatial.distance import euclidean, cityblock
from nltk.metrics import edit_distance
#import PyPDF2
from math import sqrt
from functools import reduce
import pandas as pd
import numpy as np

# Jaro-Winkler Distance
def jaro_winkler_distance(s1, s2, p=0.1):
    def jaro_similarity(s1, s2):
        if not s1 or not s2:
            return 0.0
        match_distance = (max(len(s1), len(s2)) // 2) - 1
        matches = 0
        transpositions = 0
        for i, c1 in enumerate(s1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len(s2))
            for j in range(start, end):
                if s2[j] == c1:
                    matches += 1
                    if j < i:
                        transpositions += 1
                    break
        if matches == 0:
            return 0.0
        jaro_similarity = (
            (matches / len(s1)) + (matches / len(s2)) + ((matches - transpositions) / matches)
        ) / 3
        return jaro_similarity

    jaro_similarity_score = jaro_similarity(s1, s2)
    prefix_scale = 0
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix_scale += 1
        else:
            break
    return jaro_similarity_score + (prefix_scale * p * (1 - jaro_similarity_score))

# Sørensen-Dice Coefficient
def dice_coefficient(s1, s2):
    set1, set2 = set(s1), set(s2)
    intersection = len(set1 & set2)
    return (2.0 * intersection) / (len(set1) + len(set2))

# Mahalanobis Distance
def mahalanobis_distance(x, mean, cov):
    inv_cov = np.linalg.inv(cov)
    diff = x - mean
    return sqrt(np.dot(np.dot(diff, inv_cov), diff.T))

# Function to read data from a text file into a DataFrame with each sentence as a row and an index column
def read_text_file_to_dataframe(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()

        sentences = [line.strip() for line in data]
        df = pd.DataFrame({'Sentence': sentences})
        df['Index'] = df.index

        return df
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return pd.DataFrame()

# Function for character search using Counter
def character_search(input_sequence, df):
    input_sequence_lower = input_sequence.lower()
    
    # Generate results using list comprehension
    results = [
        {
            'Text': row['Sentence'],
            'Occurrences': row['Sentence'].lower().count(input_sequence_lower)
        }
        for _, row in df.iterrows()
    ]
    
    # Sort and return top 10 results by occurrences
    return sorted(results, key=lambda x: x['Occurrences'], reverse=True)[:10]




# Function for keyword search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def keyword_search(input_words, df):
    # Custom function to preprocess text
    def preprocess_text(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return text

    # Preprocess input words
    input_words_list = input_words.split()  # Split input words into a list
    input_words_processed = ' '.join(preprocess_text(word) for word in input_words_list)  # Preprocess individual words

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Sentence'])

    # Calculate TF-IDF vector for the input phrase
    input_tfidf = vectorizer.transform([input_words_processed])

    # Calculate cosine similarity between input phrase TF-IDF vector and corpus
    similarity_scores = cosine_similarity(input_tfidf, tfidf_matrix).flatten()

    # Combine similarity scores with the sentences
    results = [{'Text': text, 'Occurrences': score} for text, score in zip(df['Sentence'], similarity_scores)]

    # Sort results by similarity score in descending order
    return sorted(results, key=lambda x: x['Occurrences'], reverse=True)[:10]


# Function for context search using TF-IDF and Cosine Similarity

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def load_glove_model(glove_file):
    print("Loading GloVe embeddings...")
    word_embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            embedding = np.array(values[1:], dtype='float32')
            word_embeddings[word] = embedding
    print(f"Total {len(word_embeddings)} word vectors loaded.")
    return word_embeddings

# Function for context search using TF-IDF and GloVe-based Cosine Similarity
def context_search(input_sentence, df, glove_embeddings):
    # Custom function to handle GloVe word embeddings
    def get_word_embeddings(sentence):
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        word_vecs = [glove_embeddings[token] for token in tokens if token in glove_embeddings]
        return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(100)  # Adjust size as per GloVe dimensions

    # Preprocess input sentence
    input_sentence = re.sub(r'[^a-zA-Z0-9\s]', '', input_sentence.lower())

    # Initialize TF-IDF Vectorizer with tuned parameters
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=0.01, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['Sentence'])

    # Get TF-IDF vectors for input sentence and corpus
    input_tfidf = vectorizer.transform([input_sentence])
    corpus_tfidf = tfidf_matrix

    # Calculate GloVe-based Cosine Similarity
    input_embeddings = get_word_embeddings(input_sentence)
    corpus_embeddings = np.array([get_word_embeddings(sentence) for sentence in df['Sentence']])

    # Calculate cosine similarity scores between input and corpus embeddings
    similarity_scores = cosine_similarity([input_embeddings], corpus_embeddings).flatten()

    # Combine TF-IDF and GloVe-based Cosine Similarity scores
    combined_scores = (0.7 * similarity_scores) + (0.3 * input_tfidf.dot(corpus_tfidf.T).toarray().flatten())

    # Zip sentences with combined similarity scores
    results = [{'Text': text, 'Similarity': score} for text, score in zip(df['Sentence'], combined_scores)]

    # Sort results by similarity score in descending order
    return sorted(results, key=lambda x: x['Similarity'], reverse=True)[:10]



# Function for Cosine Similarity
def calculate_cosine_similarity(input_sentence, df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([input_sentence] + df['Sentence'].tolist())
    input_tfidf = tfidf_matrix[0]
    corpus_tfidf = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(input_tfidf, corpus_tfidf)
    results = [{'Text': text, 'Similarity': score} for text, score in zip(df['Sentence'], similarity_scores.flatten())]
    return sorted(results, key=lambda x: x['Similarity'], reverse=True)[:10]

# Function for Levenshtein Distance (Edit Distance) with Jaro-Winkler and Sørensen-Dice
def calculate_edit_distance_advanced(input_sentence, df):
    results = []
    for index, row in df.iterrows():
        edit_dist = edit_distance(input_sentence, row['Sentence'])
        jaro_winkler_dist = jaro_winkler_distance(input_sentence, row['Sentence'])
        dice_coeff = dice_coefficient(input_sentence, row['Sentence'])
        results.append({
            'Text': row['Sentence'],
            'EditDistance': edit_dist,
            'JaroWinklerDistance': jaro_winkler_dist,
            'DiceCoefficient': dice_coeff
        })
    return sorted(results, key=lambda x: (x['EditDistance'], -x['JaroWinklerDistance'], -x['DiceCoefficient']))[:10]

# Function for Jaccard Similarity with Sørensen-Dice
def calculate_jaccard_similarity_advanced(input_sentence, df):
    results = []
    for index, row in df.iterrows():
        jaccard_similarity = len(set(input_sentence.lower().split()).intersection(set(row['Sentence'].lower().split()))) / len(set(input_sentence.lower().split()).union(set(row['Sentence'].lower().split())))
        dice_coeff = dice_coefficient(input_sentence, row['Sentence'])
        results.append({
            'Text': row['Sentence'],
            'JaccardSimilarity': jaccard_similarity,
            'DiceCoefficient': dice_coeff
        })
    return sorted(results, key=lambda x: (x['JaccardSimilarity'], -x['DiceCoefficient']), reverse=True)[:10]

# Function for N-gram Similarity with Sørensen-Dice
def calculate_ngram_similarity_advanced(input_sentence, df):
    from nltk import ngrams
    input_words = input_sentence.lower().split()
    n = len(input_sentence)
    input_ngrams = set(ngrams(input_sentence.lower().split(), n))
    results = []
    for index, row in df.iterrows():
        text_ngrams = set(ngrams(row['Sentence'].lower().split(), n))
        if not input_ngrams and not text_ngrams:
            continue
        similarity = len(input_ngrams.intersection(text_ngrams)) / (len(input_ngrams.union(text_ngrams)) or 1)
        dice_coeff = dice_coefficient(input_sentence, row['Sentence'])
        results.append({
            'Text': row['Sentence'],
            'NGramSimilarity': similarity,
            'DiceCoefficient': dice_coeff
        })
    return sorted(results, key=lambda x: (x['NGramSimilarity'], -x['DiceCoefficient']), reverse=True)[:10]

# Word Embeddings Similarity Function (using GloVe)
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd

def calculate_word_embeddings_similarity(input_sentence, df, glove_embeddings):
    raw_text_data = df['Sentence'].tolist()

    input_tokens = input_sentence.lower().split()
    input_vector = np.mean([glove_embeddings[token] for token in input_tokens if token in glove_embeddings], axis=0)

    results = []
    for text in raw_text_data:
        text_tokens = text.lower().split()
        text_vector = np.mean([glove_embeddings[token] for token in text_tokens if token in glove_embeddings], axis=0)

        # Check if vectors contain at least one element
        if np.any(input_vector) and np.any(text_vector):
            # Check for array shapes equality
            if input_vector.shape == text_vector.shape:
                similarity = np.dot(input_vector, text_vector) / (np.linalg.norm(input_vector) * np.linalg.norm(text_vector))
                results.append({'Text': text, 'Similarity': similarity})

    return sorted(results, key=lambda x: x['Similarity'], reverse=True)[:10]



"""
# Function for reading data from a PDF file
def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return [text]
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return []
"""

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

def preprocess_text(text):
    # Convert text to lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def preprocess_dataframe(df):
    # Apply preprocessing to the 'Sentence' column in the DataFrame
    df['Sentence'] = df['Sentence'].apply(preprocess_text)
    return df

def print_all_results(user_input, df):
    # Preprocess user input
    user_input = preprocess_text(user_input)

    # Preprocess DataFrame
    df = preprocess_dataframe(df)

    character_results = character_search(user_input, df)
    keyword_results = keyword_search(user_input, df)
    context_results = context_search(user_input, df)
    cosine_results = calculate_cosine_similarity(user_input, df)
    edit_distance_results = calculate_edit_distance_advanced(user_input, df)
    jaccard_results = calculate_jaccard_similarity_advanced(user_input, df)
    ngram_results = calculate_ngram_similarity_advanced(user_input, df)

    print("Character Search Results:")
    display_results(character_results)
    print("\nKeyword Search Results:")
    display_results(keyword_results)
    print("\nContext Search Results:")
    display_results(context_results)
    print("\nCosine Similarity Results:")
    display_results(cosine_results)
    print("\nEdit Distance Results:")
    display_results(edit_distance_results)
    print("\nJaccard Similarity Results:")
    display_results(jaccard_results)
    print("\nN-gram Similarity Results:")
    display_results(ngram_results)
