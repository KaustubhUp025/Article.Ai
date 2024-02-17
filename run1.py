'''import numpy as np
import pandas as pd

import os
import math
import time


# Below libraries are for text processing using NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Below libraries are for feature representation using sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Below libraries are for similarity matrices using sklearn
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances

import nltk
nltk.download('wordnet')
nltk.download('stopwords')

df=pd.read_csv("C:/Users/hp/Documents/OpenHack/archive/data.csv")

# =========================================================================



# ==========================================================================

# Data Preprocessing
# Fill missing values
df.fillna("", inplace=True)

# Concatenate relevant text columns
df['combined_text'] = df['title'] + " " + df['description'] + " " + df['content']

# Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    processed_text = ""
    for word in text.split():
        word = "".join(e for e in word if e.isalnum())
        word = word.lower()
        if word not in stop_words:
            processed_text += lemmatizer.lemmatize(word, pos="v") + " "
    return processed_text.strip()

df['preprocessed_text'] = df['combined_text'].apply(preprocess_text)

# Headline based similarity on new articles
def recommend_article_based_on_input(user_input, num_similar_items=1):
    user_input = preprocess_text(user_input)
    w2v_user_input = np.zeros(300, dtype="float32")

    for word in user_input.split():
        if word in loaded_model:
            w2v_user_input = np.add(w2v_user_input, loaded_model[word])

    w2v_user_input = np.divide(w2v_user_input, len(user_input.split()))
    w2v_text = np.array([vec for vec in df['word2vec_vector']])
    couple_dist = pairwise_distances(w2v_text, w2v_user_input.reshape(1, -1))
    indices = np.argsort(couple_dist.ravel())[:num_similar_items]
    recommended_article = df.iloc[indices[0]]
    return recommended_article['title'], recommended_article['url']

# Example of usage
user_input = "Woman fired Trump's Motorcade"
headline, url = recommend_article_based_on_input(user_input)
print(f"{headline} :- {url}")'''

#2nd try

'''import numpy as np
import pandas as pd
import os
import math
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import gensim

import nltk
nltk.download('wordnet')
nltk.download('stopwords')

# Load Word2Vec model
model_path = "C:/Users/hp/Documents/OpenHack/word2vec_model.bin"  # Replace with the path to your Word2Vec model

loaded_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, encoding='latin1')


df=pd.read_csv("C:/Users/hp/Documents/OpenHack/archive/data.csv")

# Data Preprocessing
# Fill missing values
df.fillna("", inplace=True)

# Concatenate relevant text columns
df['combined_text'] = df['title'] + " " + df['description'] + " " + df['content']

# Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    processed_text = ""
    for word in text.split():
        word = "".join(e for e in word if e.isalnum())
        word = word.lower()
        if word not in stop_words:
            processed_text += lemmatizer.lemmatize(word, pos="v") + " "
    return processed_text.strip()

df['preprocessed_text'] = df['combined_text'].apply(preprocess_text)

# Split dataset into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Function to convert text to Word2Vec vectors
def text_to_word2vec(text):
    vector = np.zeros(300)
    for word in text.split():
        if word in loaded_model:
            vector += loaded_model[word]
    return vector / len(text.split())

train_df['word2vec_vector'] = train_df['preprocessed_text'].apply(text_to_word2vec)
test_df['word2vec_vector'] = test_df['preprocessed_text'].apply(text_to_word2vec)

# Headline based similarity on new articles
def recommend_article_based_on_input(user_input, num_similar_items=1):
    user_input = preprocess_text(user_input)
    w2v_user_input = np.zeros(300, dtype="float32")

    for word in user_input.split():
        if word in loaded_model:
            w2v_user_input = np.add(w2v_user_input, loaded_model[word])

    w2v_user_input = np.divide(w2v_user_input, len(user_input.split()))
    w2v_text = np.array([vec for vec in train_df['word2vec_vector']])
    couple_dist = pairwise_distances(w2v_text, w2v_user_input.reshape(1, -1))
    indices = np.argsort(couple_dist.ravel())[:num_similar_items]
    recommended_article = train_df.iloc[indices[0]]
    return recommended_article['title'], recommended_article['url']

# Example of usage
user_input = "Woman fired"
headline, url = recommend_article_based_on_input(user_input)
print(f"{headline} :- {url}")
'''


######################### 3rd try
'''import numpy as np
import pandas as pd
import os
import math
import time

# Below libraries are for text processing using NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Below libraries are for feature representation using sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Below libraries are for similarity matrices using sklearn
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances

import nltk
from gensim.models import Word2Vec

nltk.download('wordnet')
nltk.download('stopwords')

df = pd.read_csv("C:/Users/hp/Documents/OpenHack/archive/data.csv")

# =========================================================================
# Data Preprocessing
# Fill missing values
df.fillna("", inplace=True)

# Concatenate relevant text columns
df['combined_text'] = df['title'] + " " + df['description'] + " " + df['content']

# Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    processed_text = ""
    for word in text.split():
        word = "".join(e for e in word if e.isalnum())
        word = word.lower()
        if word not in stop_words:
            processed_text += lemmatizer.lemmatize(word, pos="v") + " "
    return processed_text.strip()

df['preprocessed_text'] = df['combined_text'].apply(preprocess_text)

# Load the Word2Vec model
model_path = 'C:/Users/hp/Documents/OpenHack/word2vec_model.bin'
loaded_model = Word2Vec.load(model_path)

# Headline based similarity on new articles
def recommend_article_based_on_input(user_input, num_similar_items=1):
    user_input = preprocess_text(user_input)
    w2v_user_input = np.zeros(100, dtype="float32")

    for word in user_input.split():
        if word in loaded_model.wv:
            w2v_user_input = np.add(w2v_user_input, loaded_model.wv[word])

    w2v_user_input = np.divide(w2v_user_input, len(user_input.split()))
    w2v_text = np.array([vec for vec in df['word2vec_vector']])
    couple_dist = pairwise_distances(w2v_text, w2v_user_input.reshape(1, -1))
    indices = np.argsort(couple_dist.ravel())[:num_similar_items]
    recommended_article = df.iloc[indices[0]]
    return recommended_article['title'], recommended_article['url']


def recommend_article_based_on_input(user_input, num_similar_items=1):
    user_input = preprocess_text(user_input)
    w2v_user_input = np.zeros(100, dtype="float32")

    for word in user_input.split():
        if word in model.wv:
            w2v_user_input = np.add(w2v_user_input, model.wv[word])

    w2v_user_input = np.divide(w2v_user_input, len(user_input.split()))
    
    # Generate vectors for articles
    df['word2vec_vector'] = df['preprocessed_text'].apply(lambda text: np.mean([model.wv[word] for word in text.split() if word in model.wv], axis=0))

    w2v_text = np.array([vec for vec in df['word2vec_vector']])
    couple_dist = pairwise_distances(w2v_text, w2v_user_input.reshape(1, -1))
    indices = np.argsort(couple_dist.ravel())[:num_similar_items]
    recommended_article = df.iloc[indices[0]]
    return recommended_article['title'], recommended_article['url']

# Example of usage
user_input = "Woman fired Trump's Motorcade"
headline, url = recommend_article_based_on_input(user_input)
print(f"{headline} :- {url}")'''


#  ================ 4th try

'''import numpy as np
import pandas as pd
import gensim
from sklearn.metrics.pairwise import pairwise_distances
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the Word2Vec model
model = gensim.models.Word2Vec.load('C:/Users/hp/Documents/OpenHack/word2vec_model.bin')

def preprocess_text(text):
    if not isinstance(text, str):  # Check if text is not a string
        return ''  # Return an empty string for non-string values
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(text)
    processed_text = ""
    for word in tokens:
        word = word.lower()
        if word not in stop_words:
            processed_text += lemmatizer.lemmatize(word, pos="v") + " "
    return processed_text.strip()
# Load dataset
df = pd.read_csv("C:/Users/hp/Documents/OpenHack/archive/data.csv")

# Concatenate relevant text columns
df['combined_text'] = df['title'] + " " + df['description'] + " " + df['content']

# Preprocess text
df['preprocessed_text'] = df['combined_text'].apply(preprocess_text)

# Function for recommending articles based on input
def recommend_article_based_on_input(user_input, model, df, num_similar_items):
    user_input = preprocess_text(user_input)
    if not user_input:
        return "No input provided", ""
    
    w2v_user_input = np.zeros(100, dtype="float32")

    for word in user_input.split():
        if word in model.wv:
            w2v_user_input = np.add(w2v_user_input, model.wv[word])

    w2v_user_input = np.divide(w2v_user_input, np.maximum(1, len(user_input.split())))
    
    # Generate vectors for articles
    df['word2vec_vector'] = df['preprocessed_text'].apply(lambda text: np.mean([model.wv[word] for word in text.split() if word in model.wv], axis=0))
    
    # Modify the code to handle NaN values
    df = df.dropna(subset=['word2vec_vector'])
    if df.empty:
        print("No vectors were generated for the articles.")
        return "", ""
    
    w2v_text = np.array(df['word2vec_vector'].tolist())

    couple_dist = pairwise_distances(w2v_text, w2v_user_input.reshape(1, -1))
    indices = np.argsort(couple_dist.ravel())[:num_similar_items]
    recommended_article = df.iloc[indices[0]]
    return recommended_article['title'], recommended_article['url']


user_input = "Budget 2024"
num_similar_items = 5
headline, url = recommend_article_based_on_input(user_input, model, df, num_similar_items)
print(f"{headline} :- {url}")
'''

# =================================== 5th try :-

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

# Load the Word2Vec model
from gensim.models import Word2Vec

# Load the Word2Vec model
loaded_model = Word2Vec.load('C:/Users/hp/Documents/OpenHack/word2vec_model.bin')


# Define preprocessing function
def preprocess_text(text):
    if pd.isna(text):  # Check if the value is NaN
        return ''      # Return an empty string if NaN
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(str(text))  # Convert to string
    processed_text = ""
    for word in tokens:
        word = word.lower()
        if word not in stop_words:
            processed_text += lemmatizer.lemmatize(word, pos="v") + " "
    return processed_text.strip()



# Load dataset
df = pd.read_csv("C:/Users/hp/Documents/OpenHack/archive/data.csv")

# Concatenate relevant text columns
df['combined_text'] = df['title'] + " " + df['description'] + " " + df['content']

# Preprocess text
df['preprocessed_text'] = df['combined_text'].apply(preprocess_text)

# Define the function for recommending articles based on input
'''def recommend_article_based_on_input(user_input, loaded_model, df, num_similar_items):
    vocabulary = loaded_model.wv.key_to_index
    w2v_headline = []
    for headline in df['title']:
        w2Vec_word = np.zeros(300, dtype="float32")
        for word in headline.split():
            if word in vocabulary:
                w2Vec_word = np.add(w2Vec_word, loaded_model.wv[word])
        w2Vec_word = np.divide(w2Vec_word, len(headline.split()))
        w2v_headline.append(w2Vec_word)
    w2v_headline = np.array(w2v_headline)

    # Calculate pairwise distances
    w2v_user_input = np.zeros(300, dtype="float32")
    for word in user_input.split():
        if word in vocabulary:
            w2v_user_input = np.add(w2v_user_input, loaded_model.wv[word])
    w2v_user_input = np.divide(w2v_user_input, len(user_input.split()))
    
    couple_dist = pairwise_distances(w2v_headline, w2v_user_input.reshape(1, -1))
    
    # Get indices of most similar headlines
    indices = np.argsort(couple_dist.ravel())[:num_similar_items]

    # Extract recommended headlines and URLs
    recommended_articles = []
    for index in indices:
        recommended_articles.append((df['title'][index], df['url'][index]))

    return recommended_articles'''

'''def recommend_article_based_on_input(user_input, loaded_model, df, num_similar_items):
    vocabulary = loaded_model.wv.key_to_index
    w2v_headline = []
    for headline in df['title'].astype(str):  # Convert to string explicitly
        w2Vec_word = np.zeros(300, dtype="float32")
        for word in headline.split():
            if word in vocabulary:
                w2Vec_word = np.add(w2Vec_word, loaded_model.wv[word])
        w2Vec_word = np.divide(w2Vec_word, len(headline.split()))
        w2v_headline.append(w2Vec_word)
    w2v_headline = np.array(w2v_headline)

    # Calculate pairwise distances
    w2v_user_input = np.zeros(300, dtype="float32")
    for word in user_input.split():
        if word in vocabulary:
            w2v_user_input = np.add(w2v_user_input, loaded_model.wv[word])
    w2v_user_input = np.divide(w2v_user_input, len(user_input.split()))
    
    couple_dist = pairwise_distances(w2v_headline, w2v_user_input.reshape(1, -1))
    
    # Get indices of most similar headlines
    indices = np.argsort(couple_dist.ravel())[:num_similar_items]

    # Extract recommended headlines and URLs
    recommended_articles = []
    for index in indices:
        recommended_articles.append((df['title'][index], df['url'][index]))

    return recommended_articles'''

from sklearn.feature_extraction.text import TfidfVectorizer

def recommend_article_based_on_input(user_input, loaded_model, df, num_similar_items):
    vocabulary = loaded_model.wv.key_to_index
    
    # Concatenate relevant text columns
    combined_text = df['title'] + " " + df['description'] + " " + df['content']
    
    # Calculate TF-IDF vectors for combined text
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text.astype(str))
    
    # Calculate TF-IDF vector for user input
    user_tfidf_vector = tfidf_vectorizer.transform([user_input])
    
    # Calculate cosine similarity between user input and combined text
    cosine_similarities = pairwise_distances(tfidf_matrix, user_tfidf_vector, metric='cosine').ravel()
    
    # Get indices of most similar articles
    indices = np.argsort(cosine_similarities)[:num_similar_items]

    # Extract recommended articles and URLs
    recommended_articles = []
    for index in indices:
        recommended_articles.append((df['title'][index], df['url'][index]))

    return recommended_articles



# Example of usage
user_input = "climate change news in the world"
num_similar_items = 5
recommended_articles = recommend_article_based_on_input(user_input, loaded_model, df, num_similar_items)
for article in recommended_articles:
    print(f"{article[0]} :- {article[1]}")

