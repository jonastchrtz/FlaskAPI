import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

users = [1, 2, 3, 4, 5]

def recommend_users():
    return users