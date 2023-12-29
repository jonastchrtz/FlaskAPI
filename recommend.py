import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

firebase_url = "https://socialactivity-app-default-rtdb.europe-west1.firebasedatabase.app/"

def get_all_users():
    return requests.get(firebase_url + "users.json").json()

data = pd.DataFrame.from_dict(get_all_users(), orient='index')

# Split the 'favouriteActivities' into a list of activities
data['favouriteActivities'] = data['favouriteActivities'].str.split(', ')

# One-hot encode these activities
# This creates a new DataFrame with binary columns for each sport
activities = data['favouriteActivities'].apply(lambda x: pd.Series(1, index=x)).fillna(0)

# Combine with the original data
data_with_dummies = data.join(activities)

# Assuming you only want to consider the one-hot encoded sports activities
# Extract the relevant columns (i.e., the one-hot encoded columns)
sports_columns = activities.columns
activity_vectors = data_with_dummies[sports_columns]

# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(activity_vectors)

# Convert to DataFrame for easier handling
similarity_df = pd.DataFrame(similarity_matrix, index=data['username'], columns=data['username'])

def get_similar_patients(username, similarity_df, top_n=5):
    # Get the similarity scores for the specific patient
    patient_similarity = similarity_df[username]
    
    # Sort the similarity scores in descending order
    most_similar_patients = patient_similarity.sort_values(ascending=False)
    
    # Get the top N most similar patients, excluding the patient themselves
    most_similar_patients = most_similar_patients[most_similar_patients.index != username]
    
    return most_similar_patients.head(top_n).index.tolist()

# Example usage
patient_id = "AlexJohnson"  # Replace with the actual patient ID
similar_patients = get_similar_patients(patient_id, similarity_df, top_n=5)

print(similar_patients)