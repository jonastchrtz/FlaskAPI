import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, TransformerMixin

class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Wraps `MultiLabelBinarizer` in a form that can work with `ColumnTransformer`. Note
    that input X has to be a `pandas.DataFrame`.
    """
    def __init__(self):
        self.mlbs = list()
        self.n_columns = 0
        self.categories_ = self.classes_ = list()

    def fit(self, X:pd.DataFrame, y=None):
        for i in range(X.shape[1]): # X can be of multiple columns
            mlb = MultiLabelBinarizer()
            mlb.fit(X.iloc[:,i])
            self.mlbs.append(mlb)
            self.classes_.append(mlb.classes_)
            self.n_columns += 1
        return self

    def transform(self, X:pd.DataFrame):
        if self.n_columns == 0:
            raise ValueError('Please fit the transformer first.')
        if self.n_columns != X.shape[1]:
            raise ValueError(f'The fit transformer deals with {self.n_columns} columns '
                             f'while the input has {X.shape[1]}.'
                            )
        result = list()
        for i in range(self.n_columns):
            result.append(self.mlbs[i].transform(X.iloc[:,i]))

        result = np.concatenate(result, axis=1)
        return result
    
firebase_url = "https://socialactivity-app-default-rtdb.europe-west1.firebasedatabase.app/"

def get_all_users_df():
        json = requests.get(firebase_url + "users.json").json()
        return pd.DataFrame.from_dict(json, orient='index')

def find_most_similar_user(username):
    numeric_features = ['age', 'height', 'weight', 'physicalActivityFrequency', 'waterConsumption', 'mainMeals']
    categorical_features = ['transportation', 'calorieMonitoring', 'foodBetweenMainMeals', 'alcoholConsumption']

    kmeans_model = joblib.load('kmeans_model.joblib')

    data = get_all_users_df()
    data_old_features = data[numeric_features + categorical_features]
    data_old_features['waterConsumption'] = data_old_features['waterConsumption'].astype(int)

    cluster_labels = kmeans_model.fit_predict(data_old_features)
    data['cluster'] = cluster_labels

    # Check if the user exists
    if username not in data.index:
        return "User not found."

    # Get the cluster of the given user
    user_cluster = data.loc[username, 'cluster']

    # Filter data for users in the same cluster
    cluster_data = data[data['cluster'] == user_cluster]
    user_data = cluster_data.loc[[username]]

    # Preprocess the features for users in the cluster
    new_categorical_features = ['socialDirection']
    new_numerical_features = ['metabolicEquivalentTask']
    new_multilabel_features = ['favouriteActivities']
    new_features = new_categorical_features + new_multilabel_features + new_numerical_features

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), new_numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), new_categorical_features),
            ('mlb', MultiHotEncoder(), new_multilabel_features)
            ])
    
    X_user = preprocessor.fit_transform(user_data[new_features])
    X_cluster = preprocessor.transform(cluster_data[new_features])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(X_user, X_cluster)

    # Flatten the similarity scores and set the user's own score to -1
    similarity_scores = similarity_scores.flatten()
    user_index_in_cluster = cluster_data.index.get_loc(username)
    similarity_scores[user_index_in_cluster] = -1

    # Find the index of the most similar user
    most_similar_user_index = similarity_scores.argmax()

    # Retrieve the most similar user's username
    most_similar_user = cluster_data.iloc[most_similar_user_index].name
    
    if most_similar_user == username:
        most_similar_user = find_most_similar_user_out_of_cluster(username, data)

    return most_similar_user

def find_most_similar_user_out_of_cluster(username, data):
    # Check if the user exists
    if username not in data.index:
        return "User not found."

    user_data = data.loc[[username]]

    # Preprocess the features for users in the cluster
    new_categorical_features = ['socialDirection']
    new_numerical_features = ['metabolicEquivalentTask']
    new_multilabel_features = ['favouriteActivities']
    new_features = new_categorical_features + new_multilabel_features + new_numerical_features

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), new_numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), new_categorical_features),
            ('mlb', MultiHotEncoder(), new_multilabel_features)
            ])
    
    X_user = preprocessor.fit_transform(user_data[new_features])
    X_cluster = preprocessor.transform(data[new_features])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(X_user, X_cluster)

    # Flatten the similarity scores and set the user's own score to -1
    similarity_scores = similarity_scores.flatten()
    user_index_in_cluster = data.index.get_loc(username)
    similarity_scores[user_index_in_cluster] = -1

    # Find the index of the most similar user
    most_similar_user_index = similarity_scores.argmax()

    # Retrieve the most similar user's username
    most_similar_user = data.iloc[most_similar_user_index].name

    return most_similar_user