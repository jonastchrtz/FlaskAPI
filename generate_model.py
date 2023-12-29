import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

# Load data
data = pd.read_csv('dataset.csv')
cleaned_data = data.drop(columns=['family_history_with_overweight', 'FAVC', 'FCVC', 'NObeyesdad', 'TUE'])

# Preprocessing
numeric_features = ['Age', 'Height', 'Weight', 'FAF', 'CH2O', 'NCP']
categorical_features = ['MTRANS', 'SCC', 'SMOKE', 'CAEC', 'CALC']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Apply PCA
pca = PCA(n_components=2)

# K-Means Clustering
kmeans = Pipeline(steps=[('preprocessor', preprocessor),
                         #('pca', pca),
                         ('cluster', KMeans(n_clusters=4))])

joblib.dump(kmeans, 'kmeans_model.joblib')

cluster_labels = kmeans.fit_predict(cleaned_data)
# Assuming 'kmeans' is your trained K-means model
centroids = kmeans.named_steps['cluster'].cluster_centers_

numeric_features_transformed = numeric_features  # StandardScaler doesn't change feature names
categorical_features_transformed = list(preprocessor.named_transformers_['cat'].get_feature_names_out())
all_feature_names = numeric_features_transformed + categorical_features_transformed
# Adjust the index below based on the number of categorical features after one-hot encoding
centroid_df = pd.DataFrame(centroids, columns=all_feature_names)

print(centroid_df)
centroid_df.to_csv("centroid.csv")

reduced_df = pd.DataFrame(kmeans.named_steps['pca'].transform(
    kmeans.named_steps['preprocessor'].transform(cleaned_data)), 
    columns=['PC1', 'PC2'])

reduced_df['Cluster'] = cluster_labels

# Different colors for clusters
colors = ['red', 'green', 'blue', 'purple', 'orange']  # Adjust based on the number of clusters

plt.figure(figsize=(10, 7))
for i in range(kmeans.named_steps['cluster'].n_clusters):
    cluster_data = reduced_df[reduced_df['Cluster'] == i]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], color=colors[i], label=f'Cluster {i}')

plt.title('Cluster Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()