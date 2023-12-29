import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('dataset.csv')
cleaned_data = data.drop(columns=['family_history_with_overweight', 'FAVC', 'FCVC', 'NObeyesdad'])

# Preprocessing
numeric_features = ['Age', 'Height', 'Weight', 'FAF', 'TUE', 'CH2O', 'NCP']
categorical_features = ['MTRANS', 'SCC', 'SMOKE', 'CAEC', 'CALC']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

preprocessed_data = preprocessor.fit_transform(cleaned_data)

ssd = []
K = range(1, 15)  # Example range, adjust according to your dataset
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(preprocessed_data)
    ssd.append(km.inertia_)

# Plot SSD for each k
plt.figure(figsize=(10, 6))
plt.plot(K, ssd, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
#plt.show()

K = range(2, 15)  # Silhouette score is not defined for k=1

silhouette_scores = []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(preprocessed_data)
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(preprocessed_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, the average silhouette_score is : {silhouette_avg}")

# Plotting silhouette scores for each k
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Score For Optimal k')
plt.show()

# The optimal k is the one with the highest average silhouette score
optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
print(f"The optimal number of clusters k is: {optimal_k}")