import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo

print("Loading dataset...")

# LOAD DATA
dataset = fetch_ucirepo(id=555)
df = dataset.data.features.copy()

print("Original shape:", df.shape)

# Drop useless columns
drop_cols = [
    'title', 'body', 'address', 'cityname', 'source',
    'category', 'currency', 'amenities'
]
df = df.drop(columns=drop_cols, errors='ignore')

# Convert numeric columns
num_cols = ['bathrooms', 'bedrooms', 'square_feet',
            'price_display', 'latitude', 'longitude']

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing numeric
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Handle categorical
cat_cols = ['pets_allowed', 'has_photo', 'fee', 'price_type', 'state']
df[cat_cols] = df[cat_cols].fillna('Unknown')

# One-hot encoding
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("After encoding:", df.shape)

# Reduce dataset size to avoid freezing
df = df.sample(n=5000, random_state=42)

print("After sampling:", df.shape)

# Convert to numpy
df = df.select_dtypes(include=[np.number])
X = df.values

# K-MEANS (RAW DATA)
print("Non-numeric columns left:")
print(df.select_dtypes(include=['object']).columns)

print("Running K-Means...")

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

print("K-Means done")

# ELBOW METHOD (FAST VERSION)

print("Calculating Elbow...")

wcss = []
k_range = range(1, 6)  # reduced range for speed

for i in k_range:
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure()
plt.plot(k_range, wcss, marker='o')
plt.title("Elbow Method (Raw Data)")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

# SCALING

print("Scaling data...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-MEANS (SCALED)

kmeans_scaled = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_scaled = kmeans_scaled.fit_predict(X_scaled)

# PCA (FOR NICE VISUALIZATION)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_scaled, cmap='viridis', s=10)
plt.title("K-Means Clusters (PCA Visualization)")
plt.show()

# ELBOW (SCALED)

wcss_scaled = []

for i in k_range:
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss_scaled.append(km.inertia_)

plt.figure()
plt.plot(k_range, wcss_scaled, marker='o')
plt.title("Elbow Method (Scaled Data)")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

'''
1. What happens when K changes?

Small K → few large clusters (may miss patterns)
Large K → many small clusters (may overfit)
Goal: find optimal K (using Elbow method)

2. Why is scaling important?

K-Means uses distance
Large-value features (like price) dominate
Scaling makes all features equally important

3. How does K-Means find clusters?

Randomly choose K centroids
Assign points to nearest centroid
Recalculate centroids
Repeat until stable

'''