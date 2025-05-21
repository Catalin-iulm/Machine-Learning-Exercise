import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin
import streamlit as st

# Genera i dati
def genera_dati():
    np.random.seed(42)
    x1 = np.random.normal(160, 10, 50)
    y1 = np.random.normal(4, 1, 50)

    x2 = np.random.normal(100, 15, 50)
    y2 = np.random.normal(10, 2, 50)

    x3 = np.random.normal(130, 10, 50)
    y3 = np.random.normal(5, 1, 50)

    x4 = np.random.normal(115, 12, 50)
    y4 = np.random.normal(13, 2, 50)

    X = np.concatenate([x1, x2, x3, x4])
    Y = np.concatenate([y1, y2, y3, y4])
    data = np.column_stack((X, Y))
    return data

# Esegue una sola iterazione di K-Means manualmente
def kmeans_iter(data, centroids):
    labels = pairwise_distances_argmin(data, centroids)
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(len(centroids))])
    return new_centroids, labels

# Plot con Streamlit per ogni iterazione
def plot_iteration(data, centroids, labels, iter_num):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['red', 'green', 'blue', 'purple']
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, c=colors[i], label=f'Cluster {i+1}', alpha=0.6)
        ax.scatter(centroids[i, 0], centroids[i, 1], c='black', marker='X', s=200, edgecolor='white')

    ax.set_title(f"Iterazione {iter_num}")
    ax.set_xlabel('Spesa Media Mensile (€)')
    ax.set_ylabel('Numero Acquisti Mensili')
    ax.legend()
    st.pyplot(fig)

# Streamlit UI
st.title("🔁 Visualizzazione delle Iterazioni di K-Means")

data = genera_dati()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Inizializzazione casuale dei centroidi
np.random.seed(0)
initial_centroids = data_scaled[np.random.choice(len(data_scaled), 4, replace=False)]

centroids = initial_centroids
for i in range(1, 4):  # 3 iterazioni
    centroids, labels = kmeans_iter(data_scaled, centroids)
    centroids_unscaled = scaler.inverse_transform(centroids)
    data_unscaled = scaler.inverse_transform(data_scaled)
    plot_iteration(data_unscaled, centroids_unscaled, labels, i)
