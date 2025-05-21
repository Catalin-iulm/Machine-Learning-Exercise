import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin

# Setup
st.set_page_config(page_title="Segmentazione Clienti PetCare", layout="centered")
st.title("🐶 Segmentazione Clienti PetCare con K-Means Interattivo")
st.write("Modifica le iterazioni K-Means e analizza il comportamento di un nuovo cliente.")

# Funzioni
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
    return np.column_stack((X, Y))

def kmeans_iterativo(data, k, max_iter):
    np.random.seed(0)
    centroids = data[np.random.choice(len(data), k, replace=False)]
    for _ in range(max_iter):
        labels = pairwise_distances_argmin(data, centroids)
        centroids = np.array([data[labels == i].mean(axis=0) if len(data[labels == i]) > 0 else centroids[i] for i in range(k)])
    return centroids, labels

def nomi_cluster(label):
    mapping = {
        0: "Pet Lover Strategici",
        1: "Amici Animali Premium",
        2: "Cacciatori di Offerte",
        3: "Fanatici del Pet Care"
    }
    return mapping.get(label, f"Cluster {label + 1}")

# UI
n_iter = st.slider("Numero di Iterazioni K-Means", 1, 10, 3)
spesa = st.slider("Spesa media mensile (€)", 50, 200, 120)
acquisti = st.slider("Numero di acquisti mensili", 1, 20, 10)
nuovo_cliente = np.array([[spesa, acquisti]])

# Elaborazione dati
dati = genera_dati()
scaler = StandardScaler()
dati_scaled = scaler.fit_transform(dati)
nuovo_cliente_scaled = scaler.transform(nuovo_cliente)

# Clustering
centroids_scaled, labels = kmeans_iterativo(dati_scaled, k=4, max_iter=n_iter)
centroids = scaler.inverse_transform(centroids_scaled)
labels = pairwise_distances_argmin(dati_scaled, centroids_scaled)

# Assegnazione nuovo cliente
cluster_nuovo = pairwise_distances_argmin(nuovo_cliente_scaled, centroids_scaled)[0]
nome_cluster = nomi_cluster(cluster_nuovo)

# Visualizzazione
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['red', 'green', 'blue', 'purple']
for i in range(4):
    ax.scatter(dati[labels == i][:, 0], dati[labels == i][:, 1], 
               label=nomi_cluster(i), s=60, alpha=0.6, color=colors[i])
    ax.scatter(centroids[i, 0], centroids[i, 1], marker='X', s=200, c='black', edgecolor='white')

# Nuovo cliente
ax.scatter(nuovo_cliente[0, 0], nuovo_cliente[0, 1], c='orange', edgecolors='k', s=150, label='Nuovo Cliente')

ax.set_xlabel("Spesa Media Mensile (€)")
ax.set_ylabel("Numero di Acquisti Mensili")
ax.set_title(f"Risultato con {n_iter} Iterazioni")
ax.legend()
st.pyplot(fig)

# Output testuale
st.success(f"📍 Il nuovo cliente è stato assegnato al gruppo **{nome_cluster}**.")
