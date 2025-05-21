import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Funzione per generare i dati sintetici
def genera_dati():
    np.random.seed(42)
    x1 = np.random.normal(loc=160, scale=10, size=50)
    y1 = np.random.normal(loc=4, scale=1, size=50)

    x2 = np.random.normal(loc=100, scale=15, size=50)
    y2 = np.random.normal(loc=10, scale=2, size=50)

    x3 = np.random.normal(loc=130, scale=10, size=50)
    y3 = np.random.normal(loc=5, scale=1, size=50)

    x4 = np.random.normal(loc=115, scale=12, size=50)
    y4 = np.random.normal(loc=13, scale=2, size=50)

    X = np.concatenate([x1, x2, x3, x4])
    Y = np.concatenate([y1, y2, y3, y4])
    data = pd.DataFrame({'SpesaMediaMensile': X, 'NumeroAcquisti': Y})
    return data

# App Streamlit
st.title("🎯 Segmentazione Clienti con K-Means")
st.write("Inserisci i dati del nuovo cliente per vedere a quale cluster appartiene.")

# Input utente
spesa = st.slider("Spesa media mensile (€)", min_value=50, max_value=200, value=120)
acquisti = st.slider("Numero di acquisti mensili", min_value=1, max_value=20, value=10)
nuovo_cliente = np.array([[spesa, acquisti]])

# Generazione e clustering dei dati
data = genera_dati()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(data_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Assegnazione nuovo cliente
nuovo_cliente_scaled = scaler.transform(nuovo_cliente)
cluster_nuovo_cliente = kmeans.predict(nuovo_cliente_scaled)[0]

# Visualizzazione
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['red', 'green', 'blue', 'purple']
for i in range(4):
    ax.scatter(data[data.Cluster == i]['SpesaMediaMensile'],
               data[data.Cluster == i]['NumeroAcquisti'],
               label=f'Cluster {i+1}', alpha=0.6, s=60, c=colors[i])

# Centroidi
ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black', label='Centroidi')

# Nuovo cliente
ax.scatter(nuovo_cliente[0, 0], nuovo_cliente[0, 1], c='orange', s=150, edgecolors='k', label='Nuovo Cliente')
ax.set_xlabel('Spesa Media Mensile (€)')
ax.set_ylabel('Numero Acquisti Mensili')
ax.set_title('Visualizzazione Clustering')
ax.legend()
st.pyplot(fig)

# Risultato finale
st.success(f"✅ Il nuovo cliente è stato assegnato al **Cluster {cluster_nuovo_cliente + 1}**.")
