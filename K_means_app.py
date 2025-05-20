import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Configurazione della pagina
st.set_page_config(page_title="K-Means Clustering", page_icon="📊", layout="wide")

# Titolo e descrizione
st.title("📊 K-Means Clustering Interattivo")
st.markdown("""
Questa app dimostra l'algoritmo K-Means per il clustering di dati. 
Puoi regolare i parametri nella sidebar e vedere come cambiano i risultati.
""")

# Sidebar per i parametri
with st.sidebar:
    st.header("⚙️ Parametri")
    n_samples = st.slider("Numero di campioni", 100, 1000, 300)
    n_features = st.slider("Numero di features", 2, 5, 3)
    n_centers = st.slider("Numero di cluster reali", 2, 6, 4)
    cluster_std = st.slider("Deviazione standard dei cluster", 0.1, 2.0, 1.0)
    max_k = st.slider("Massimo K da testare", 2, 10, 6)
    random_state = st.number_input("Random state", 0, 100, 42)
    
    st.markdown("---")
    st.markdown("👨‍💻 [Codice sorgente](https://github.com/...)")
    st.markdown("📚 [Documentazione K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)")

# Generazione dati
@st.cache_data
def generate_data(n_samples, n_features, n_centers, cluster_std, random_state):
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return X, y

X, y = generate_data(n_samples, n_features, n_centers, cluster_std, random_state)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sezione Elbow Method
st.header("1. Determinazione del numero ottimale di cluster")
st.markdown("Il metodo Elbow aiuta a scegliere il numero ottimale di cluster osservando dove l'inerzia inizia a diminuire meno rapidamente.")

inertia = []
for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(range(1, max_k + 1), inertia, marker='o', linestyle='--', color='#1f77b4')
ax1.set_title('Metodo Elbow')
ax1.set_xlabel('Numero di cluster (K)')
ax1.set_ylabel('Inerzia')
ax1.grid(True)
ax1.set_xticks(range(1, max_k + 1))
st.pyplot(fig1)

# Selezione del numero di cluster
optimal_k = st.slider("Seleziona il numero di cluster (K) da usare", 2, max_k, min(4, max_k))

# Esecuzione K-Means
kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init='auto')
cluster_labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# Visualizzazione risultati
st.header("2. Risultati del clustering")

# Seleziona tipo di visualizzazione
if n_features == 2:
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    scatter = ax2.scatter(
        X_scaled[:, 0], 
        X_scaled[:, 1], 
        c=cluster_labels, 
        cmap='viridis', 
        s=50, 
        alpha=0.7,
        edgecolor='w'
    )
    ax2.scatter(
        centroids[:, 0], 
        centroids[:, 1], 
        marker='X', 
        s=200, 
        color='red', 
        label='Centroidi',
        edgecolor='k'
    )
    ax2.set_xlabel('Feature 1 (scalata)')
    ax2.set_ylabel('Feature 2 (scalata)')
    ax2.legend()
    
elif n_features >= 3:
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    scatter = ax2.scatter(
        X_scaled[:, 0], 
        X_scaled[:, 1], 
        X_scaled[:, 2], 
        c=cluster_labels, 
        cmap='viridis', 
        s=50, 
        alpha=0.7,
        edgecolor='w'
    )
    ax2.scatter(
        centroids[:, 0], 
        centroids[:, 1], 
        centroids[:, 2], 
        marker='X', 
        s=200, 
        color='red', 
        label='Centroidi',
        edgecolor='k'
    )
    ax2.set_xlabel('Feature 1 (scalata)')
    ax2.set_ylabel('Feature 2 (scalata)')
    ax2.set_zlabel('Feature 3 (scalata)')
    ax2.legend()

plt.title(f'Clustering K-Means con K={optimal_k}')
st.pyplot(fig2)

# Metriche di valutazione
st.header("3. Valutazione del clustering")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Silhouette Score")
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    st.metric("Score", value=f"{silhouette_avg:.3f}")
    
    st.progress(min(max(0, (silhouette_avg + 1) / 2), 1.0))
    
    st.markdown("""
    **Interpretazione:**
    - Vicino a +1: cluster ben separati
    - Vicino a 0: cluster che si sovrappongono
    - Vicino a -1: punti probabilmente nel cluster sbagliato
    """)

with col2:
    st.subheader("Statistiche dei cluster")
    for i in range(optimal_k):
        cluster_size = np.sum(cluster_labels == i)
        st.write(f"**Cluster {i}**: {cluster_size} punti ({cluster_size/n_samples:.1%})")
    
    st.markdown("---")
    st.write("**Posizioni dei centroidi:**")
    st.dataframe(centroids, height=200)

# Spiegazione
st.header("4. Spiegazione dell'algoritmo")
with st.expander("Come funziona K-Means?"):
    st.markdown("""
    **K-Means** è un algoritmo di clustering che:
    1. Sceglie K centroidi iniziali (random o con K-Means++)
    2. Assegna ogni punto al centroide più vicino
    3. Ricalcola la posizione dei centroidi
    4. Ripete i passi 2-3 fino a convergenza
    
    **Parametri importanti:**
    - K: numero di cluster da trovare
    - Inizializzazione: come scegliere i centroidi iniziali
    - Criterio di convergenza: quando fermare l'algoritmo
    """)

with st.expander("Come interpretare i risultati?"):
    st.markdown("""
    1. **Metodo Elbow**: Cerca il "gomito" nel grafico per scegliere K
    2. **Silhouette Score**: Valuta la qualità della separazione dei cluster
    3. **Visualizzazione**: Verifica che i cluster siano ben separati nello spazio
    """)

# Footer
st.markdown("---")
st.markdown("App creata con ❤️ per il corso di Machine Learning")
