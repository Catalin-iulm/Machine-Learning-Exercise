import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA, TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px

# Funzione per generare dati di esempio (simulazione di clienti)
def genera_dati_clienti(n_campioni=1000, n_caratteristiche=5, n_clusters=6, livello_rumore=0.05, stato_casuale=42):
    X, y = make_blobs(n_samples=n_campioni, n_features=n_caratteristiche, centers=n_clusters, cluster_std=livello_rumore, random_state=stato_casuale)
    df = pd.DataFrame(X, columns=[f'caratteristica_{i+1}' for i in range(n_caratteristiche)])
    df['cluster_vero'] = y
    return df

# Funzione per scalare le caratteristiche
def scala_caratteristiche(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop('cluster_vero', axis=1))
    return X_scaled, list(df.columns[:-1])

# Funzione per ridurre la dimensionalità
def riduci_dimensioni(X_scaled, metodo_riduzione='PCA', stato_casuale=42):
    if metodo_riduzione == 'PCA':
        riduzione = PCA(n_components=2, random_state=stato_casuale)
        X_ridotto = riduzione.fit_transform(X_scaled)
    elif metodo_riduzione == 't-SNE':
         # Per t-SNE, assicuriamoci che la perplexity sia inferiore al numero di campioni
        perplexity = min(30, len(X_scaled) - 1)
        riduzione = TSNE(n_components=2, random_state=stato_casuale, perplexity=perplexity)
        X_ridotto = riduzione.fit_transform(X_scaled)
    else:  # 'Caratteristiche Originali'
        return X_scaled, "Caratteristiche Originali"
    return X_ridotto, metodo_riduzione

# Funzione per eseguire il clustering
def esegui_clustering(X_scaled, algoritmo='K-Means', parametri_algoritmo=None, stato_casuale=42):
    if algoritmo == 'K-Means':
        n_clusters = parametri_algoritmo.get('n_clusters', 6)
        modello = KMeans(n_clusters=n_clusters, random_state=stato_casuale, n_init=10)
        etichette = modello.fit_predict(X_scaled)
        centri = modello.cluster_centers_
    elif algoritmo == 'DBSCAN':
        eps = parametri_algoritmo.get('eps', 0.5)
        min_samples = parametri_algoritmo.get('min_samples', 5)
        modello = DBSCAN(eps=eps, min_samples=min_samples)
        etichette = modello.fit_predict(X_scaled)
        centri = None  # DBSCAN non ha centri espliciti
    
    return etichette, modello, centri

# Funzione per calcolare le metriche di clustering
def calcola_metriche(X, etichette):
    # Filtra per evitare il calcolo delle metriche se c'è un solo cluster
    if len(set(etichette)) <= 1:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}

    silhouette = silhouette_score(X, etichette)
    calinski_harabasz = calinski_harabasz_score(X, etichette)
    davies_bouldin = davies_bouldin_score(X, etichette)
    return {"silhouette": silhouette, "calinski_harabasz": calinski_harabasz, "davies_bouldin": davies_bouldin}

# Interfaccia Streamlit
st.title('Esplorazione del Clustering dei Clienti')

# Sidebar per i parametri
with st.sidebar:
    st.header("Parametri di Simulazione")
    n_campioni = st.slider("Numero di clienti simulati", 500, 5000, 1000, step=100)
    livello_rumore = st.slider("Livello di rumore nei dati", 0.01, 0.50, 0.05, step=0.01)
    stato_casuale = st.number_input("Seed casuale", value=42)

    st.header("Parametri di Clustering")
    algoritmo = st.selectbox("Algoritmo di clustering", ["K-Means", "DBSCAN"])
    
    algo_params = {}
    if algoritmo == 'K-Means':
        algo_params['n_clusters'] = st.slider("Numero di cluster", 2, 10, 6)
    elif algoritmo == 'DBSCAN':
        algo_params['eps'] = st.slider("Epsilon (DBSCAN)", 0.1, 1.0, 0.5, step=0.1)
        algo_params['min_samples'] = st.slider("Min_samples (DBSCAN)", 2, 20, 5)

    dim_reduction = st.selectbox("Riduzione di dimensionalità", ["PCA", "t-SNE", "Caratteristiche Originali"], index=0)

# Genera e scala i dati
df, features = genera_dati_clienti(n_campioni, livello_rumore=livello_rumore, stato_casuale=stato_casuale)
X_scaled, features = scala_caratteristiche(df)

# Riduci la dimensionalità
X_ridotto, metodo_riduzione = riduci_dimensioni(X_scaled, dim_reduction, stato_casuale)

# Esegui il clustering
etichette, modello, centri = esegui_clustering(X_scaled, algoritmo, algo_params, stato_casuale)
df['etichetta_cluster'] = etichette

# Calcola le metriche
metriche = calcola_metriche(X_scaled, etichette)

# Visualizza i risultati
st.header("Visualizzazione dei Cluster")

if metodo_riduzione == "Caratteristiche Originali":
    fig = px.scatter_matrix(df, dimensions=features[:2], color='etichetta_cluster', title=f'Cluster con {algoritmo} (Caratteristiche Originali)')
else:
    df_ridotto = pd.DataFrame(X_ridotto, columns=['Dimensione_1', 'Dimensione_2'])
    df_ridotto['etichetta_cluster'] = df['etichetta_cluster']
    fig = px.scatter(df_ridotto, x='Dimensione_1', y='Dimensione_2', color='etichetta_cluster',
                     title=f'Cluster con {algoritmo} (Riduzione: {metodo_riduzione})',
                     labels={'Dimensione_1': 'Dimensione 1', 'Dimensione_2': 'Dimensione 2'})
    if centri is not None:  # Aggiungi i centri solo se disponibili (K-Means)
        fig.add_trace(px.scatter(x=centri[:, 0], y=centri[:, 1], color=[str(i) for i in range(len(centri))],
                                 size=[10]*len(centri), symbol='star',
                                 labels={'x': 'Dimensione 1', 'y': 'Dimensione 2'},
                                 ).data[0])

st.plotly_chart(fig, use_container_width=True)

st.header("Metriche di Clustering")
metriche_df = pd.DataFrame([metriche])
# Arrotonda tutte le colonne numeriche a 2 decimali
metriche_df = metriche_df.round(2)
st.dataframe(metriche_df)

# Mostra i dati di esempio
st.header("Dati di Esempio")
st.dataframe(df.head(10).round(2)) # Arrotonda i dati di esempio a 2 decimali
