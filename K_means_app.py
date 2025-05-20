import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from io import BytesIO

# Configurazione pagina
st.set_page_config(page_title="Clustering Retail", layout="wide", page_icon="🛒")

# Titolo
st.title("🛒 Segmentazione Clienti Retail")
st.markdown("""
Tool interattivo per segmentare la clientela retail utilizzando algoritmi di clustering.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurazione")
    
    # Selezione algoritmo
    algoritmo = st.radio("Algoritmo:", ["K-Means", "DBSCAN"])
    
    # Selezione features
    features_disponibili = [
        "Età", "Reddito Annuale", "Visite Mensili", 
        "Spesa Media", "% Acquisti Online", 
        "Frequenza Acquisti", "Valore Carrello"
    ]
    features_selezionate = st.multiselect(
        "Seleziona features per il clustering:",
        features_disponibili,
        default=["Età", "Reddito Annuale", "Visite Mensili"],
        max_selections=5
    )
    
    if algoritmo == "K-Means":
        n_clusters = st.slider("Numero cluster:", 2, 10, 4)
        max_iter = st.slider("Massime iterazioni:", 1, 10, 5)
        
        # Visualizzazione evoluzione cluster
        if st.checkbox("Mostra evoluzione iterazioni"):
            n_iter_viz = st.slider("Numero iterazioni da visualizzare:", 1, max_iter, min(3, max_iter))
    else:
        eps = st.slider("Raggio (epsilon):", 0.1, 1.0, 0.5, step=0.05)
        min_samples = st.slider("Minimo campioni:", 2, 20, 5)
    
    riduzione_dim = st.selectbox("Riduzione dimensionale:", ["PCA", "t-SNE"])
    
    if st.button("🔍 Esegui Analisi"):
        st.experimental_rerun()

# Generazione dati simulati
@st.cache_data
def genera_dati():
    np.random.seed(42)
    n = 1500
    
    # Generiamo 4 cluster naturali con tutte le features
    dati = {
        "Età": np.round(np.concatenate([
            np.random.normal(loc=25, scale=3, size=n//4),
            np.random.normal(loc=40, scale=5, size=n//4),
            np.random.normal(loc=60, scale=7, size=n//4),
            np.random.normal(loc=35, scale=4, size=n//4)
        ])),
        "Reddito Annuale": np.concatenate([
            np.random.normal(loc=40000, scale=5000, size=n//4),
            np.random.normal(loc=70000, scale=8000, size=n//4),
            np.random.normal(loc=35000, scale=4000, size=n//4),
            np.random.normal(loc=90000, scale=10000, size=n//4)
        ]),
        "Visite Mensili": np.concatenate([
            np.random.poisson(15, size=n//4),
            np.random.poisson(8, size=n//4),
            np.random.poisson(5, size=n//4),
            np.random.poisson(20, size=n//4)
        ]),
        "Spesa Media": np.concatenate([
            np.random.normal(loc=50, scale=10, size=n//4),
            np.random.normal(loc=120, scale=25, size=n//4),
            np.random.normal(loc=35, scale=8, size=n//4),
            np.random.normal(loc=200, scale=40, size=n//4)
        ]),
        "% Acquisti Online": np.concatenate([
            np.random.uniform(0.7, 0.9, size=n//4),
            np.random.uniform(0.3, 0.5, size=n//4),
            np.random.uniform(0.1, 0.3, size=n//4),
            np.random.uniform(0.8, 1.0, size=n//4)
        ]),
        "Frequenza Acquisti": np.concatenate([
            np.random.poisson(8, size=n//4),
            np.random.poisson(4, size=n//4),
            np.random.poisson(2, size=n//4),
            np.random.poisson(12, size=n//4)
        ]),
        "Valore Carrello": np.concatenate([
            np.random.normal(loc=30, scale=5, size=n//4),
            np.random.normal(loc=80, scale=15, size=n//4),
            np.random.normal(loc=20, scale=4, size=n//4),
            np.random.normal(loc=150, scale=30, size=n//4)
        ]),
        "Genere": np.random.choice(["M", "F"], size=n),
        "Cluster Reale": np.repeat([1, 2, 3, 4], n//4)
    }
    
    return pd.DataFrame(dati)

df = genera_dati()

# Verifica che siano selezionate almeno 2 features
if len(features_selezionate) < 2:
    st.warning("Seleziona almeno 2 features per il clustering!")
    st.stop()

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(df[features_selezionate])

# Riduzione dimensionale
if riduzione_dim == "PCA":
    reducer = PCA(n_components=2)
    X_ridotto = reducer.fit_transform(X)
    var_spiegata = reducer.explained_variance_ratio_.sum() * 100
    metodo_riduzione = f"PCA (Varianza: {var_spiegata:.1f}%)"
else:
    reducer = TSNE(n_components=2, perplexity=30)
    X_ridotto = reducer.fit_transform(X)
    metodo_riduzione = "t-SNE"

# Clustering
if algoritmo == "K-Means":
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init='auto')
    labels = model.fit_predict(X)
    centers = scaler.inverse_transform(model.cluster_centers_)
    
    # Visualizzazione evoluzione iterazioni
    if 'n_iter_viz' in locals():
        st.subheader("Evoluzione dei Cluster durante le Iterazioni")
        
        # Esegui K-Means per ogni iterazione
        figs = []
        for i in range(1, n_iter_viz+1):
            model_temp = KMeans(n_clusters=n_clusters, max_iter=i, n_init=1, init='random')
            labels_temp = model_temp.fit_predict(X)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(X_ridotto[:, 0], X_ridotto[:, 1], c=labels_temp, cmap='viridis', alpha=0.6)
            ax.set_title(f"Iterazione {i}")
            ax.set_xlabel("Componente 1")
            ax.set_ylabel("Componente 2")
            plt.colorbar(scatter, ax=ax, label='Cluster')
            figs.append(fig)
            plt.close()
        
        # Mostra i grafici in colonne
        cols = st.columns(n_iter_viz)
        for i, (col, fig) in enumerate(zip(cols, figs), 1):
            with col:
                st.pyplot(fig)
                # Aggiungi pulsante per scaricare l'immagine
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=100)
                st.download_button(
                    label=f"Scarica iterazione {i}",
                    data=buf,
                    file_name=f"kmeans_iterazione_{i}.png",
                    mime="image/png"
                )
else:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    centers = None

# Calcolo metriche
if len(set(labels)) > 1 and -1 not in set(labels):
    silhouette = silhouette_score(X, labels)
else:
    silhouette = "N/A"

# Visualizzazione
tab1, tab2, tab3 = st.tabs(["📊 Risultati", "📈 Analisi", "❓ Guida"])

with tab1:
    st.header("Risultati Clustering")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Plot cluster
        fig = px.scatter(
            x=X_ridotto[:, 0], y=X_ridotto[:, 1], 
            color=labels.astype(str),
            title=f"Risultati Clustering ({metodo_riduzione})",
            labels={"x": "Componente 1", "y": "Componente 2", "color": "Cluster"},
            hover_data=df[features_selezionate]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Metriche")
        st.metric("Silhouette Score", 
                 f"{silhouette:.3f}" if silhouette != "N/A" else "N/A",
                 help="Valore tra -1 e 1, più alto è meglio")
        
        st.subheader("Distribuzione Cluster")
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        st.bar_chart(cluster_counts)
        
        st.write(f"Features utilizzate: {', '.join(features_selezionate)}")

with tab2:
    st.header("Analisi Dettagliata")
    
    df["Cluster"] = labels
    
    st.subheader("Statistiche per Cluster")
    st.dataframe(
        df.groupby("Cluster")[features_selezionate]
        .agg(["mean", "median", "std"])
        .style.background_gradient(cmap="Blues")
    )
    
    st.subheader("Distribuzione Features")
    feature = st.selectbox("Seleziona feature:", features_selezionate)
    fig = px.box(df, x="Cluster", y=feature, color="Cluster")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Guida all'Uso")
    
    st.markdown("""
    ### Come utilizzare questo strumento:
    1. **Seleziona le features** che vuoi usare per il clustering (max 5)
    2. **Scegli l'algoritmo**:
       - **K-Means**: per cluster sferici di dimensioni simili
       - **DBSCAN**: per cluster di forma arbitraria e rilevamento outlier
    3. **Regola i parametri** dell'algoritmo
    4. Clicca **Esegui Analisi** per vedere i risultati
    
    ### Interpretazione:
    - **Silhouette Score**: misura la qualità del clustering (valore ideale > 0.5)
    - I grafici mostrano la distribuzione dei dati dopo riduzione dimensionale
    - Le tabelle mostrano le caratteristiche medie di ogni cluster
    """)
    
    st.subheader("Descrizione Features")
    descrizioni = {
        "Età": "Età del cliente (arrotondata all'intero più vicino)",
        "Reddito Annuale": "Reddito annuale stimato in €",
        "Visite Mensili": "Numero di visite al negozio/mese",
        "Spesa Media": "Spesa media per visita in €",
        "% Acquisti Online": "Percentuale di acquisti fatti online",
        "Frequenza Acquisti": "Numero di acquisti mensili",
        "Valore Carrello": "Valore medio del carrello in €"
    }
    
    for feat in features_selezionate:
        st.markdown(f"**{feat}**: {descrizioni.get(feat, '')}")

# Footer
st.markdown("---")
st.markdown("Progetto di Data Science - Università IULM | [GitHub](https://github.com/)")
