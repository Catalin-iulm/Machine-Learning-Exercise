import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configurazione pagina
st.set_page_config(page_title="Clustering per Retail", layout="wide", page_icon="🛒")

# Titolo
st.title("🛒 Analisi Cluster per Clienti Retail")
st.markdown("""
Applicazione per segmentare la clientela di un supermercato utilizzando algoritmi di clustering.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurazione")
    
    # Selezione algoritmo
    algoritmo = st.radio("Seleziona algoritmo:", ["K-Means", "DBSCAN"])
    
    if algoritmo == "K-Means":
        n_clusters = st.slider("Numero di cluster:", 2, 10, 4)
        max_iter = st.slider("Massime iterazioni:", 10, 100, 20)
    else:
        eps = st.slider("Raggio (epsilon):", 0.1, 1.0, 0.5, step=0.05)
        min_samples = st.slider("Minimo campioni:", 2, 20, 5)
    
    # Opzioni visualizzazione
    riduzione_dim = st.selectbox("Metodo riduzione dimensionale:", ["PCA", "t-SNE"])
    st.markdown("---")
    
    if st.button("🔍 Esegui Analisi"):
        st.experimental_rerun()

# Generazione dati simulati
@st.cache_data
def genera_dati():
    np.random.seed(42)
    n = 1000
    
    # Generiamo 4 cluster naturali
    cluster1 = np.random.normal(loc=[25, 40000, 15], scale=[3, 5000, 3], size=(n//4, 3))
    cluster2 = np.random.normal(loc=[40, 70000, 8], scale=[5, 8000, 2], size=(n//4, 3))
    cluster3 = np.random.normal(loc=[60, 35000, 5], scale=[7, 4000, 1], size=(n//4, 3))
    cluster4 = np.random.normal(loc=[35, 90000, 20], scale=[4, 10000, 4], size=(n//4, 3))
    
    dati = np.vstack([cluster1, cluster2, cluster3, cluster4])
    df = pd.DataFrame(dati, columns=["Età", "Reddito Annuale", "Visite Mensili"])
    
    # Aggiungiamo una colonna categorica
    df["Genere"] = np.random.choice(["M", "F"], size=n)
    df["Cluster Reale"] = np.repeat([1, 2, 3, 4], n//4)
    
    return df

df = genera_dati()

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(df[["Età", "Reddito Annuale", "Visite Mensili"]])

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
else:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    centers = None

# Calcolo metriche
if len(set(labels)) > 1:
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
else:
    silhouette = davies_bouldin = "N/A"

# Visualizzazione
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analisi", "❓ FAQ"])

with tab1:
    st.header("Risultati Clustering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot cluster
        fig = px.scatter(
            x=X_ridotto[:, 0], y=X_ridotto[:, 1], 
            color=labels.astype(str),
            title=f"Risultati Clustering ({metodo_riduzione})",
            labels={"x": "Componente 1", "y": "Componente 2", "color": "Cluster"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metriche
        st.subheader("Metriche di Performance")
        col_met1, col_met2 = st.columns(2)
        with col_met1:
            st.metric("Silhouette Score", f"{silhouette:.3f}" if silhouette != "N/A" else "N/A")
        with col_met2:
            st.metric("Davies-Bouldin", f"{davies_bouldin:.3f}" if davies_bouldin != "N/A" else "N/A")
    
    with col2:
        # Profili cluster
        df["Cluster"] = labels
        profili = df.groupby("Cluster").mean(numeric_only=True)
        st.subheader("Profili Cluster")
        st.dataframe(profili.style.background_gradient(cmap="Blues"))
        
        # Boxplot
        feature = st.selectbox("Seleziona feature:", ["Età", "Reddito Annuale", "Visite Mensili"])
        fig = px.box(df, x="Cluster", y=feature, color="Cluster")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Analisi Dettagliata")
    
    st.subheader("Dati Originali con Cluster")
    st.dataframe(df)
    
    st.subheader("Correlazioni tra Features")
    corr = df[["Età", "Reddito Annuale", "Visite Mensili"]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu")
    st.plotly_chart(fig, use_container_width=True)
    
    st.download_button(
        label="📥 Scarica Dati",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="clienti_clusterizzati.csv",
        mime="text/csv"
    )

with tab3:
    st.header("Domande Frequenti")
    
    st.markdown("""
    **1. Come funziona K-Means?**  
    K-Means divide i dati in K cluster cercando di minimizzare la varianza interna a ciascun cluster.
    È ideale per dati con cluster sferici di dimensioni simili.
    
    **2. Come funziona DBSCAN?**  
    DBSCAN identifica cluster come aree dense separate da aree meno dense. Non richiede di specificare
    il numero di cluster ed è robusto agli outlier.
    
    **3. Cosa misura il Silhouette Score?**  
    Misura quanto bene ogni punto si adatta al proprio cluster rispetto agli altri cluster.
    Valori vicini a 1 indicano una buona separazione tra cluster.
    
    **4. Come interpretare Davies-Bouldin?**  
    Un valore più basso indica una migliore separazione tra cluster. Idealmente dovrebbe essere < 1.
    """)
    
    st.subheader("Informazioni sul Progetto")
    st.markdown("""
    Questo strumento è stato sviluppato per dimostrare l'uso di algoritmi di clustering
    nell'analisi della clientela retail. I dati sono simulati ma rappresentano comportamenti
    realistici di clienti di supermercati.
    """)

# Footer
st.markdown("---")
st.markdown("Progetto di Analisi Cluster - Università IULM")
