import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="Segmentazione Clienti Retail", page_icon="🛒")

# --- Titolo e Introduzione ---
st.title("🛒 Strumento di Segmentazione Clienti")
st.markdown("""
Applicazione per l'analisi dei clienti di supermercati tramite clustering. 
Identifica gruppi di clienti con comportamenti simili per strategie di marketing mirate.
""")

# --- Controlli Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    
    st.subheader("1. Generazione Dati")
    n_clienti = st.slider("Numero clienti simulati", 100, 2500, 1000, step=50)
    random_state = st.slider("Seed casuale", 0, 100, 42)
    rumore = st.slider("Livello di rumore (%)", 0, 30, 5)
    
    st.subheader("2. Selezione Algoritmo")
    algoritmo = st.radio("Algoritmo:", ["K-Means", "DBSCAN"], index=0)

    # Parametri di default
    n_clusters = 4
    max_iter = 10
    eps = 0.5
    min_samples = 5

    if algoritmo == "K-Means":
        n_clusters = st.slider("Numero cluster", 2, 10, 4)
        max_iter = st.slider("Massime iterazioni", 5, 15, 10)
        
    elif algoritmo == "DBSCAN":
        eps = st.slider("Raggio (epsilon)", 0.1, 1.0, 0.5, step=0.05)
        min_samples = st.slider("Minimo campioni", 2, 20, 5)
    
    st.subheader("3. Visualizzazione")
    riduzione_dim = st.selectbox("Riduzione dimensionalità", ["PCA", "t-SNE"])
    motore_grafico = st.selectbox("Motore grafico", ["Matplotlib", "Plotly"])
    
    st.markdown("---")
    if st.button("🔄 Esegui Analisi"):
        st.experimental_rerun()

# --- Generazione Dati ---
@st.cache_data
def genera_dati_clienti(n_clienti, rumore, random_state):
    np.random.seed(random_state)
    
    tipologie = {
        "Giovani Professionisti": {
            "età": (28, 5), "reddito": (55000, 12000), 
            "visite_online": (18, 5), "spesa_media": (45, 10),
            "organico": (0.35, 0.1), "sensibilità_sconti": (0.6, 0.15),
            "visite_negozio": (4, 2), "fedeltà_marca": (0.7, 0.1)
        },
        "Famiglie Economiche": {
            "età": (38, 6), "reddito": (45000, 8000),
            "visite_online": (8, 3), "spesa_media": (75, 15),
            "organico": (0.15, 0.08), "sensibilità_sconti": (0.9, 0.05),
            "visite_negozio": (12, 3), "fedeltà_marca": (0.4, 0.15)
        },
        "Clienti Premium": {
            "età": (45, 8), "reddito": (95000, 20000),
            "visite_online": (12, 4), "spesa_media": (120, 25),
            "organico": (0.5, 0.15), "sensibilità_sconti": (0.3, 0.1),
            "visite_negozio": (6, 2), "fedeltà_marca": (0.85, 0.08)
        }
    }
    
    dati = []
    clienti_per_tipo = n_clienti // len(tipologie)
    
    for tipo, parametri in tipologie.items():
        n = clienti_per_tipo
        
        età = np.random.normal(parametri["età"][0], parametri["età"][1], n)
        reddito = np.random.normal(parametri["reddito"][0], parametri["reddito"][1], n)
        visite_online = np.random.poisson(parametri["visite_online"][0], n)
        spesa_media = np.abs(np.random.normal(parametri["spesa_media"][0], parametri["spesa_media"][1], n))
        organico = np.clip(np.random.normal(parametri["organico"][0], parametri["organico"][1], n), 0, 1)
        sensibilità_sconti = np.clip(np.random.normal(parametri["sensibilità_sconti"][0], parametri["sensibilità_sconti"][1], n), 0, 1)
        visite_negozio = np.random.poisson(parametri["visite_negozio"][0], n)
        fedeltà_marca = np.clip(np.random.normal(parametri["fedeltà_marca"][0], parametri["fedeltà_marca"][1], n), 0, 1)
        
        maschera_rumore = np.random.random(n) < (rumore/100)
        
        if maschera_rumore.any():
            fattore_rumore = 1 + np.random.normal(0, 0.5, size=n)
            
            età[maschera_rumore] = età[maschera_rumore] * fattore_rumore[maschera_rumore]
            reddito[maschera_rumore] = reddito[maschera_rumore] * fattore_rumore[maschera_rumore]
            spesa_media[maschera_rumore] = np.abs(spesa_media[maschera_rumore] * fattore_rumore[maschera_rumore])
            organico[maschera_rumore] = np.clip(organico[maschera_rumore] * fattore_rumore[maschera_rumore], 0, 1)
        
        for i in range(n):
            genere = np.random.choice(["Maschio", "Femmina"], p=[0.45, 0.55])
            carta_fedeltà = np.random.choice([True, False], p=[0.7, 0.3])
            
            dati.append([
                max(18, min(80, int(età[i]))),
                max(20000, min(200000, int(reddito[i]))),
                max(0, int(visite_online[i]))),
                max(10, float(spesa_media[i]))),
                float(organico[i])),
                float(sensibilità_sconti[i])),
                max(0, int(visite_negozio[i]))),
                float(fedeltà_marca[i])),
                carta_fedeltà,
                tipo
            ])
    
    df = pd.DataFrame(dati, columns=[
        "Età", "Reddito Annuale ($)", "Visite Online Mensili", 
        "Spesa Media ($)", "% Acquisti Organici", 
        "Sensibilità agli Sconti", "Visite al Negozio Mensili", 
        "Indice Fedeltà alla Marca", "Carta Fedeltà", "Segmento Reale"
    ])
    
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    features = [
        "Età", "Reddito Annuale ($)", "Visite Online Mensili", 
        "Spesa Media ($)", "% Acquisti Organici", 
        "Sensibilità agli Sconti", "Visite al Negozio Mensili", 
        "Indice Fedeltà alla Marca"
    ]
    
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled, features

df, X_scaled, features = genera_dati_clienti(n_clienti, rumore, random_state)

# --- Riduzione Dimensionalità ---
@st.cache_data
def riduci_dimensionalita(X, metodo, random_state):
    if metodo == "PCA":
        reducer = PCA(n_components=2, random_state=random_state)
        ridotto = reducer.fit_transform(X)
        varianza_spiegata = reducer.explained_variance_ratio_.sum() * 100
        return ridotto, f"PCA (Varianza: {varianza_spiegata:.1f}%)"
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=random_state)
        ridotto = reducer.fit_transform(X)
        return ridotto, "t-SNE"

X_ridotto, metodo_riduzione = riduci_dimensionalita(X_scaled, riduzione_dim, random_state)

# --- Clustering ---
@st.cache_data
def esegui_clustering(X, algoritmo, parametri):
    if algoritmo == "K-Means":
        model = KMeans(
            n_clusters=parametri['n_clusters'],
            max_iter=parametri['max_iter'],
            random_state=42,
            n_init='auto'
        )
        labels = model.fit_predict(X)
        centers = model.cluster_centers_
    else:  # DBSCAN
        model = DBSCAN(
            eps=parametri['eps'],
            min_samples=parametri['min_samples']
        )
        labels = model.fit_predict(X)
        centers = None
    
    return labels, model, centers

parametri_algoritmo = {
    "K-Means": {'n_clusters': n_clusters, 'max_iter': max_iter},
    "DBSCAN": {'eps': eps, 'min_samples': min_samples}
}

labels, model, centers = esegui_clustering(X_scaled, algoritmo, parametri_algoritmo[algoritmo])

# --- Metriche di Valutazione ---
def calcola_metriche(X, labels):
    metriche = {}
    unique_labels = set(labels)
    n_cluster_reali = len(unique_labels) - (1 if -1 in unique_labels else 0)

    if n_cluster_reali > 1:
        try:
            metriche['Silhouette Score'] = silhouette_score(X, labels)
            metriche['Davies-Bouldin Index'] = davies_bouldin_score(X, labels)
            metriche['Calinski-Harabasz Index'] = calinski_harabasz_score(X, labels)
        except:
            pass
    
    unique, counts = np.unique(labels, return_counts=True)
    metriche['Numero Cluster'] = n_cluster_reali
    metriche['Punti Rumore'] = counts[unique == -1][0] if -1 in unique else 0
    
    return metriche

metriche = calcola_metriche(X_scaled, labels)

# --- Visualizzazione ---
def crea_grafico_cluster(X, labels, centers, metodo, motore, df_originale):
    plot_df = pd.DataFrame({
        "x": X[:, 0],
        "y": X[:, 1],
        "cluster": labels,
        "dimensione": df_originale["Spesa Media ($)"] / 10
    })
    
    if motore == "Plotly":
        fig = px.scatter(
            plot_df, x="x", y="y", color="cluster",
            size="dimensione", 
            title=f"Segmentazione Clienti ({metodo})",
            labels={"x": "Componente 1", "y": "Componente 2"}
        )
        
        if centers is not None:
            fig.add_scatter(
                x=centers[:, 0], y=centers[:, 1],
                mode="markers", marker=dict(size=12, color="black", symbol="x"),
                name="Centri Cluster"
            )
        
        return fig
    else:
        plt.figure(figsize=(10, 7))
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=df_originale["Spesa Media ($)"]/5)
        if centers is not None:
            plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=200, c="red")
        plt.title(f"Segmentazione Clienti ({metodo})")
        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
        return plt.gcf()

grafico_cluster = crea_grafico_cluster(
    X_ridotto, labels, centers, 
    metodo_riduzione, motore_grafico, df
)

# --- Profilazione Cluster ---
def profila_cluster(df, labels):
    df_clusterizzato = df.copy()
    df_clusterizzato["Cluster"] = labels
    profili = df_clusterizzato.groupby("Cluster").mean()
    return profili, df_clusterizzato

profili_cluster, df_clusterizzato = profila_cluster(df, labels)

# --- Interfaccia Principale ---
tab1, tab2 = st.tabs(["📊 Visualizzazione", "📈 Analisi"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if motore_grafico == "Plotly":
            st.plotly_chart(grafico_cluster, use_container_width=True)
        else:
            st.pyplot(grafico_cluster)
            plt.close()
    
    with col2:
        st.metric("Numero Cluster", metriche.get('Numero Cluster', 'N/A'))
        st.metric("Punti Rumore", metriche.get('Punti Rumore', 0))
        
        if 'Silhouette Score' in metriche:
            st.metric("Silhouette Score", f"{metriche['Silhouette Score']:.3f}")
        if 'Davies-Bouldin Index' in metriche:
            st.metric("Davies-Bouldin", f"{metriche['Davies-Bouldin Index']:.3f}")

with tab2:
    st.dataframe(profili_cluster.style.background_gradient(cmap="Blues"))
    
    feature_selezionata = st.selectbox("Seleziona feature", features)
    
    if motore_grafico == "Plotly":
        fig = px.box(df_clusterizzato, x="Cluster", y=feature_selezionata)
        st.plotly_chart(fig)
    else:
        plt.figure(figsize=(10, 5))
        df_clusterizzato.boxplot(column=feature_selezionata, by="Cluster")
        plt.suptitle("")
        st.pyplot(plt.gcf())
        plt.close()

# --- Footer ---
st.markdown("---")
st.markdown("*Strumento di Analisi Clienti* | Creato con Streamlit")
