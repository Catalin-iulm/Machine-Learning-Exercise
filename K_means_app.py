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
import io
import warnings
warnings.filterwarnings('ignore')

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="Segmentazione Clienti Retail", page_icon="🛒")

# --- Titolo e Introduzione ---
st.title("🛒 Strumento di Segmentazione Clienti per Retail")
st.markdown("""
Questa applicazione permette di analizzare e segmentare la clientela di un supermercato utilizzando algoritmi di clustering. 
Scopri gruppi di clienti con comportamenti simili per ottimizzare le strategie di marketing.
""")

# --- Controlli Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    
    st.subheader("1. Generazione Dati")
    n_clienti = st.slider("Numero di clienti simulati", 500, 10000, 3000, step=100)
    random_state = st.slider("Seed casuale", 0, 100, 42)
    rumore = st.slider("Livello di rumore (%)", 0, 30, 5)
    
    st.subheader("2. Selezione Algoritmo")
    algoritmo = st.radio("Algoritmo di Clustering:", 
                         ["K-Means", "DBSCAN"], 
                         index=0)

    # Parametri di default
    n_clusters = 6
    init_method = "k-means++"
    max_iter = 300
    eps = 0.5
    min_samples = 10
    metric = "euclidean"

    if algoritmo == "K-Means":
        n_clusters = st.slider("Numero di cluster (K)", 2, 15, 6)
        init_method = st.selectbox("Metodo di inizializzazione", 
                                  ["k-means++", "random"])
        max_iter = st.slider("Massime iterazioni", 100, 500, 300)
        
    elif algoritmo == "DBSCAN":
        eps = st.slider("Raggio (epsilon)", 0.1, 2.0, 0.5, step=0.05)
        min_samples = st.slider("Minimo campioni", 1, 50, 10)
        metric = st.selectbox("Metrica di distanza", 
                               ["euclidean", "cosine", "manhattan"])
    
    st.subheader("3. Visualizzazione")
    riduzione_dim = st.selectbox("Riduzione dimensionalità", 
                                 ["PCA", "t-SNE", "Features Originali"])
    motore_grafico = st.selectbox("Motore grafico", 
                               ["Matplotlib", "Plotly"])
    
    st.markdown("---")
    if st.button("🔄 Esegui Analisi"):
        st.experimental_rerun()

# --- Generazione Dati ---
@st.cache_data
def genera_dati_clienti(n_clienti, rumore, random_state):
    np.random.seed(random_state)
    
    # Definizione di 6 tipologie di clienti
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
        },
        "Pensionati": {
            "età": (65, 5), "reddito": (40000, 10000),
            "visite_online": (4, 2), "spesa_media": (55, 12),
            "organico": (0.25, 0.1), "sensibilità_sconti": (0.7, 0.1),
            "visite_negozio": (8, 2), "fedeltà_marca": (0.6, 0.12)
        },
        "Appassionati di Salute": {
            "età": (35, 7), "reddito": (60000, 15000),
            "visite_online": (15, 4), "spesa_media": (65, 15),
            "organico": (0.75, 0.1), "sensibilità_sconti": (0.5, 0.15),
            "visite_negozio": (6, 2), "fedeltà_marca": (0.65, 0.12)
        },
        "Clienti Convenienza": {
            "età": (32, 8), "reddito": (48000, 10000),
            "visite_online": (25, 6), "spesa_media": (30, 8),
            "organico": (0.2, 0.1), "sensibilità_sconti": (0.8, 0.1),
            "visite_negozio": (2, 1), "fedeltà_marca": (0.3, 0.15)
        }
    }
    
    dati = []
    clienti_per_tipo = n_clienti // len(tipologie)
    
    for tipo, parametri in tipologie.items():
        n = clienti_per_tipo
        
        # Generazione dati
        età = np.random.normal(parametri["età"][0], parametri["età"][1], n)
        reddito = np.random.normal(parametri["reddito"][0], parametri["reddito"][1], n)
        visite_online = np.random.poisson(parametri["visite_online"][0], n)
        spesa_media = np.abs(np.random.normal(parametri["spesa_media"][0], parametri["spesa_media"][1], n))
        organico = np.clip(np.random.normal(parametri["organico"][0], parametri["organico"][1], n), 0, 1)
        sensibilità_sconti = np.clip(np.random.normal(parametri["sensibilità_sconti"][0], parametri["sensibilità_sconti"][1], n), 0, 1)
        visite_negozio = np.random.poisson(parametri["visite_negozio"][0], n)
        fedeltà_marca = np.clip(np.random.normal(parametri["fedeltà_marca"][0], parametri["fedeltà_marca"][1], n), 0, 1)
        
        # Aggiunta rumore
        maschera_rumore = np.random.random(n) < (rumore/100)
        
        if maschera_rumore.any():
            fattore_rumore = 1 + np.random.normal(0, 0.5, size=n)
            
            età[maschera_rumore] = età[maschera_rumore] * fattore_rumore[maschera_rumore]
            reddito[maschera_rumore] = reddito[maschera_rumore] * fattore_rumore[maschera_rumore]
            visite_online[maschera_rumore] = np.abs(visite_online[maschera_rumore] * fattore_rumore[maschera_rumore])
            spesa_media[maschera_rumore] = np.abs(spesa_media[maschera_rumore] * fattore_rumore[maschera_rumore])
            organico[maschera_rumore] = np.clip(organico[maschera_rumore] * fattore_rumore[maschera_rumore], 0, 1)
            sensibilità_sconti[maschera_rumore] = np.clip(sensibilità_sconti[maschera_rumore] * fattore_rumore[maschera_rumore], 0, 1)
            visite_negozio[maschera_rumore] = np.abs(visite_negozio[maschera_rumore] * fattore_rumore[maschera_rumore])
            fedeltà_marca[maschera_rumore] = np.clip(fedeltà_marca[maschera_rumore] * fattore_rumore[maschera_rumore], 0, 1)
        
        # Creazione record
        for i in range(n):
            genere = np.random.choice(["Maschio", "Femmina"], p=[0.45, 0.55])
            carta_fedeltà = np.random.choice([True, False], p=[0.7, 0.3])
            
            dati.append([
                max(18, min(80, int(età[i]))),
                genere,
                max(20000, min(200000, int(reddito[i]))),
                max(0, int(visite_online[i])),
                max(10, float(spesa_media[i])),
                float(organico[i]),
                float(sensibilità_sconti[i]),
                max(0, int(visite_negozio[i])),
                float(fedeltà_marca[i]),
                carta_fedeltà,
                tipo  # Segmento reale per valutazione
            ])
    
    # Creazione DataFrame
    df = pd.DataFrame(dati, columns=[
        "Età", "Genere", "Reddito Annuale ($)", 
        "Visite Online Mensili", "Spesa Media ($)",
        "% Acquisti Organici", "Sensibilità agli Sconti",
        "Visite al Negozio Mensili", "Indice Fedeltà alla Marca",
        "Carta Fedeltà", "Segmento Reale"
    ])
    
    # Mescola i dati
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Prepara le features per il clustering
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

# Genera i dati
df, X_scaled, features = genera_dati_clienti(n_clienti, rumore, random_state)

# --- Riduzione Dimensionalità ---
@st.cache_data
def riduci_dimensionalita(X, metodo, random_state):
    if metodo == "PCA":
        n_componenti = min(X.shape[1], 2)
        if n_componenti < 2:
            st.warning("Features insufficienti per PCA (minimo 2). Utilizzo la prima feature disponibile.")
            return X[:, :1], f"PCA (Componente 1)"
        reducer = PCA(n_components=n_componenti, random_state=random_state)
        ridotto = reducer.fit_transform(X)
        varianza_spiegata = reducer.explained_variance_ratio_.sum() * 100
        return ridotto, f"PCA (Varianza Spiegata: {varianza_spiegata:.1f}%)"
    elif metodo == "t-SNE":
        perplexity_val = min(30, len(X) - 1)
        if perplexity_val <= 1:
            st.warning("Campioni insufficienti per t-SNE (minimo 2). Utilizzo le features originali.")
            return X[:, :2], "Features Originali (Prime 2)"
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity_val)
        ridotto = reducer.fit_transform(X)
        return ridotto, "t-SNE"
    else:
        if X.shape[1] < 2:
            st.warning("Features insufficienti per plot 2D. Utilizzo la prima feature disponibile.")
            return X[:, :1], "Features Originali (Prima 1)"
        return X[:, :2], "Features Originali (Prime 2)"

X_ridotto, metodo_riduzione = riduci_dimensionalita(X_scaled, riduzione_dim, random_state)

# --- Clustering ---
@st.cache_data
def esegui_clustering(X, algoritmo, parametri, random_state):
    labels = np.array([])
    model = None
    centers = None

    if X.shape[0] == 0:
        st.warning("Nessun dato da clusterizzare. Aumenta il numero di clienti simulati.")
        return np.array([-1]*len(X)), None, None
    
    if algoritmo == "K-Means":
        if parametri['n_clusters'] >= X.shape[0]:
            st.warning(f"K-Means: Numero di cluster ({parametri['n_clusters']}) deve essere minore del numero di campioni ({X.shape[0]}). Impostato a 1 cluster.")
            labels = np.zeros(X.shape[0], dtype=int)
            centers = np.mean(X, axis=0).reshape(1, -1) if X.shape[0] > 0 else np.array([])
        else:
            model = KMeans(
                n_clusters=parametri['n_clusters'],
                init=parametri['init_method'],
                max_iter=parametri['max_iter'],
                random_state=random_state,
                n_init='auto'
            )
            labels = model.fit_predict(X)
            centers = model.cluster_centers_ if hasattr(model, 'cluster_centers_') else None
        
    elif algoritmo == "DBSCAN":
        if parametri['min_samples'] >= X.shape[0] or parametri['eps'] <= 0:
             st.warning(f"DBSCAN: Parametri non validi (min_samples={parametri['min_samples']}, eps={parametri['eps']}). Tutti i punti saranno rumore o assegnati a un singolo cluster.")
             labels = np.array([-1]*X.shape[0])
        else:
            model = DBSCAN(
                eps=parametri['eps'],
                min_samples=parametri['min_samples'],
                metric=parametri['metric']
            )
            labels = model.fit_predict(X)
            centers = None
    
    return labels, model, centers

# Prepara i parametri dell'algoritmo
parametri_algoritmo = {
    "K-Means": {
        'n_clusters': n_clusters,
        'init_method': init_method,
        'max_iter': max_iter
    },
    "DBSCAN": {
        'eps': eps,
        'min_samples': min_samples,
        'metric': metric
    }
}

labels, model, centers = esegui_clustering(
    X_scaled, 
    algoritmo, 
    parametri_algoritmo[algoritmo], 
    random_state
)

# --- Metriche di Valutazione ---
def calcola_metriche(X, labels):
    metriche = {}
    
    unique_labels = set(labels)
    n_cluster_reali = len(unique_labels) - (1 if -1 in unique_labels else 0)

    if n_cluster_reali > 1 and len(labels) > 1 and not all(l == -1 for l in labels):
        try:
            metriche['Silhouette Score'] = silhouette_score(X, labels)
        except Exception:
            metriche['Silhouette Score'] = np.nan
        
        try:
            metriche['Davies-Bouldin Index'] = davies_bouldin_score(X, labels)
        except Exception:
            metriche['Davies-Bouldin Index'] = np.nan
        
        try:
            metriche['Calinski-Harabasz Index'] = calinski_harabasz_score(X, labels)
        except Exception:
            metriche['Calinski-Harabasz Index'] = np.nan
    else:
        metriche['Silhouette Score'] = np.nan
        metriche['Davies-Bouldin Index'] = np.nan
        metriche['Calinski-Harabasz Index'] = np.nan
    
    # Conteggio cluster
    unique, counts = np.unique(labels, return_counts=True)
    metriche['Distribuzione Cluster'] = dict(zip(unique, counts))
    metriche['Numero Cluster'] = n_cluster_reali
    metriche['Punti Rumore'] = counts[unique == -1][0] if -1 in unique else 0
    
    return metriche

metriche = calcola_metriche(X_scaled, labels)

# --- Visualizzazione ---
def crea_grafico_cluster(X, labels, centers, metodo, motore, df_originale):
    if X.shape[1] < 2:
        st.warning(f"Metodo di riduzione '{metodo}' ha prodotto meno di 2 dimensioni. Impossibile creare un grafico 2D.")
        fig = plt.figure(figsize=(10, 7))
        plt.text(0.5, 0.5, "Dimensioni insufficienti per il grafico 2D", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
        plt.axis('off')
        return fig

    plot_df = pd.DataFrame({
        "x": X[:, 0],
        "y": X[:, 1],
        "cluster": labels,
        "dimensione": df_originale["Spesa Media ($)"].fillna(0).astype(float) / 10
    })
    
    plot_df['cluster_str'] = plot_df['cluster'].astype(str)

    if motore == "Plotly":
        fig = px.scatter(
            plot_df, x="x", y="y", color="cluster_str",
            size="dimensione", hover_data={
                "x": False, "y": False,
                "Età": df_originale["Età"],
                "Reddito": df_originale["Reddito Annuale ($)"],
                "Visite Online": df_originale["Visite Online Mensili"],
                "Spesa Media": df_originale["Spesa Media ($)"]
            },
            title=f"Segmentazione Clienti ({metodo})",
            labels={"x": "Componente 1", "y": "Componente 2"},
            color_discrete_map={'-1': 'grey'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        if centers is not None and centers.shape[1] >= 2:
            if metodo.startswith("PCA"):
                centers_df = pd.DataFrame({
                    "x": centers[:, 0],
                    "y": centers[:, 1],
                    "cluster": ["Centro"] * len(centers)
                })
                fig.add_scatter(
                    x=centers_df["x"], y=centers_df["y"],
                    mode="markers", marker=dict(size=12, color="black", symbol="x"),
                    name="Centri Cluster"
                )
        
        fig.update_layout(
            hovermode="closest",
            legend_title_text='Cluster'
        )
        
        return fig
    
    else:  # Matplotlib
        plt.figure(figsize=(10, 7))
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels) - (1 if -1 in unique_labels else 0)))
        
        color_map = {label: colors[i] for i, label in enumerate(unique_labels) if label != -1}
        if -1 in unique_labels:
            color_map[-1] = 'grey'

        mapped_colors = [color_map[label] for label in labels]

        scatter = plt.scatter(
            X[:, 0], X[:, 1], 
            c=mapped_colors, s=df_originale["Spesa Media ($)"].fillna(0).astype(float)/5,
            alpha=0.7
        )
        
        if centers is not None and centers.shape[1] >= 2:
            plt.scatter(
                centers[:, 0], centers[:, 1],
                marker="X", s=200, c="red", 
                edgecolors="black", linewidths=1.5,
                label="Centri Cluster"
            )
        
        plt.title(f"Segmentazione Clienti ({metodo})")
        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
        
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {l}',
                              markerfacecolor=color_map[l], markersize=10) 
                   for l in sorted(unique_labels) if l != -1]
        if -1 in unique_labels:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Rumore (-1)',
                                      markerfacecolor=color_map[-1], markersize=10))
        if centers is not None and centers.shape[1] >= 2:
            handles.append(plt.Line2D([0], [0], marker='X', color='w', label='Centri Cluster',
                                      markerfacecolor='red', markeredgecolor='black', markersize=12))

        plt.legend(handles=handles, title="Cluster")

        plt.grid(alpha=0.2)
        plt.tight_layout()
        return plt.gcf()

# Crea il grafico
grafico_cluster = crea_grafico_cluster(
    X_ridotto, labels, 
    centers if algoritmo == "K-Means" and riduzione_dim != "t-SNE" else None, 
    metodo_riduzione,
    motore_grafico,
    df
)

# --- Profilazione Cluster ---
def profila_cluster(df, labels, features):
    df_clusterizzato = df.copy()
    df_clusterizzato["Cluster"] = labels
    
    # Riassunto numerico
    profili_cluster = df_clusterizzato.groupby("Cluster")[features].agg(
        ["mean", "median", "std", "count"]
    )
    
    # Riassunto categorico
    if "Genere" in df.columns:
        if not df_clusterizzato.empty:
            distribuzione_genere = pd.crosstab(df_clusterizzato["Cluster"], df_clusterizzato["Genere"])
            somma_genere = distribuzione_genere.sum(1)
            distribuzione_genere_pct = distribuzione_genere.div(somma_genere, axis=0) if not somma_genere.empty else distribuzione_genere
            profili_cluster = pd.concat([profili_cluster, distribuzione_genere_pct], axis=1)
    
    if "Carta Fedeltà" in df.columns:
        if not df_clusterizzato.empty:
            distribuzione_fedeltà = pd.crosstab(df_clusterizzato["Cluster"], df_clusterizzato["Carta Fedeltà"])
            somma_fedeltà = distribuzione_fedeltà.sum(1)
            distribuzione_fedeltà_pct = distribuzione_fedeltà.div(somma_fedeltà, axis=0) if not somma_fedeltà.empty else distribuzione_fedeltà
            profili_cluster = pd.concat([profili_cluster, distribuzione_fedeltà_pct], axis=1)
    
    return profili_cluster, df_clusterizzato

profili_cluster, df_clusterizzato = profila_cluster(df, labels, features)

# --- Confronto con Segmenti Reali ---
def confronta_segmenti_reali(df_clusterizzato):
    if "Segmento Reale" not in df_clusterizzato.columns or df_clusterizzato.empty:
        return None, None
    
    df_filtrato = df_clusterizzato[df_clusterizzato["Cluster"] != -1]
    if df_filtrato.empty:
        return None, None

    confronto = pd.crosstab(
        df_filtrato["Cluster"],
        df_filtrato["Segmento Reale"],
        normalize="index"
    )
    
    purezza = confronto.max(axis=1).mean()
    
    return confronto, purezza

confronto_segmenti, punteggio_purezza = confronta_segmenti_reali(df_clusterizzato)

# --- Visualizzazione Principale ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visualizzazione", 
    "📈 Analisi Cluster",
    "📋 Esplora Dati",
    "📚 Documentazione"
])

with tab1:
    st.header("Visualizzazione Segmentazione Clienti")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if motore_grafico == "Plotly":
            st.plotly_chart(grafico_cluster, use_container_width=True)
        else:
            st.pyplot(grafico_cluster)
            plt.close()
    
    with col2:
        st.subheader("Metriche Algoritmo")
        
        st.metric("Numero di Cluster", metriche["Numero Cluster"])
        st.metric("Punti Rumore", metriche["Punti Rumore"])
        
        if not np.isnan(metriche['Silhouette Score']):
            st.metric("Silhouette Score", f"{metriche['Silhouette Score']:.3f}",
                      help="Valori più alti indicano cluster meglio definiti (-1 a 1)")
        else:
            st.metric("Silhouette Score", "N/A", help="Impossibile calcolare con meno di 2 cluster o solo punti rumore.")
            
        if not np.isnan(metriche['Davies-Bouldin Index']):
            st.metric("Indice Davies-Bouldin", f"{metriche['Davies-Bouldin Index']:.3f}",
                      help="Valori più bassi indicano clustering migliore (0 a ∞)")
        else:
            st.metric("Indice Davies-Bouldin", "N/A", help="Impossibile calcolare con meno di 2 cluster o solo punti rumore.")
            
        if not np.isnan(metriche['Calinski-Harabasz Index']):
            st.metric("Indice Calinski-Harabasz", f"{metriche['Calinski-Harabasz Index']:.3f}",
                      help="Valori più alti indicano clustering migliore (0 a ∞)")
        else:
            st.metric("Indice Calinski-Harabasz", "N/A", help="Impossibile calcolare con meno di 2 cluster o solo punti rumore.")
            
        if confronto_segmenti is not None and punteggio_purezza is not None:
            st.metric("Purezza Segmenti", f"{punteggio_purezza:.1%}",
                      help="Quanto bene i cluster corrispondono ai segmenti reali")
        else:
            st.metric("Purezza Segmenti", "N/A", help="Segmenti reali non disponibili o dati insufficienti.")

with tab2:
    st.header("Analisi Cluster")
    
    st.subheader("Caratteristiche Cluster")
    st.dataframe(profili_cluster[profili_cluster.index != -1].style.background_gradient(cmap="Blues"))
    
    st.subheader("Distribuzione Features per Cluster")
    feature_selezionata = st.selectbox("Seleziona feature da visualizzare", features)
    
    if motore_grafico == "Plotly":
        df_per_boxplot = df_clusterizzato[df_clusterizzato["Cluster"] != -1]
        if not df_per_boxplot.empty:
            fig = px.box(df_per_boxplot, x="Cluster", y=feature_selezionata, color="Cluster")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Nessun cluster valido per visualizzare le distribuzioni.")
    else:
        df_per_boxplot = df_clusterizzato[df_clusterizzato["Cluster"] != -1]
        if not df_per_boxplot.empty:
            plt.figure(figsize=(10, 5))
            df_per_boxplot.boxplot(column=feature_selezionata, by="Cluster", grid=False)
            plt.title(f"Distribuzione di {feature_selezionata} per Cluster")
            plt.suptitle("")
            st.pyplot(plt.gcf())
            plt.close()
        else:
            st.warning("Nessun cluster valido per visualizzare le distribuzioni.")
            plt.close()

    if confronto_segmenti is not None:
        st.subheader("Confronto con Segmenti Reali")
        st.dataframe(confronto_segmenti.style.background_gradient(cmap="Greens", axis=1))
        
        fig = px.imshow(confronto_segmenti, text_auto=".1%")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Esplora Dati")
    
    st.subheader("Dati Grezzi con Assegnazione Cluster")
    st.dataframe(df_clusterizzato)
    
    # Pulsante download
    csv = df_clusterizzato.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Scarica dati clusterizzati come CSV",
        data=csv,
        file_name="clienti_retail_clusterizzati.csv",
        mime="text/csv"
    )
    
    st.subheader("Correlazioni tra Features")
    matrice_correlazione = df_clusterizzato[features].corr()
    
    if motore_grafico == "Plotly":
        fig = px.imshow(matrice_correlazione, text_auto=".2f", color_continuous_scale="RdBu")
        st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(10, 8))
        plt.imshow(matrice_correlazione, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(features)), features, rotation=45)
        plt.yticks(range(len(features)), features)
        for i in range(len(features)):
            for j in range(len(features)):
                plt.text(j, i, f"{matrice_correlazione.iloc[i, j]:.2f}", 
                               ha="center", va="center", color="black")
        st.pyplot(plt.gcf())
        plt.close()

with tab4:
    st.header("Documentazione")
    
    st.subheader("Informazioni sullo Strumento")
    st.markdown("""
    Questo strumento interattivo permette agli analisti retail di:
    - Esplorare la segmentazione della clientela usando diversi algoritmi
    - Confrontare le performance degli algoritmi con metriche multiple
    - Profilare i segmenti di clienti in base alle loro caratteristiche
    - Scoprire pattern nascosti nel comportamento dei clienti
    
    **Funzionalità Principali:**
    - Due algoritmi di clustering (K-Means, DBSCAN)
    - Tecniche di riduzione dimensionale
    - Visualizzazioni interattive con Plotly
    - Profilazione completa dei cluster
    - Valutazione rispetto alla verità nota (quando disponibile)
    """)
    
    st.subheader("Descrizione Features")
    descrizioni_features = {
        "Età": "Età del cliente in anni",
        "Reddito Annuale ($)": "Reddito annuale stimato del nucleo familiare",
        "Visite Online Mensili": "Numero di visite al negozio online/app per mese",
        "Spesa Media ($)": "Spesa media per visita di acquisto",
        "% Acquisti Organici": "Percentuale di acquisti di prodotti biologici",
        "Sensibilità agli Sconti": "Probabilità di risposta agli sconti (scala 0-1)",
        "Visite al Negozio Mensili": "Numero di visite al negozio fisico per mese",
        "Indice Fedeltà alla Marca": "Preferenza per marche vs prodotti generici (scala 0-1)",
        "Carta Fedeltà": "Se il cliente possiede carta fedeltà",
        "Segmento Reale": "Segmento reale (solo in dati simulati)"
    }
    
    for feat, desc in descrizioni_features.items():
        st.markdown(f"**{feat}**: {desc}")
    
    st.subheader("Guida agli Algoritmi")
    st.markdown("""
    **K-Means**:
    - Ideale per cluster sferici di dimensioni simili
    - Richiede di specificare il numero di cluster
    - Sensibile agli outlier
    
    **DBSCAN**:
    - Trova cluster di forma arbitraria
    - Identifica punti rumore
    - Richiede la regolazione di epsilon e min_samples
    """)

# --- Footer ---
st.markdown("---")
st.markdown("""
    *Strumento di Analisi Retail* | Creato con Streamlit | 
    [Repository GitHub](https://github.com/your-repo)
""")
