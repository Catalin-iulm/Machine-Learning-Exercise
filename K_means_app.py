import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN # AgglomerativeClustering rimosso come richiesto
from sklearn.decomposition import PCA, TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px

# --- Configurazione Pagina Streamlit ---
st.set_page_config(layout="wide", page_title="Analisi Avanzata della Segmentazione Clienti", page_icon="🛒")

# --- Titolo e Introduzione ---
st.title("🛒 Analisi Avanzata della Segmentazione Clienti per il Retail")
st.markdown("""
Questa applicazione interattiva ti permette di esplorare algoritmi di clustering avanzati su dati simulati di clienti di supermercati.
Scopri segmenti di clienti nascosti e ottieni insight utili per strategie di marketing mirate.
""")

# --- Controlli della Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurazione dell'Esperimento")

    st.subheader("1. Generazione Dati")
    # Numero massimo di clienti ridotto a 5000 come richiesto
    n_campioni = st.slider("Numero di clienti simulati", 500, 5000, 1000, step=100)
    livello_rumore = st.slider("Livello di rumore (%)", 0, 30, 5)
    stato_casuale = st.slider("Seed casuale per la riproducibilità", 0, 100, 42)

    st.subheader("2. Selezione Algoritmo")
    # Opzione "Gerarchico" rimossa come richiesto
    algoritmo = st.radio("Algoritmo di Clustering:",
                         ["K-Means", "DBSCAN"],
                         index=0) # Default a K-Means

    # MODIFICA: Inizializza tutte le variabili dei parametri degli algoritmi con valori di default.
    # Questo assicura che le variabili esistano sempre, anche se i loro widget non sono visualizzati.
    n_clusters = 6 # Default per K-Means
    init_method = "k-means++" # Default per K-Means
    max_iter = 300 # Default per K-Means

    eps = 0.5 # Default per DBSCAN
    min_samples = 10 # Default per DBSCAN
    metric = "euclidean" # Default per DBSCAN

    if algoritmo == "K-Means":
        n_clusters = st.slider("Numero di cluster (K)", 2, 15, 6)
        init_method = st.selectbox("Metodo di inizializzazione",
                                  ["k-means++", "random"])
        max_iter = st.slider("Max iterazioni", 100, 500, 300)

    elif algoritmo == "DBSCAN":
        eps = st.slider("Epsilon (raggio di vicinato)", 0.1, 2.0, 0.5, step=0.05)
        min_samples = st.slider("Campioni minimi", 1, 50, 10)
        metric = st.selectbox("Metrica di distanza",
                               ["euclidean", "cosine", "manhattan"])

    st.subheader("3. Visualizzazione")
    dim_reduction = st.selectbox("Riduzione di dimensionalità",
                                 ["PCA", "t-SNE", "Caratteristiche Originali"])
    plot_engine = st.selectbox("Motore di plotting",
                               ["Matplotlib", "Plotly"])

    st.markdown("---")
    if st.button("🔄 Esegui Analisi"):
        st.experimental_rerun()

# --- Generazione Dati ---
@st.cache_data
def genera_dati_retail(n_campioni, livello_rumore, stato_casuale):
    np.random.seed(stato_casuale)

    # Definisci 6 archetipi di clienti con comportamenti di retail più realistici
    archetipi = {
        "Giovani Professionisti Urbani": {
            "eta": (28, 5), "reddito": (55000, 12000),
            "visite_online_mensili": (18, 5), "dimensione_cestino_media": (45, 10),
            "pct_acquisti_bio": (0.35, 0.1), "sensibilita_sconto": (0.6, 0.15),
            "visite_negozio_mensili": (4, 2), "lealta_brand": (0.7, 0.1)
        },
        "Famiglie con Budget Limitato": {
            "eta": (38, 6), "reddito": (45000, 8000),
            "visite_online_mensili": (8, 3), "dimensione_cestino_media": (75, 15),
            "pct_acquisti_bio": (0.15, 0.08), "sensibilita_sconto": (0.9, 0.05),
            "visite_negozio_mensili": (12, 3), "lealta_brand": (0.4, 0.15)
        },
        "Acquirenti Premium": {
            "eta": (45, 8), "reddito": (95000, 20000),
            "visite_online_mensili": (12, 4), "dimensione_cestino_media": (120, 25),
            "pct_acquisti_bio": (0.5, 0.15), "sensibilita_sconto": (0.3, 0.1),
            "visite_negozio_mensili": (6, 2), "lealta_brand": (0.85, 0.08)
        },
        "Coppie in Pensione": {
            "eta": (65, 5), "reddito": (40000, 10000),
            "visite_online_mensili": (4, 2), "dimensione_cestino_media": (55, 12),
            "pct_acquisti_bio": (0.25, 0.1), "sensibilita_sconto": (0.7, 0.1),
            "visite_negozio_mensili": (8, 2), "lealta_brand": (0.6, 0.12)
        },
        "Appassionati di Salute": {
            "eta": (35, 7), "reddito": (60000, 15000),
            "visite_online_mensili": (15, 4), "dimensione_cestino_media": (65, 15),
            "pct_acquisti_bio": (0.75, 0.1), "sensibilita_sconto": (0.5, 0.15),
            "visite_negozio_mensili": (6, 2), "lealta_brand": (0.65, 0.12)
        },
        "Acquirenti di Convenienza": {
            "eta": (32, 8), "reddito": (48000, 10000),
            "visite_online_mensili": (25, 6), "dimensione_cestino_media": (30, 8),
            "pct_acquisti_bio": (0.2, 0.1), "sensibilita_sconto": (0.8, 0.1),
            "visite_negozio_mensili": (2, 1), "lealta_brand": (0.3, 0.15)
        }
    }

    data = []
    campioni_per_tipo = n_campioni // len(archetipi)

    for nome_archetipo, params in archetipi.items():
        n = campioni_per_tipo

        # Genera i dati del cluster principale
        eta = np.random.normal(params["eta"][0], params["eta"][1], n)
        reddito = np.random.normal(params["reddito"][0], params["reddito"][1], n)
        visite_online_mensili = np.random.poisson(params["visite_online_mensili"][0], n)
        dimensione_cestino_media = np.abs(np.random.normal(params["dimensione_cestino_media"][0], params["dimensione_cestino_media"][1], n))
        pct_acquisti_bio = np.clip(np.random.normal(params["pct_acquisti_bio"][0], params["pct_acquisti_bio"][1], n), 0, 1)
        sensibilita_sconto = np.clip(np.random.normal(params["sensibilita_sconto"][0], params["sensibilita_sconto"][1], n), 0, 1)
        visite_negozio_mensili = np.random.poisson(params["visite_negozio_mensili"][0], n)
        lealta_brand = np.clip(np.random.normal(params["lealta_brand"][0], params["lealta_brand"][1], n), 0, 1)

        # Aggiungi un po' di rumore
        maschera_rumore = np.random.random(n) < (livello_rumore/100)

        if maschera_rumore.any():
            temp_eta = eta[maschera_rumore].copy()
            temp_reddito = reddito[maschera_rumore].copy()
            temp_visite_online_mensili = visite_online_mensili[maschera_rumore].copy()
            temp_dimensione_cestino_media = dimensione_cestino_media[maschera_rumore].copy()
            temp_pct_acquisti_bio = pct_acquisti_bio[maschera_rumore].copy()
            temp_sensibilita_sconto = sensibilita_sconto[maschera_rumore].copy()
            temp_visite_negozio_mensili = visite_negozio_mensili[maschera_rumore].copy()
            temp_lealta_brand = lealta_brand[maschera_rumore].copy()

            fattore_rumore_singolo = 1 + np.random.normal(0, 0.5, size=temp_eta.shape)

            eta[maschera_rumore] = temp_eta * fattore_rumore_singolo
            reddito[maschera_rumore] = temp_reddito * fattore_rumore_singolo
            visite_online_mensili[maschera_rumore] = np.abs(temp_visite_online_mensili * fattore_rumore_singolo)
            dimensione_cestino_media[maschera_rumore] = np.abs(temp_dimensione_cestino_media * fattore_rumore_singolo)
            pct_acquisti_bio[maschera_rumore] = np.clip(temp_pct_acquisti_bio * fattore_rumore_singolo, 0, 1)
            sensibilita_sconto[maschera_rumore] = np.clip(temp_sensibilita_sconto * fattore_rumore_singolo, 0, 1)
            visite_negozio_mensili[maschera_rumore] = np.abs(temp_visite_negozio_mensili * fattore_rumore_singolo)
            lealta_brand[maschera_rumore] = np.clip(temp_lealta_brand * fattore_rumore_singolo, 0, 1)

        # Crea i record
        for i in range(n):
            genere = np.random.choice(["Maschio", "Femmina"], p=[0.45, 0.55])
            carta_fedelta = np.random.choice([True, False], p=[0.7, 0.3])

            data.append([
                max(18, min(80, int(eta[i]))),
                genere,
                max(20000, min(200000, int(reddito[i]))),
                max(0, int(visite_online_mensili[i])),
                max(10, float(dimensione_cestino_media[i])),
                float(pct_acquisti_bio[i]),
                float(sensibilita_sconto[i]),
                max(0, int(visite_negozio_mensili[i])),
                float(lealta_brand[i]),
                carta_fedelta,
                nome_archetipo  # Segmento vero per la valutazione
            ])

    # Crea DataFrame
    df = pd.DataFrame(data, columns=[
        "Età", "Genere", "Reddito Annuo ($)",
        "Visite Online Mensili", "Dimensione Cestino Media ($)",
        "Percentuale Acquisti Bio", "Sensibilità allo Sconto",
        "Visite Mensili al Negozio", "Punteggio Lealtà Brand",
        "Membro Carta Fedeltà", "Segmento Vero"
    ])

    # Mischia i dati
    df = df.sample(frac=1, random_state=stato_casuale).reset_index(drop=True)

    # Prepara le caratteristiche per il clustering
    caratteristiche = [
        "Età", "Reddito Annuo ($)", "Visite Online Mensili",
        "Dimensione Cestino Media ($)", "Percentuale Acquisti Bio",
        "Sensibilità allo Sconto", "Visite Mensili al Negozio",
        "Punteggio Lealtà Brand"
    ]

    X = df[caratteristiche].copy()
    scaler = StandardScaler()
    X_scalato = scaler.fit_transform(X)

    return df, X_scalato, caratteristiche

# Genera dati
df, X_scalato, caratteristiche = genera_dati_retail(n_campioni, livello_rumore, stato_casuale)

# --- Riduzione Dimensionalità ---
@st.cache_data
def riduci_dimensioni(X, metodo, stato_casuale):
    if metodo == "PCA":
        n_componenti_pca = min(X.shape[1], 2)
        if n_componenti_pca < 2:
            st.warning("Non abbastanza caratteristiche per PCA (richiede almeno 2). Usando la prima caratteristica disponibile.")
            return X[:, :1], f"PCA (Componente 1)"
        riduttore = PCA(n_components=n_componenti_pca, random_state=stato_casuale)
        ridotto = riduttore.fit_transform(X)
        varianza_spiegata = riduttore.explained_variance_ratio_.sum() * 100
        return ridotto, f"PCA (Varianza Spiegata: {varianza_spiegata:.1f}%)"
    elif metodo == "t-SNE":
        perplexity_val = min(30, len(X) - 1)
        if perplexity_val <= 1:
            st.warning("Non abbastanza campioni per t-SNE (richiede almeno 2 campioni). Usando le caratteristiche originali.")
            return X[:, :2], "Caratteristiche Originali (Prime 2)"
        riduttore = TSNE(n_components=2, random_state=stato_casuale, perplexity=perplexity_val)
        ridotto = riduttore.fit_transform(X)
        return ridotto, "t-SNE"
    else: # Caratteristiche Originali
        if X.shape[1] < 2:
            st.warning("Non abbastanza caratteristiche originali per un grafico 2D. Usando la prima caratteristica disponibile.")
            return X[:, :1], "Caratteristiche Originali (Prima 1)"
        return X[:, :2], "Caratteristiche Originali (Prime 2)"

X_ridotto, metodo_riduzione = riduci_dimensioni(X_scalato, dim_reduction, stato_casuale)

# --- Clustering ---
@st.cache_data
def esegui_clustering(X, algoritmo, params, stato_casuale):
    etichette = np.array([]) # Inizializza etichette per ogni caso
    modello = None
    centri = None

    if X.shape[0] == 0:
        st.warning("Nessun punto dati da clusterizzare. Si prega di aumentare il 'Numero di clienti simulati'.")
        return np.array([-1]*len(X)), None, None # Restituisce tutti come rumore

    if algoritmo == "K-Means":
        if params['n_clusters'] >= X.shape[0]:
            st.warning(f"K-Means: Il numero di cluster ({params['n_clusters']}) deve essere inferiore al numero di campioni ({X.shape[0]}). Impostato a 1 cluster.")
            etichette = np.zeros(X.shape[0], dtype=int)
            centri = np.mean(X, axis=0).reshape(1, -1) if X.shape[0] > 0 else np.array([])
        else:
            modello = KMeans(
                n_clusters=params['n_clusters'],
                init=params['init_method'],
                max_iter=params['max_iter'],
                random_state=stato_casuale,
                n_init='auto'
            )
            etichette = modello.fit_predict(X)
            centri = modello.cluster_centers_ if hasattr(modello, 'cluster_centers_') else None

    elif algoritmo == "DBSCAN":
        if params['min_samples'] >= X.shape[0] or params['eps'] <= 0:
             st.warning(f"DBSCAN: Parametri non validi (min_samples={params['min_samples']}, eps={params['eps']}). Tutti i punti saranno rumore o assegnati a un singolo cluster.")
             etichette = np.array([-1]*X.shape[0]) # Tutti rumore
        else:
            modello = DBSCAN(
                eps=params['eps'],
                min_samples=params['min_samples'],
                metric=params['metric']
            )
            etichette = modello.fit_predict(X)
            # DBSCAN non ha centri di cluster espliciti come K-Means
            centri = None

    return etichette, modello, centri

# Prepara i parametri dell'algoritmo
parametri_algo_selezionato = {
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

etichette, modello, centri = esegui_clustering(
    X_scalato,
    algoritmo,
    parametri_algo_selezionato[algoritmo],
    stato_casuale
)

# --- Metriche di Valutazione ---
def calcola_metriche_di_valutazione(X, etichette):
    metriche = {}

    # Calcola le metriche solo se ci sono almeno 2 cluster e più di 1 campione
    # e se non tutti i punti sono rumore (-1)
    etichette_uniche = set(etichette)
    n_cluster_reali = len(etichette_uniche) - (1 if -1 in etichette_uniche else 0)

    if n_cluster_reali > 1 and len(etichette) > 1 and not all(l == -1 for l in etichette):
        try:
            metriche['Silhouette Score'] = silhouette_score(X, etichette)
        except Exception:
            metriche['Silhouette Score'] = np.nan

        try:
            metriche['Indice Davies-Bouldin'] = davies_bouldin_score(X, etichette)
        except Exception:
            metriche['Indice Davies-Bouldin'] = np.nan

        try:
            metriche['Indice Calinski-Harabasz'] = calinski_harabasz_score(X, etichette)
        except Exception:
            metriche['Indice Calinski-Harabasz'] = np.nan
    else:
        metriche['Silhouette Score'] = np.nan
        metriche['Indice Davies-Bouldin'] = np.nan
        metriche['Indice Calinski-Harabasz'] = np.nan

    # Conteggio cluster
    unique, counts = np.unique(etichette, return_counts=True)
    metriche['Distribuzione Cluster'] = dict(zip(unique, counts))
    metriche['Numero di Cluster Rilevati'] = n_cluster_reali
    metriche['Punti di Rumore'] = counts[unique == -1][0] if -1 in unique else 0

    return metriche

metriche = calcola_metriche_di_valutazione(X_scalato, etichette)

# --- Visualizzazione ---
def crea_grafico_cluster(X, etichette, centri, metodo, motore, df_originale):
    if X.shape[1] < 2:
        st.warning(f"Il metodo di riduzione di dimensionalità '{metodo}' ha prodotto meno di 2 dimensioni. Impossibile creare un grafico a dispersione 2D.")
        return None

    df_plot = pd.DataFrame({
        "x": X[:, 0],
        "y": X[:, 1],
        "cluster": etichette,
        "dimensione_punto": df_originale["Dimensione Cestino Media ($)"].fillna(0).astype(float) / 10
    })

    df_plot['cluster_str'] = df_plot['cluster'].astype(str)

    if motore == "Plotly":
        fig = px.scatter(
            df_plot, x="x", y="y", color="cluster_str",
            size="dimensione_punto", hover_data={
                "x": False, "y": False,
                "Età": df_originale["Età"],
                "Reddito": df_originale["Reddito Annuo ($)"],
                "Visite Online": df_originale["Visite Online Mensili"],
                "Dimensione Cestino": df_originale["Dimensione Cestino Media ($)"]
            },
            title=f"Segmenti di Clienti ({metodo})",
            labels={"x": "Componente 1", "y": "Componente 2"},
            color_discrete_map={'-1': 'grey'}, # Colora il rumore di grigio
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        if centri is not None and centri.shape[1] >= 2:
            fig.add_scatter(
                x=centri[:, 0], y=centri[:, 1],
                mode="markers", marker=dict(size=12, color="black", symbol="x"),
                name="Centri Cluster"
            )

        fig.update_layout(
            hovermode="closest",
            legend_title_text='Cluster'
        )

        return fig

    else:  # Matplotlib (solo se necessario, Plotly è più potente e interattivo)
        # Il tuo codice Matplotlib qui, ma raccomando Plotly per un'app Streamlit
        st.warning("Per un'esperienza migliore, si consiglia di usare 'Plotly' come motore di plotting.")
        return None # Restituisci None o una figura di default per Matplotlib

# Crea il grafico
cluster_plot = crea_grafico_cluster(
    X_ridotto, etichette,
    centri if algoritmo == "K-Means" and dim_reduction != "t-SNE" else None,
    metodo_riduzione,
    plot_engine,
    df
)

# --- Profilazione dei Cluster ---
def profila_cluster(df, etichette, caratteristiche):
    df_clusterizzato = df.copy()
    df_clusterizzato["Cluster"] = etichette

    # Riepilogo numerico
    profili_cluster = df_clusterizzato.groupby("Cluster")[caratteristiche].agg(
        ["mean", "median", "std", "count"]
    ).round(2) # Arrotonda a 2 decimali

    # Riepilogo categorico
    if "Genere" in df.columns:
        if not df_clusterizzato.empty:
            dist_genere = pd.crosstab(df_clusterizzato["Cluster"], df_clusterizzato["Genere"])
            sum_dist_genere = dist_genere.sum(1)
            dist_genere_pct = dist_genere.div(sum_dist_genere, axis=0).round(2) if not sum_dist_genere.empty else dist_genere
            profili_cluster = pd.concat([profili_cluster, dist_genere_pct], axis=1)

    if "Membro Carta Fedeltà" in df.columns:
        if not df_clusterizzato.empty:
            dist_carta_fedelta = pd.crosstab(df_clusterizzato["Cluster"], df_clusterizzato["Membro Carta Fedeltà"])
            sum_dist_carta_fedelta = dist_carta_fedelta.sum(1)
            dist_carta_fedelta_pct = dist_carta_fedelta.div(sum_dist_carta_fedelta, axis=0).round(2) if not sum_dist_carta_fedelta.empty else dist_carta_fedelta
            profili_cluster = pd.concat([profili_cluster, dist_carta_fedelta_pct], axis=1)

    return profili_cluster, df_clusterizzato

profili_cluster, df_clusterizzato = profila_cluster(df, etichette, caratteristiche)

# --- Comparazione Segmento Vero ---
def compara_segmenti_veri(df_clusterizzato):
    if "Segmento Vero" not in df_clusterizzato.columns or df_clusterizzato.empty:
        return None, None

    df_filtrato = df_clusterizzato[df_clusterizzato["Cluster"] != -1]
    if df_filtrato.empty:
        return None, None

    comparazione = pd.crosstab(
        df_filtrato["Cluster"],
        df_filtrato["Segmento Vero"],
        normalize="index"
    ).round(2) # Arrotonda a 2 decimali

    purezza = comparazione.max(axis=1).mean()

    return comparazione, purezza

comparazione_segmento_vero, punteggio_purezza = compara_segmenti_veri(df_clusterizzato)

# --- Display Principale ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visualizzazione",
    "📈 Analisi Cluster",
    "📋 Esploratore Dati",
    "📚 Documentazione"
])

with tab1:
    st.header("Visualizzazione della Segmentazione Clienti")

    col1, col2 = st.columns([3, 1])

    with col1:
        if cluster_plot: # Controlla se il grafico è stato creato
            st.plotly_chart(cluster_plot, use_container_width=True)
        else:
            st.info("Impossibile creare il grafico 2D. Controlla i parametri di riduzione dimensionalità o il numero di features.")

    with col2:
        st.subheader("Metriche Algoritmo")

        st.metric("Numero di Cluster Rilevati", metrics["Numero di Cluster Rilevati"])
        st.metric("Punti di Rumore", metrics["Punti di Rumore"])

        if not np.isnan(metrics['Silhouette Score']):
            st.metric("Silhouette Score", f"{metrics['Silhouette Score']:.3f}",
                      help="Valori più alti indicano cluster meglio definiti (-1 a 1)")
        else:
            st.metric("Silhouette Score", "N/D", help="Impossibile calcolare il punteggio per meno di 2 cluster o solo punti di rumore.")

        if not np.isnan(metrics['Indice Davies-Bouldin']):
            st.metric("Indice Davies-Bouldin", f"{metrics['Indice Davies-Bouldin']:.3f}",
                      help="Valori più bassi indicano un clustering migliore (0 a ∞)")
        else:
            st.metric("Indice Davies-Bouldin", "N/D", help="Impossibile calcolare il punteggio per meno di 2 cluster o solo punti di rumore.")

        if not np.isnan(metrics['Indice Calinski-Harabasz']):
            st.metric("Indice Calinski-Harabasz", f"{metrics['Indice Calinski-Harabasz']:.3f}",
                      help="Valori più alti indicano un clustering migliore (0 a ∞)")
        else:
            st.metric("Indice Calinski-Harabasz", "N/D", help="Impossibile calcolare il punteggio per meno di 2 cluster o solo punti di rumore.")

        if comparazione_segmento_vero is not None and punteggio_purezza is not None:
            st.metric("Purezza Segmento", f"{punteggio_purezza:.1%}",
                      help="Quanto bene i cluster corrispondono ai segmenti veri")
        else:
            st.metric("Purezza Segmento", "N/D", help="Segmenti veri non disponibili o dati insufficienti per il confronto.")

with tab2:
    st.header("Analisi dei Cluster")

    st.subheader("Caratteristiche dei Cluster")
    st.dataframe(profili_cluster[profili_cluster.index != -1].style.background_gradient(cmap="Blues"))

    st.subheader("Distribuzioni delle Caratteristiche per Cluster")
    caratteristica_selezionata = st.selectbox("Seleziona caratteristica da visualizzare", caratteristiche)

    df_per_boxplot = df_clusterizzato[df_clusterizzato["Cluster"] != -1]
    if not df_per_boxplot.empty:
        fig = px.box(df_per_boxplot, x="Cluster", y=caratteristica_selezionata, color="Cluster")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Nessun cluster valido per visualizzare le distribuzioni delle caratteristiche.")

    if comparazione_segmento_vero is not None:
        st.subheader("Comparazione con i Segmenti Veri")
        st.dataframe(comparazione_segmento_vero.style.background_gradient(cmap="Greens", axis=1))

        fig = px.imshow(comparazione_segmento_vero, text_auto=".1%", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Esploratore Dati")

    st.subheader("Dati Grezzi con Assegnazioni Cluster")
    st.dataframe(df_clusterizzato.round(2)) # Arrotonda il dataframe a 2 decimali

    # Bottone per il download
    csv = df_clusterizzato.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Scarica dati clusterizzati come CSV",
        data=csv,
        file_name="clienti_retail_clusterizzati.csv",
        mime="text/csv"
    )

    st.subheader("Correlazioni tra Caratteristiche")
    matrice_correlazione = df_clusterizzato[caratteristiche].corr()

    fig = px.imshow(matrice_correlazione, text_auto=".2f", color_continuous_scale="RdBu",
                    title="Matrice di Correlazione delle Caratteristiche")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Documentazione")

    st.subheader("Informazioni su Questo Strumento")
    st.markdown("""
    Questo strumento interattivo permette agli analisti retail di:
    - Esplorare la segmentazione dei clienti utilizzando diversi algoritmi
    - Confrontare le performance degli algoritmi tramite metriche multiple
    - Profilare i segmenti di clienti basandosi sulle loro caratteristiche
    - Scoprire pattern nascosti nel comportamento dei clienti

    **Caratteristiche Chiave:**
    - Due algoritmi di clustering (K-Means, DBSCAN)
    - Diverse tecniche di riduzione di dimensionalità
    - Visualizzazioni interattive con Plotly
    - Profilazione completa dei cluster
    - Valutazione rispetto alla "verità" (quando disponibile nei dati simulati)
    """)

    st.subheader("Descrizioni delle Caratteristiche")
    descrizioni_caratteristiche = {
        "Età": "Età del cliente in anni",
        "Reddito Annuo ($)": "Reddito annuo stimato del nucleo familiare",
        "Visite Online Mensili": "Numero di visite al negozio online/app al mese",
        "Dimensione Cestino Media ($)": "Spesa media per ogni acquisto",
        "Percentuale Acquisti Bio": "Percentuale di acquisti di prodotti biologici",
        "Sensibilità allo Sconto": "Probabilità di rispondere agli sconti (scala 0-1)",
        "Visite Mensili al Negozio": "Numero di visite al negozio fisico al mese",
        "Punteggio Lealtà Brand": "Preferenza per brand noti vs. marche private (scala 0-1)",
        "Membro Carta Fedeltà": "Se il cliente possiede una carta fedeltà",
        "Segmento Vero": "Segmento reale (solo nei dati simulati)"
    }

    for feat, desc in descrizioni_caratteristiche.items():
        st.markdown(f"**{feat}**: {desc}")

    st.subheader("Guida agli Algoritmi")
    st.markdown("""
    **K-Means**:
    - Ideale per cluster sferici di dimensioni simili.
    - Richiede la specificazione del numero di cluster.
    - Sensibile agli outlier.

    **DBSCAN**:
    - Trova cluster di forma arbitraria.
    - Identifica i punti di rumore.
    - Richiede la taratura di epsilon e min_samples.
    """)

# --- Footer ---
st.markdown("---")
st.markdown("""
    *Strumento Avanzato di Analisi Retail* | Creato con Streamlit |
    [Repository GitHub](https://github.com/tuo-repository)
""")
