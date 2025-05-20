import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px

# --- Configurazione Pagina Streamlit ---
st.set_page_config(layout="wide", page_title="Clustering Didattico: K-Means & DBSCAN", page_icon="🔬")

st.title("🔬 Clustering Didattico: K-Means & DBSCAN")
st.markdown("""
Questa applicazione ti permette di **esplorare e capire** gli algoritmi di clustering K-Means e DBSCAN su dati simulati di clienti retail.
Modifica i parametri, osserva i risultati e scopri come funzionano questi algoritmi!
""")

# --- SIDEBAR: Parametri e spiegazioni algoritmi ---
with st.sidebar:
    st.header("⚙️ Configurazione Esperimento")

    st.subheader("1. Generazione Dati")
    n_campioni = st.slider("Numero di clienti simulati", 500, 5000, 1000, step=100,
                           help="Numero totale di dati generati per l'esperimento.")
    livello_rumore = st.slider("Livello di rumore (%)", 0, 30, 5,
                               help="Percentuale di dati generati con valori casuali/rumorosi.")
    stato_casuale = st.slider("Seed casuale", 0, 100, 42,
                              help="Cambia questo valore per ottenere una diversa generazione casuale dei dati.")

    st.subheader("2. Selezione Algoritmo")
    algoritmo = st.radio("Algoritmo di Clustering:",
                         ["K-Means", "DBSCAN"],
                         index=0)

    # Parametri algoritmi con tooltip
    if algoritmo == "K-Means":
        n_clusters = st.slider("Numero di cluster (K)", 2, 15, 6,
                               help="Numero di gruppi che vuoi trovare nei dati.")
        init_method = st.selectbox("Metodo di inizializzazione",
                                   ["k-means++", "random"],
                                   help="Come vengono scelti i centroidi iniziali.")
        max_iter = st.slider("Max iterazioni", 100, 500, 300,
                             help="Quante volte aggiornare i centroidi al massimo.")
        show_steps = st.checkbox("Mostra passi K-Means", value=False,
                                 help="Spiega i passi principali dell'algoritmo K-Means.")
    else:
        eps = st.slider("Epsilon (raggio di vicinato)", 0.1, 2.0, 0.5, step=0.05,
                        help="Distanza massima per considerare due punti come vicini.")
        min_samples = st.slider("Campioni minimi", 1, 50, 10,
                                help="Numero minimo di punti per formare un cluster denso.")
        metric = st.selectbox("Metrica di distanza",
                              ["euclidean", "cosine", "manhattan"],
                              help="Come si misura la distanza tra i punti.")

    st.subheader("3. Visualizzazione")
    dim_reduction = st.selectbox("Riduzione di dimensionalità",
                                 ["PCA", "t-SNE", "Caratteristiche Originali"])
    plot_engine = st.selectbox("Motore di plotting",
                               ["Plotly", "Matplotlib"])

    st.markdown("---")
    st.header("📚 Spiegazione Algoritmo")
    if algoritmo == "K-Means":
        st.markdown("""
        **K-Means** cerca di dividere i dati in *K* gruppi ("cluster") minimizzando la distanza tra i punti e il centroide del proprio gruppo.
        - Ogni punto viene assegnato al centroide più vicino.
        - I centroidi vengono aggiornati iterativamente.
        - Il processo termina quando i centroidi non cambiano più o si raggiunge il numero massimo di iterazioni.
        """)
    else:
        st.markdown("""
        **DBSCAN** raggruppa i punti che sono vicini tra loro (densità) e identifica i punti "rumore".
        - Un punto centrale ha almeno *min_samples* punti nel suo raggio *eps*.
        - I cluster si formano collegando punti densi.
        - I punti isolati sono considerati rumore.
        """)

# --- Generazione Dati ---
@st.cache_data
def genera_dati_retail(n_campioni, livello_rumore, stato_casuale):
    np.random.seed(stato_casuale)
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
        eta = np.random.normal(params["eta"][0], params["eta"][1], n)
        reddito = np.random.normal(params["reddito"][0], params["reddito"][1], n)
        visite_online_mensili = np.random.poisson(params["visite_online_mensili"][0], n)
        dimensione_cestino_media = np.abs(np.random.normal(params["dimensione_cestino_media"][0], params["dimensione_cestino_media"][1], n))
        pct_acquisti_bio = np.clip(np.random.normal(params["pct_acquisti_bio"][0], params["pct_acquisti_bio"][1], n), 0, 1)
        sensibilita_sconto = np.clip(np.random.normal(params["sensibilita_sconto"][0], params["sensibilita_sconto"][1], n), 0, 1)
        visite_negozio_mensili = np.random.poisson(params["visite_negozio_mensili"][0], n)
        lealta_brand = np.clip(np.random.normal(params["lealta_brand"][0], params["lealta_brand"][1], n), 0, 1)

        maschera_rumore = np.random.random(n) < (livello_rumore/100)
        if maschera_rumore.any():
            fattore_rumore_singolo = 1 + np.random.normal(0, 0.5, size=maschera_rumore.sum())
            eta[maschera_rumore] *= fattore_rumore_singolo
            reddito[maschera_rumore] *= fattore_rumore_singolo
            visite_online_mensili[maschera_rumore] = np.abs(visite_online_mensili[maschera_rumore] * fattore_rumore_singolo)
            dimensione_cestino_media[maschera_rumore] = np.abs(dimensione_cestino_media[maschera_rumore] * fattore_rumore_singolo)
            pct_acquisti_bio[maschera_rumore] = np.clip(pct_acquisti_bio[maschera_rumore] * fattore_rumore_singolo, 0, 1)
            sensibilita_sconto[maschera_rumore] = np.clip(sensibilita_sconto[maschera_rumore] * fattore_rumore_singolo, 0, 1)
            visite_negozio_mensili[maschera_rumore] = np.abs(visite_negozio_mensili[maschera_rumore] * fattore_rumore_singolo)
            lealta_brand[maschera_rumore] = np.clip(lealta_brand[maschera_rumore] * fattore_rumore_singolo, 0, 1)

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
                nome_archetipo
            ])

    df = pd.DataFrame(data, columns=[
        "Età", "Genere", "Reddito Annuo ($)",
        "Visite Online Mensili", "Dimensione Cestino Media ($)",
        "Percentuale Acquisti Bio", "Sensibilità allo Sconto",
        "Visite Mensili al Negozio", "Punteggio Lealtà Brand",
        "Membro Carta Fedeltà", "Segmento Vero"
    ])
    df = df.sample(frac=1, random_state=stato_casuale).reset_index(drop=True)

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

df, X_scalato, caratteristiche = genera_dati_retail(n_campioni, livello_rumore, stato_casuale)

# --- Riduzione Dimensionalità ---
@st.cache_data
def riduci_dimensioni(X, metodo, stato_casuale):
    if metodo == "PCA":
        n_componenti_pca = min(X.shape[1], 2)
        if n_componenti_pca < 2:
            st.warning("Non abbastanza caratteristiche per PCA (richiede almeno 2).")
            return X[:, :1], f"PCA (Componente 1)"
        riduttore = PCA(n_components=n_componenti_pca, random_state=stato_casuale)
        ridotto = riduttore.fit_transform(X)
        varianza_spiegata = riduttore.explained_variance_ratio_.sum() * 100
        return ridotto, f"PCA (Varianza Spiegata: {varianza_spiegata:.1f}%)"
    elif metodo == "t-SNE":
        perplexity_val = min(30, len(X) - 1)
        if perplexity_val <= 1:
            st.warning("Non abbastanza campioni per t-SNE.")
            return X[:, :2], "Caratteristiche Originali (Prime 2)"
        riduttore = TSNE(n_components=2, random_state=stato_casuale, perplexity=perplexity_val)
        ridotto = riduttore.fit_transform(X)
        return ridotto, "t-SNE"
    else:
        if X.shape[1] < 2:
            st.warning("Non abbastanza caratteristiche originali per un grafico 2D.")
            return X[:, :1], "Caratteristiche Originali (Prima 1)"
        return X[:, :2], "Caratteristiche Originali (Prime 2)"

X_ridotto, metodo_riduzione = riduci_dimensioni(X_scalato, dim_reduction, stato_casuale)

# --- Clustering ---
@st.cache_data
def esegui_clustering(X, algoritmo, params, stato_casuale):
    etichette = np.array([])
    modello = None
    centri = None

    if algoritmo == "K-Means":
        if params['n_clusters'] >= X.shape[0]:
            st.warning(f"K-Means: Il numero di cluster ({params['n_clusters']}) deve essere inferiore al numero di campioni ({X.shape[0]}).")
            etichette = np.zeros(X.shape[0], dtype=int)
            centri = np.mean(X, axis=0).reshape(1, -1)
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
            st.warning(f"DBSCAN: Parametri non validi. Tutti i punti saranno rumore.")
            etichette = np.array([-1]*X.shape[0])
        else:
            modello = DBSCAN(
                eps=params['eps'],
                min_samples=params['min_samples'],
                metric=params['metric']
            )
            etichette = modello.fit_predict(X)
            centri = None
    return etichette, modello, centri

if algoritmo == "K-Means":
    params_algo = {
        'n_clusters': n_clusters,
        'init_method': init_method,
        'max_iter': max_iter
    }
else:
    params_algo = {
        'eps': eps,
        'min_samples': min_samples,
        'metric': metric
    }

etichette, modello, centri = esegui_clustering(
    X_scalato,
    algoritmo,
    params_algo,
    stato_casuale
)

# --- Metriche di Valutazione ---
def calcola_metriche_di_valutazione(X, etichette):
    metriche = {}
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
        "cluster": etichette
    })

    if motore == "Plotly":
        fig = px.scatter(
            df_plot, x="x", y="y", color=df_plot["cluster"].astype(str),
            title=f"Clustering ({metodo})",
            labels={"color": "Cluster"}
        )
        # Centroidi per K-Means
        if centri is not None and centri.shape[1] >= 2:
            # Riduci i centroidi nelle stesse dimensioni
            if metodo == "PCA":
                pca = PCA(n_components=2, random_state=42)
                pca.fit(X_scalato)
                centri_2d = pca.transform(centri)
            elif metodo == "t-SNE":
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(centri)-1))
                centri_2d = tsne.fit_transform(centri)
            else:
                centri_2d = centri[:, :2]
            fig.add_scatter(
                x=centri_2d[:, 0], y=centri_2d[:, 1],
                mode='markers', marker=dict(color='black', size=16, symbol='x'),
                name='Centroidi'
            )
        # Evidenzia punti di rumore per DBSCAN
        if (etichette == -1).any():
            rumore = df_plot[etichette == -1]
            fig.add_scatter(
                x=rumore["x"], y=rumore["y"],
                mode='markers', marker=dict(color='red', size=8, symbol='circle-open'),
                name='Rumore'
            )
        st.plotly_chart(fig, use_container_width=True)
    else:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(df_plot["x"], df_plot["y"], c=df_plot["cluster"], cmap="tab10", alpha=0.7)
        if centri is not None and centri.shape[1] >= 2:
            plt.scatter(centri[:, 0], centri[:, 1], c='black', s=200, marker='X', label='Centroidi')
        if (etichette == -1).any():
            plt.scatter(df_plot["x"][etichette == -1], df_plot["y"][etichette == -1], c='red', s=30, label='Rumore', marker='o')
        plt.title(f"Clustering ({metodo})")
        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

# --- OUTPUT PRINCIPALE ---
st.header("🔎 Visualizzazione Clustering")
crea_grafico_cluster(X_ridotto, etichette, centri, metodo_riduzione, plot_engine, df)

# --- Step by Step (K-Means) ---
if algoritmo == "K-Means" and 'show_steps' in locals() and show_steps:
    st.subheader("🔍 Passi dell'algoritmo K-Means")
    st.markdown("""
    1. **Inizializzazione dei centroidi** (casuale o k-means++)
    2. **Assegnazione** di ogni punto al centroide più vicino
    3. **Aggiornamento** dei centroidi come media dei punti assegnati
    4. Ripeti i passi 2-3 fino a convergenza o max iterazioni
    """)
    if modello is not None:
        st.write("Centroidi finali (spazio originale):")
        st.dataframe(pd.DataFrame(modello.cluster_centers_, columns=caratteristiche))

# --- Metriche e spiegazioni ---
st.header("📏 Metriche di valutazione clustering")
for nome, valore in metriche.items():
    if isinstance(valore, dict):
        st.write(f"**{nome}:** {valore}")
    else:
        st.write(f"**{nome}:** {valore:.3f}" if isinstance(valore, float) else f"**{nome}:** {valore}")

st.markdown("""
- **Silhouette Score**: quanto i cluster sono separati (più vicino a 1 è meglio).
- **Davies-Bouldin**: quanto i cluster sono compatti (più basso è meglio).
- **Calinski-Harabasz**: rapporto tra dispersione intra-cluster e inter-cluster (più alto è meglio).
""")

st.markdown("### Come cambiano i risultati al variare dei parametri?")
if algoritmo == "K-Means":
    st.info("""
    - **Se aumenti K**, i gruppi diventano più piccoli e specifici.
    - **Se scegli 'random' come inizializzazione**, i risultati possono cambiare a ogni esecuzione.
    """)
else:
    st.info("""
    - **Se aumenti eps**, i cluster diventano più grandi e meno selettivi.
    - **Se aumenti min_samples**, servono più punti vicini per formare un cluster (più punti saranno rumore).
    """)

st.markdown("---")
st.markdown("**App didattica realizzata con ❤️ per l'apprendimento del clustering!**")
