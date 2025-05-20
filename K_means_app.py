import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE # Importazione corretta di TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go # Importazione necessaria per go.Scatter

# --- Configurazione Pagina Streamlit ---
st.set_page_config(layout="wide", page_title="Analisi Avanzata della Segmentazione Clienti", page_icon="🛒")

# --- Titolo e Introduzione ---
st.title("🛒 Analisi Avanzata della Segmentazione Clienti per il Retail")
st.markdown("""
Questa applicazione interattiva ti permette di esplorare algoritmi di clustering avanzati su dati **simulati di clienti di supermercati**.
Scopri segmenti di clienti nascosti e ottieni insight utili per strategie di marketing mirate.
""")

# --- Funzione per Generare Dati Simulati di Clienti Retail ---
@st.cache_data
def genera_dati_retail(n_campioni, livello_rumore, stato_casuale):
    """
    Genera un dataset simulato di clienti retail con varie caratteristiche.
    Include archetipi di clienti e aggiunge rumore.
    """
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
    # Distribuisci equamente i campioni tra gli archetipi, gestendo i rimanenti
    campioni_per_tipo = n_campioni // len(archetipi)
    campioni_rimanenti = n_campioni % len(archetipi)

    for i, (nome_archetipo, params) in enumerate(archetipi.items()):
        n = campioni_per_tipo + (1 if i < campioni_rimanenti else 0)

        # Genera i dati del cluster principale
        eta = np.random.normal(params["eta"][0], params["eta"][1], n)
        reddito = np.random.normal(params["reddito"][0], params["reddito"][1], n)
        visite_online_mensili = np.random.poisson(params["visite_online_mensili"][0], n)
        dimensione_cestino_media = np.abs(np.random.normal(params["dimensione_cestino_media"][0], params["dimensione_cestino_media"][1], n))
        pct_acquisti_bio = np.clip(np.random.normal(params["pct_acquisti_bio"][0], params["pct_acquisti_bio"][1], n), 0, 1)
        sensibilita_sconto = np.clip(np.random.normal(params["sensibilita_sconto"][0], params["sensibilita_sconto"][1], n), 0, 1)
        visite_negozio_mensili = np.random.poisson(params["visite_negozio_mensili"][0], n)
        lealta_brand = np.clip(np.random.normal(params["lealta_brand"][0], params["lealta_brand"][1], n), 0, 1)

        # Aggiungi un po' di rumore ai dati
        maschera_rumore = np.random.random(n) < (livello_rumore/100)
        if maschera_rumore.any():
            fattore_rumore_singolo = 1 + np.random.normal(0, 0.5, size=np.sum(maschera_rumore))
            eta[maschera_rumore] = eta[maschera_rumore] * fattore_rumore_singolo
            reddito[maschera_rumore] = reddito[maschera_rumore] * fattore_rumore_singolo
            visite_online_mensili[maschera_rumore] = np.abs(visite_online_mensili[maschera_rumore] * fattore_rumore_singolo)
            dimensione_cestino_media[maschera_rumore] = np.abs(dimensione_cestino_media[maschera_rumore] * fattore_rumore_singolo)
            pct_acquisti_bio[maschera_rumore] = np.clip(pct_acquisti_bio[maschera_rumore] * fattore_rumore_singolo, 0, 1)
            sensibilita_sconto[maschera_rumore] = np.clip(sensibilita_sconto[maschera_rumore] * fattore_rumore_singolo, 0, 1)
            visite_negozio_mensili[maschera_rumore] = np.abs(visite_negozio_mensili[maschera_rumore] * fattore_rumore_singolo)
            lealta_brand[maschera_rumore] = np.clip(lealta_brand[maschera_rumore] * fattore_rumore_singolo, 0, 1)

        # Crea i record
        for j in range(n):
            genere = np.random.choice(["Maschio", "Femmina"], p=[0.45, 0.55])
            carta_fedelta = np.random.choice([True, False], p=[0.7, 0.3])

            data.append([
                max(18, min(80, int(eta[j]))),
                genere,
                max(20000, min(200000, int(reddito[j]))),
                max(0, int(visite_online_mensili[j])),
                max(10, float(dimensione_cestino_media[j])),
                float(pct_acquisti_bio[j]),
                float(sensibilita_sconto[j]),
                max(0, int(visite_negozio_mensili[j])),
                float(lealta_brand[j]),
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

    # Mischia i dati per rimuovere l'ordinamento per archetipo
    df = df.sample(frac=1, random_state=stato_casuale).reset_index(drop=True)

    # Prepara le caratteristiche numeriche per il clustering
    caratteristiche_numeriche = [
        "Età", "Reddito Annuo ($)", "Visite Online Mensili",
        "Dimensione Cestino Media ($)", "Percentuale Acquisti Bio",
        "Sensibilità allo Sconto", "Visite Mensili al Negozio",
        "Punteggio Lealtà Brand"
    ]

    X = df[caratteristiche_numeriche].copy()
    scaler = StandardScaler()
    X_scalato = scaler.fit_transform(X)

    return df, X_scalato, caratteristiche_numeriche, scaler # Restituisce lo scaler per proiezioni future

# --- Funzione per la Riduzione Dimensionalità ---
@st.cache_data
def riduci_dimensioni(X, metodo, stato_casuale):
    """
    Applica tecniche di riduzione dimensionalità ai dati per la visualizzazione.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if n_samples == 0:
        st.warning("Nessun campione per la riduzione di dimensionalità.")
        return np.array([]).reshape(0, 2), "Nessuna Dimensione"
    if n_features < 1:
        st.warning("Nessuna caratteristica per la riduzione di dimensionalità.")
        return np.array([]).reshape(n_samples, 0), "Nessuna Dimensione"

    if metodo == "PCA":
        n_components_pca = min(n_features, 2)
        if n_components_pca < 2:
            st.warning(f"PCA richiede almeno 2 caratteristiche per la visualizzazione 2D. Verranno usate {n_components_pca} componenti.")
            if n_components_pca == 1:
                return X[:, :1], "PCA (1 Componente)"
            return np.array([]).reshape(n_samples, 0), "Nessuna Dimensione" # Caso 0 componenti
        
        riduttore = PCA(n_components=n_components_pca, random_state=stato_casuale)
        ridotto = riduttore.fit_transform(X)
        varianza_spiegata = riduttore.explained_variance_ratio_.sum() * 100
        return ridotto, f"PCA (Varianza Spiegata: {varianza_spiegata:.1f}%)"
    
    elif metodo == "t-SNE":
        # Per t-SNE, perplexity deve essere < numero di campioni e > 1
        # E richiede almeno 2 features per funzionare
        if n_samples <= 1:
            st.warning("t-SNE richiede almeno 2 campioni. Impossibile applicare.")
            return X[:, :min(n_features, 2)], "Caratteristiche Originali (Prime 2)"
        if n_features < 2:
            st.warning("t-SNE richiede almeno 2 caratteristiche. Impossibile applicare.")
            return X[:, :min(n_features, 2)], "Caratteristiche Originali (Prime 2)"

        perplexity_val = min(30, max(5, n_samples - 1)) # Perplexity ragionevole

        riduttore = TSNE(n_components=2, random_state=stato_casuale, perplexity=perplexity_val, n_jobs=-1) # n_jobs per velocità
        ridotto = riduttore.fit_transform(X)
        return ridotto, "t-SNE"
    
    else: # Caratteristiche Originali
        if n_features < 2:
            st.warning("Il dataset ha meno di 2 caratteristiche. Visualizzazione 2D limitata.")
            if n_features == 1:
                return X[:, :1], "Caratteristiche Originali (Prima 1)"
            return np.array([]).reshape(n_samples, 0), "Nessuna Dimensione"
        return X[:, :2], "Caratteristiche Originali (Prime 2)"

# --- Funzione per Eseguire il Clustering ---
def esegui_clustering(X_scalato, algoritmo, params, stato_casuale):
    """
    Esegue l'algoritmo di clustering selezionato.
    """
    etichette = np.array([])
    modello = None
    centri = None
    inertia = None # Solo per K-Means

    if X_scalato.shape[0] == 0:
        st.warning("Nessun punto dati da clusterizzare. Si prega di aumentare il 'Numero di clienti simulati'.")
        return np.array([-1]*0), None, None, None # Ritorna array vuoti

    if algoritmo == "K-Means":
        n_clusters_effettivo = min(params['n_clusters'], X_scalato.shape[0])
        if n_clusters_effettivo < 1:
            st.warning("K-Means: Il numero di cluster K non può essere inferiore a 1. Impostato a 1.")
            etichette = np.zeros(X_scalato.shape[0], dtype=int)
            centri = np.mean(X_scalato, axis=0).reshape(1, -1) if X_scalato.shape[0] > 0 else np.array([])
            inertia = 0
        elif n_clusters_effettivo == 0: # Caso estremo con 0 campioni
             etichette = np.array([-1]*X_scalato.shape[0])
             centri = None
             inertia = None
        else:
            kmeans = KMeans(
                n_clusters=n_clusters_effettivo,
                init=params['init_method'],
                max_iter=params['max_iter'],
                random_state=stato_casuale,
                n_init='auto' # Usa 'auto' per le versioni recenti di scikit-learn
            )
            etichette = kmeans.fit_predict(X_scalato)
            centri = kmeans.cluster_centers_
            inertia = kmeans.inertia_

    elif algoritmo == "DBSCAN":
        if params['min_samples'] >= X_scalato.shape[0] or params['eps'] <= 0:
             st.warning(f"DBSCAN: Parametri non validi (min_samples={params['min_samples']}, eps={params['eps']}). Tutti i punti saranno rumore.")
             etichette = np.array([-1]*X_scalato.shape[0]) # Tutti rumore
        else:
            dbscan = DBSCAN(
                eps=params['eps'],
                min_samples=params['min_samples'],
                metric=params['metric']
            )
            etichette = dbscan.fit_predict(X_scalato)
            centri = None # DBSCAN non ha centri espliciti

    return etichette, modello, centri, inertia

# --- Funzione per Calcolare le Metriche di Clustering ---
def calcola_metriche(X, etichette):
    """
    Calcola varie metriche di valutazione del clustering.
    """
    metriche = {}

    # Filtra per escludere il rumore (-1) dal calcolo delle metriche intrinseche
    X_validi = X[etichette != -1]
    etichette_valide = etichette[etichette != -1]

    etichette_uniche_valide = set(etichette_valide)
    n_cluster_reali = len(etichette_uniche_valide)

    # Calcola le metriche solo se ci sono almeno 2 cluster validi e punti sufficienti
    if n_cluster_reali >= 2 and len(etichette_valide) > 1:
        try:
            metriche['Silhouette Score'] = silhouette_score(X_validi, etichette_valide)
        except Exception as e:
            metriche['Silhouette Score'] = np.nan
            # st.error(f"Errore calcolo Silhouette: {e}") # Debug: abilita per vedere errori specifici

        try:
            metriche['Indice Davies-Bouldin'] = davies_bouldin_score(X_validi, etichette_valide)
        except Exception as e:
            metriche['Indice Davies-Bouldin'] = np.nan
            # st.error(f"Errore calcolo Davies-Bouldin: {e}")

        try:
            metriche['Indice Calinski-Harabasz'] = calinski_harabasz_score(X_validi, etichette_valide)
        except Exception as e:
            metriche['Indice Calinski-Harabasz'] = np.nan
            # st.error(f"Errore calcolo Calinski-Harabasz: {e}")
    else:
        metriche['Silhouette Score'] = np.nan
        metriche['Indice Davies-Bouldin'] = np.nan
        metriche['Indice Calinski-Harabasz'] = np.nan

    # Conteggio cluster e rumore per tutti i punti
    unique_full, counts_full = np.unique(etichette, return_counts=True)
    metriche['Distribuzione Cluster'] = dict(zip(unique_full, counts_full))
    metriche['Numero di Cluster Rilevati'] = n_cluster_reali
    metriche['Punti di Rumore'] = counts_full[unique_full == -1][0] if -1 in unique_full else 0

    return metriche

# --- Controlli della Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurazione dell'Esperimento")

    st.subheader("1. Generazione Dati Clienti")
    n_campioni = st.slider("Numero di clienti simulati", 500, 5000, 1000, step=100)
    livello_rumore_ds = st.slider("Livello di rumore (%)", 0, 30, 5)
    stato_casuale_ds = st.slider("Seed casuale dati", 0, 100, 42)

    st.markdown("---")

    st.subheader("2. Selezione Algoritmo")
    algoritmo = st.radio("Algoritmo di Clustering:", ["K-Means", "DBSCAN"], index=0)

    st.markdown("---")

    st.subheader(f"3. Parametri {algoritmo}")
    parametri_algo_selezionato = {}
    if algoritmo == "K-Means":
        parametri_algo_selezionato['n_clusters'] = st.slider("Numero di Cluster (K)", 2, 10, 6)
        parametri_algo_selezionato['init_method'] = st.selectbox("Metodo di Inizializzazione", ["k-means++", "random"])
        parametri_algo_selezionato['max_iter'] = st.slider("Max Iterazioni", 100, 500, 300)
    elif algoritmo == "DBSCAN":
        parametri_algo_selezionato['eps'] = st.slider("Epsilon (raggio di vicinato)", 0.1, 2.0, 0.5, step=0.05)
        parametri_algo_selezionato['min_samples'] = st.slider("Campioni minimi nel vicinato", 1, 50, 5)
        parametri_algo_selezionato['metric'] = st.selectbox("Metrica di distanza", ["euclidean", "cosine", "manhattan"])

    st.markdown("---")

    st.subheader("4. Visualizzazione")
    dim_reduction = st.selectbox("Riduzione di dimensionalità per il grafico", ["PCA", "t-SNE", "Caratteristiche Originali"], index=0)

    st.markdown("---")
    if st.button("🔄 Esegui Analisi"):
        st.experimental_rerun()

# --- Esecuzione Pipeline ---
with st.spinner('Preparazione dati e esecuzione analisi...'):
    df_originale, X_scalato_ds, caratteristiche_numeriche, scaler_ds = genera_dati_retail(
        n_campioni, livello_rumore_ds, stato_casuale_ds
    )

    X_ridotto_ds, metodo_riduzione_etichetta = riduci_dimensioni(
        X_scalato_ds, dim_reduction, stato_casuale_ds
    )

    etichette_pred, modello_cluster, centri_cluster_scalati, inertia_val = esegui_clustering(
        X_scalato_ds, algoritmo, parametri_algo_selezionato, stato_casuale_ds
    )
    df_originale['etichetta_cluster'] = etichette_pred

    metriche_risultati = calcola_metriche(X_scalato_ds, etichette_pred)

# --- Visualizzazione dei Cluster con Plotly ---
st.header(f"🚀 Risultati del Clustering: {algoritmo}")

col_grafico, col_metriche = st.columns([2, 1])

with col_grafico:
    st.subheader("Grafico dei Cluster Rilevati")

    if X_ridotto_ds.shape[1] < 2:
        st.warning("Impossibile generare un grafico 2D: la riduzione di dimensionalità ha prodotto meno di 2 dimensioni. Riprova con più caratteristiche o un altro metodo.")
    elif X_ridotto_ds.shape[0] == 0:
        st.info("Nessun dato da visualizzare. Aumenta il numero di clienti simulati.")
    else:
        df_plot = pd.DataFrame({
            "Dimensione 1": X_ridotto_ds[:, 0],
            "Dimensione 2": X_ridotto_ds[:, 1],
            "Cluster": etichette_pred,
            "Età": df_originale["Età"],
            "Reddito Annuo ($)": df_originale["Reddito Annuo ($)"],
            "Visite Online Mensili": df_originale["Visite Online Mensili"],
            "Dimensione Cestino Media ($)": df_originale["Dimensione Cestino Media ($)"]
        })
        df_plot['Cluster'] = df_plot['Cluster'].astype(str) # Converti in stringa per colorazione categoriale

        fig = px.scatter(
            df_plot,
            x="Dimensione 1",
            y="Dimensione 2",
            color="Cluster",
            hover_data={
                "Età": True,
                "Reddito Annuo ($)": True,
                "Visite Online Mensili": True,
                "Dimensione Cestino Media ($)": True,
                "Cluster": True,
                "Dimensione 1": ':.2f', # Arrotonda per hover
                "Dimensione 2": ':.2f'
            },
            title=f"Cluster dei Clienti ({metodo_riduzione_etichetta} - {algoritmo})",
            labels={"Dimensione 1": "Componente 1", "Dimensione 2": "Componente 2"},
            color_discrete_map={'-1': 'grey'}, # Colora il rumore di grigio
            color_discrete_sequence=px.colors.qualitative.Plotly # Sequenza di colori predefinita per i cluster
        )

        # Aggiungi i centroidi per K-Means se applicabile e visualizzabile
        if algoritmo == "K-Means" and centri_cluster_scalati is not None:
            centri_per_plot = None
            if dim_reduction == "PCA":
                # Ottieni il modello PCA usato dalla cache
                # Per proiettare i centroidi, abbiamo bisogno del modello PCA "addestrato"
                # Poiché `riduci_dimensioni` è @st.cache_data, possiamo ri-eseguirla
                # per ottenere il riduttore. Un po' inefficiente ma funzionale con cache.
                temp_X_ridotto, _ = riduci_dimensioni(X_scalato_ds, "PCA", stato_casuale_ds)
                temp_pca_model = PCA(n_components=min(X_scalato_ds.shape[1], 2), random_state=stato_casuale_ds)
                temp_pca_model.fit(X_scalato_ds) # Fit sui dati scalati originali
                if temp_pca_model.n_components_ >= 2:
                    centri_per_plot = temp_pca_model.transform(centri_cluster_scalati)
            elif dim_reduction == "Caratteristiche Originali":
                # Se visualizziamo le caratteristiche originali, prendiamo i centroidi scalati
                # ma mostriamo solo le prime due dimensioni per coerenza con il plot.
                if centri_cluster_scalati.shape[1] >= 2:
                    centri_per_plot = centri_cluster_scalati[:, :2]

            if centri_per_plot is not None and centri_per_plot.shape[1] >= 2:
                fig.add_trace(go.Scatter(x=centri_per_plot[:, 0], y=centri_per_plot[:, 1],
                                         mode='markers',
                                         marker=dict(symbol='star', size=15, color='red', line=dict(width=2, color='DarkSlateGrey')),
                                         name='Centroidi K-Means'
                                         ))
            elif dim_reduction == "t-SNE":
                st.info("I centroidi K-Means non sono direttamente interpretabili nello spazio t-SNE e non vengono visualizzati.")


        fig.update_layout(hovermode="closest", legend_title_text='ID Cluster')
        st.plotly_chart(fig, use_container_width=True)

with col_metriche:
    st.subheader("Metriche di Valutazione")
    if algoritmo == "K-Means":
        st.metric(label="Cluster Richiesti (K)", value=parametri_algo_selezionato['n_clusters'])
        if inertia_val is not None:
            st.metric(label="Inerzia (WCSS)", value=f"{inertia_val:.2f}",
                      help="Somma delle distanze quadrate dei punti dai centroidi del proprio cluster. Valori più bassi indicano cluster più compatti.")
    elif algoritmo == "DBSCAN":
        st.metric(label="Epsilon (eps)", value=f"{parametri_algo_selezionato['eps']:.2f}")
        st.metric(label="Min Samples", value=parametri_algo_selezionato['min_samples'])

    st.metric(label="Cluster Rilevati (escluso Rumore)", value=metriche_risultati['Numero di Cluster Rilevati'])
    st.metric(label="Punti di Rumore", value=metriche_risultati['Punti di Rumore'])

    if not np.isnan(metriche_risultati['Silhouette Score']):
        st.metric("Silhouette Score", f"{metriche_risultati['Silhouette Score']:.3f}",
                  help="Misura quanto bene i punti sono raggruppati all'interno dei cluster e separati da altri cluster (-1 a +1). Più alto è, meglio definiti sono i cluster.")
    else:
        st.info("Silhouette Score non calcolabile (es. 1 solo cluster, tutti rumore, o dati insufficienti).")

    if not np.isnan(metriche_risultati['Indice Davies-Bouldin']):
        st.metric("Indice Davies-Bouldin", f"{metriche_risultati['Indice Davies-Bouldin']:.3f}",
                  help="Valuta la somiglianza media tra i cluster. Valori più bassi indicano un clustering migliore (0 a ∞).")
    else:
        st.info("Indice Davies-Bouldin non calcolabile (es. 1 solo cluster, tutti rumore, o dati insufficienti).")

    if not np.isnan(metriche_risultati['Indice Calinski-Harabasz']):
        st.metric("Indice Calinski-Harabasz", f"{metriche_risultati['Indice Calinski-Harabasz']:.3f}",
                  help="Relazione tra la dispersione all'interno dei cluster e la dispersione tra i cluster. Valori più alti indicano cluster più densi e ben separati (0 a ∞).")
    else:
        st.info("Indice Calinski-Harabasz non calcolabile (es. 1 solo cluster, tutti rumore, o dati insufficienti).")


---
st.header("📈 Approfondimento sui Cluster")

tab_profilazione, tab_distribuzione, tab_dati, tab_docs = st.tabs([
    "🎯 Profilazione Dettagliata",
    "📊 Distribuzione Caratteristiche",
    "📋 Dati Completi",
    "📚 Documentazione Algoritmi"
])

with tab_profilazione:
    st.subheader("Profili Medi dei Segmenti Clienti")
    # Filtra il rumore (-1) dalla profilazione
    df_clusterizzato_senza_rumore = df_originale[df_originale['etichetta_cluster'] != -1]
    
    if not df_clusterizzato_senza_rumore.empty:
        # Calcola le medie per le caratteristiche numeriche
        profili_numerici = df_clusterizzato_senza_rumore.groupby('etichetta_cluster')[caratteristiche_numeriche].mean().round(2)
        st.write("Medie delle Caratteristiche Numeriche per Cluster:")
        st.dataframe(profili_numerici)

        # Normalizza i profili per il grafico radar per una migliore comparazione
        # Non scalare le caratteristiche originali, ma crea una copia per la normalizzazione del plot
        profili_normalizzati = profili_numerici.copy()
        for col in profili_normalizzati.columns:
            min_val = profili_normalizzati[col].min()
            max_val = profili_normalizzati[col].max()
            if max_val - min_val > 0:
                profili_normalizzati[col] = (profili_normalizzati[col] - min_val) / (max_val - min_val)
            else:
                profili_normalizzati[col] = 0.5 # Se tutti i valori sono uguali, mettili a metà scala

        if len(profili_normalizzati) > 0 and len(caratteristiche_numeriche) > 2:
            # Resetta l'indice per far diventare 'etichetta_cluster' una colonna normale
            df_radar = profili_normalizzati.reset_index().melt(id_vars=['etichetta_cluster'], var_name='Caratteristica', value_name='Valore Normalizzato')
            fig_radar = px.line_polar(df_radar,
                                     r='Valore Normalizzato', theta='Caratteristica', line_group='etichetta_cluster', color='etichetta_cluster',
                                     line_close=True, title="Profili Radar dei Cluster (Valori Normalizzati)",
                                     hover_name="etichetta_cluster",
                                     hover_data={"Valore Normalizzato": ':.2f'})
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Non abbastanza cluster o caratteristiche (minimo 3) per creare un grafico radar.")


        # Profilazione per caratteristiche categoriche
        st.subheader("Distribuzione di Genere per Cluster")
        if 'Genere' in df_originale.columns:
            if not df_clusterizzato_senza_rumore.empty:
                genere_dist = pd.crosstab(df_clusterizzato_senza_rumore['etichetta_cluster'], df_clusterizzato_senza_rumore['Genere'], normalize='index').round(2)
                st.dataframe(genere_dist)
                fig_genere = px.bar(genere_dist.reset_index().melt(id_vars=['etichetta_cluster']),
                                    x='etichetta_cluster', y='value', color='Genere',
                                    title="Distribuzione Genere per Cluster", barmode='group',
                                    labels={'value': 'Percentuale'})
                fig_genere.update_yaxes(tickformat=".0%") # Formatta l'asse Y come percentuale
                st.plotly_chart(fig_genere)
            else:
                st.info("Nessun dato clusterizzato valido per la distribuzione di genere.")

        st.subheader("Distribuzione Membro Carta Fedeltà per Cluster")
        if 'Membro Carta Fedeltà' in df_originale.columns:
            if not df_clusterizzato_senza_rumore.empty:
                carta_fedelta_dist = pd.crosstab(df_clusterizzato_senza_rumore['etichetta_cluster'], df_clusterizzato_senza_rumore['Membro Carta Fedeltà'], normalize='index').round(2)
                st.dataframe(carta_fedelta_dist)
                fig_carta_fedelta = px.bar(carta_fedelta_dist.reset_index().melt(id_vars=['etichetta_cluster']),
                                          x='etichetta_cluster', y='value', color='Membro Carta Fedeltà',
                                          title="Distribuzione Membro Carta Fedeltà per Cluster", barmode='group',
                                          labels={'value': 'Percentuale'})
                fig_carta_fedelta.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_carta_fedelta)
            else:
                st.info("Nessun dato clusterizzato valido per la distribuzione carta fedeltà.")
    else:
        st.info("Nessun cluster valido trovato (tutti i punti sono rumore o non ci sono cluster) per la profilazione.")


with tab_distribuzione:
    st.subheader("Esplorazione delle Distribuzioni delle Caratteristiche")
    
    if not df_originale.empty:
        caratteristica_selezionata = st.selectbox("Seleziona una caratteristica da analizzare:", caratteristiche_numeriche)

        fig_box = px.box(df_originale, x="etichetta_cluster", y=caratteristica_selezionata, color="etichetta_cluster",
                         title=f"Distribuzione di '{caratteristica_selezionata}' per Cluster",
                         labels={"etichetta_cluster": "Cluster ID"})
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        st.subheader("Conteggio dei Punti per Cluster")
        counts = df_originale['etichetta_cluster'].value_counts().sort_index().rename("Numero Punti")
        st.dataframe(counts)

        fig_bar_counts = px.bar(counts.reset_index(), x='index', y='Numero Punti', color='index',
                                title="Conteggio Punti per ID Cluster",
                                labels={'index': 'Cluster ID'})
        st.plotly_chart(fig_bar_counts, use_container_width=True)
    else:
        st.info("Nessun dato disponibile per esplorare le distribuzioni.")


with tab_dati:
    st.subheader("Dati di Esempio con Assegnazione Cluster")
    if not df_originale.empty:
        st.dataframe(df_originale.round(2)) # Arrotonda tutto il dataframe a 2 decimali

        csv = df_originale.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Scarica dati clusterizzati (CSV)",
            data=csv,
            file_name="clienti_clusterizzati.csv",
            mime="text/csv"
        )

        st.subheader("Matrice di Correlazione delle Caratteristiche Numeriche")
        matrice_correlazione = df_originale[caratteristiche_numeriche].corr().round(2) # Arrotonda la matrice di correlazione
        fig_corr = px.imshow(matrice_correlazione, text_auto=True, color_continuous_scale="RdBu",
                             title="Matrice di Correlazione delle Caratteristiche")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Nessun dato disponibile.")


with tab_docs:
    st.header("📚 Documentazione Algoritmi di Clustering")

    st.subheader("Informazioni su Questo Strumento")
    st.markdown("""
    Questo strumento interattivo permette agli analisti retail di:
    - Esplorare la segmentazione dei clienti utilizzando diversi algoritmi su dati simulati.
    - Confrontare le performance degli algoritmi tramite metriche multiple.
    - Profilare i segmenti di clienti basandosi sulle loro caratteristiche per strategie mirate.
    - Ottenere una comprensione visiva del funzionamento degli algoritmi.

    **Caratteristiche Chiave:**
    - Generazione di dati simulati di clienti con archetipi realistici.
    - Due algoritmi di clustering popolari: **K-Means** e **DBSCAN**.
    - Diverse tecniche di riduzione di dimensionalità (PCA, t-SNE) per la visualizzazione.
    - Visualizzazioni interattive con Plotly.
    - Profilazione dettagliata dei cluster con medie e distribuzioni.
    - Calcolo di metriche di valutazione del clustering.
    """)

    st.subheader("Descrizioni delle Caratteristiche dei Clienti Simulate")
    descrizioni_caratteristiche = {
        "Età": "Età del cliente in anni.",
        "Genere": "Genere del cliente (Maschio/Femmina).",
        "Reddito Annuo ($)": "Reddito annuo stimato del nucleo familiare in dollari.",
        "Visite Online Mensili": "Numero di visite al negozio online o all'app al mese.",
        "Dimensione Cestino Media ($)": "Spesa media per ogni singolo acquisto (valore del cestino).",
        "Percentuale Acquisti Bio": "La percentuale di prodotti biologici acquistati dal cliente (0-1, dove 1 è 100%).",
        "Sensibilità allo Sconto": "La probabilità che il cliente sia influenzato dagli sconti o promozioni (0-1).",
        "Visite Mensili al Negozio": "Numero di visite al negozio fisico al mese.",
        "Punteggio Lealtà Brand": "Un punteggio che indica la preferenza del cliente per brand specifici rispetto a marche private (0-1).",
        "Membro Carta Fedeltà": "Indica se il cliente possiede una carta fedeltà (True/False).",
        "Segmento Vero": "Il segmento di cliente predefinito in fase di generazione (utile per la comparazione in dati simulati)."
    }

    for feat, desc in descrizioni_caratteristiche.items():
        st.markdown(f"**{feat}**: {desc}")

    st.subheader("Guida agli Algoritmi di Clustering")
    st.markdown("""
    **K-Means**:
    - **Funzionamento**: Un algoritmo di partizionamento che mira a dividere $N$ osservazioni in $K$ cluster, dove ogni osservazione appartiene al cluster con il centroide (media dei punti del cluster) più vicino.
    - **Punti Chiave**:
        - Richiede di specificare il **numero di cluster ($K$)** in anticipo.
        - Assume cluster di forma **sferica/globulare** e di dimensioni simili.
        - Sensibile alla **posizione iniziale dei centroidi** (per questo l'opzione `n_init='auto'` esegue l'algoritmo più volte e sceglie il migliore).
        - Sensibile agli **outlier**.
        - È generalmente **veloce** e scalabile per grandi dataset con un $K$ ragionevole.

    **DBSCAN**:
    - **Funzionamento**: Un algoritmo di clustering basato sulla densità che raggruppa i punti che sono vicini tra loro e identifica i punti isolati come rumore. Non richiede di specificare il numero di cluster.
    - **Concetti Chiave**:
        - **Epsilon ($\epsilon$)**: Il raggio massimo per considerare due campioni come vicini.
        - **Min Samples (MinPts)**: Il numero minimo di campioni in un raggio $\epsilon$ affinché un punto sia considerato un "core point".
    - **Tipi di Punti**:
        - **Core Point**: Ha almeno `MinPts` punti nel suo $\epsilon$-vicinato.
        - **Border Point**: Non è un core point, ma è nel $\epsilon$-vicinato di un core point.
        - **Noise Point (Outlier)**: Non è né core né border.
    - **Punti Chiave**:
        - **Non richiede di specificare il numero di cluster** in anticipo.
        - Può trovare cluster di **forma arbitraria**.
        - È **robusto agli outlier**, identificandoli esplicitamente.
        - La sua performance dipende molto dalla scelta di `eps` e `MinPts`.
        - Può faticare con cluster di **densità molto diverse**.
    """)

---
st.markdown("""
    *Strumento Avanzato di Analisi Retail* | Creato con Streamlit |
    [Repository GitHub](https://github.com/tuo-repository)
""")
