import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE # TSNE è stato spostato in sklearn.manifold
import plotly.express as px # Usiamo Plotly per grafici più interattivi

# --- Configurazione Pagina Streamlit ---
st.set_page_config(layout="wide", page_title="Analisi Avanzata della Segmentazione Clienti", page_icon="🛒")

# --- Titolo e Introduzione ---
st.title("🛒 Analisi Avanzata della Segmentazione Clienti per il Retail")
st.markdown("""
Questa applicazione interattiva ti permette di esplorare algoritmi di clustering avanzati su dati **simulati di clienti di supermercati**.
Scopri segmenti di clienti nascosti e ottieni insight utili per strategie di marketing mirate.
""")

# --- Funzione per Generare Dati Simulati di Clienti Retail ---
# Utilizzeremo questa funzione al posto dei dataset sintetici make_blobs, make_moons, ecc.
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
    # Assicurati che il numero di campioni sia distribuito tra gli archetipi
    campioni_per_tipo = n_campioni // len(archetipi)
    # Calcola i rimanenti per distribuirli tra i primi archetipi
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

    return df, X_scalato, caratteristiche_numeriche

# --- Funzione per la Riduzione Dimensionalità ---
@st.cache_data
def riduci_dimensioni(X, metodo, stato_casuale):
    if metodo == "PCA":
        # Assicurati che n_components sia <= min(n_samples, n_features)
        n_components_pca = min(X.shape[1], 2)
        if n_components_pca < 2 and X.shape[1] >= 1:
            st.warning("PCA richiede almeno 2 caratteristiche per la visualizzazione 2D. Verrà usata la prima caratteristica disponibile.")
            return X[:, :1], f"PCA (1 Componente)" # Ritorna una sola dimensione se non ce ne sono due
        elif n_components_pca < 1:
            return np.array([]).reshape(X.shape[0], 0), "Nessuna Dimensione"
        riduttore = PCA(n_components=n_components_pca, random_state=stato_casuale)
        ridotto = riduttore.fit_transform(X)
        if n_components_pca == 2:
            varianza_spiegata = riduttore.explained_variance_ratio_.sum() * 100
            return ridotto, f"PCA (Varianza Spiegata: {varianza_spiegata:.1f}%)"
        else: # Solo 1 componente
            return ridotto, f"PCA (1 Componente)"
    elif metodo == "t-SNE":
        # Per t-SNE, perplexity deve essere < numero di campioni e > 1
        perplexity_val = min(30, max(2, len(X) - 1)) # Assicurati che perplexity sia almeno 2
        if len(X) <= 1:
            st.warning("t-SNE richiede almeno 2 campioni per la visualizzazione. Verranno usate le caratteristiche originali.")
            return X[:, :2] if X.shape[1] >=2 else X, "Caratteristiche Originali (Prime 2)" # Fallback se non ci sono abbastanza campioni
        riduttore = TSNE(n_components=2, random_state=stato_casuale, perplexity=perplexity_val)
        ridotto = riduttore.fit_transform(X)
        return ridotto, "t-SNE"
    else: # Caratteristiche Originali
        if X.shape[1] < 2:
            st.warning("Il dataset ha meno di 2 caratteristiche. Impossibile visualizzare in 2D le 'Caratteristiche Originali'.")
            return X[:, :1] if X.shape[1] >=1 else X, "Caratteristiche Originali (Prima 1)"
        return X[:, :2], "Caratteristiche Originali (Prime 2)" # Prende solo le prime 2 per la visualizzazione

# --- Funzione per Eseguire il Clustering ---
def esegui_clustering(X_scalato, algoritmo, params, stato_casuale):
    etichette = np.array([])
    modello = None
    centri = None
    inertia = None # Solo per K-Means

    if X_scalato.shape[0] == 0:
        st.warning("Nessun punto dati da clusterizzare. Si prega di aumentare il 'Numero di clienti simulati'.")
        return np.array([-1]*X_scalato.shape[0]), None, None, None

    if algoritmo == "K-Means":
        # Assicurati che n_clusters non superi il numero di campioni
        n_clusters_effettivo = min(params['n_clusters'], X_scalato.shape[0])
        if n_clusters_effettivo < 1:
            st.warning("Il numero di cluster K-Means non può essere inferiore a 1.")
            etichette = np.zeros(X_scalato.shape[0], dtype=int)
            centri = np.mean(X_scalato, axis=0).reshape(1, -1) if X_scalato.shape[0] > 0 else np.array([])
            inertia = 0
        else:
            modello = KMeans(
                n_clusters=n_clusters_effettivo,
                init=params['init_method'],
                max_iter=params['max_iter'], # Utilizzo del parametro iterazioni
                random_state=stato_casuale,
                n_init='auto' # Usa 'auto' per le versioni recenti di scikit-learn
            )
            etichette = modello.fit_predict(X_scalato)
            centri = modello.cluster_centers_ if hasattr(modello, 'cluster_centers_') else None
            inertia = modello.inertia_ if hasattr(modello, 'inertia_') else None

    elif algoritmo == "DBSCAN":
        if params['min_samples'] >= X_scalato.shape[0] or params['eps'] <= 0:
             st.warning(f"DBSCAN: Parametri non validi (min_samples={params['min_samples']}, eps={params['eps']}). Tutti i punti saranno rumore o assegnati a un singolo cluster.")
             etichette = np.array([-1]*X_scalato.shape[0]) # Tutti rumore
        else:
            modello = DBSCAN(
                eps=params['eps'],
                min_samples=params['min_samples'],
                metric=params['metric']
            )
            etichette = modello.fit_predict(X_scalato)
            centri = None # DBSCAN non ha centri espliciti

    return etichette, modello, centri, inertia

# --- Funzione per Calcolare le Metriche di Clustering ---
def calcola_metriche(X, etichette):
    metriche = {}

    # Filtra per escludere il rumore (-1) dal calcolo delle metriche intrinseche
    # che richiedono assegnazioni a cluster validi
    X_validi = X[etichette != -1]
    etichette_valide = etichette[etichette != -1]

    # Conteggio cluster reali (escludendo il rumore)
    etichette_uniche = set(etichette_valide)
    n_cluster_reali = len(etichette_uniche)

    # Calcola le metriche solo se ci sono almeno 2 cluster validi e punti sufficienti
    if n_cluster_reali >= 2 and len(etichette_valide) > 1:
        try:
            metriche['Silhouette Score'] = silhouette_score(X_validi, etichette_valide)
        except Exception:
            metriche['Silhouette Score'] = np.nan

        try:
            metriche['Indice Davies-Bouldin'] = davies_bouldin_score(X_validi, etichette_valide)
        except Exception:
            metriche['Indice Davies-Bouldin'] = np.nan

        try:
            metriche['Indice Calinski-Harabasz'] = calinski_harabasz_score(X_validi, etichette_valide)
        except Exception:
            metriche['Indice Calinski-Harabasz'] = np.nan
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
    livello_rumore_ds = st.slider("Livello di rumore (%)", 0, 30, 5) # Rinominato per evitare conflitto
    stato_casuale_ds = st.slider("Seed casuale dati", 0, 100, 42) # Rinominato

    st.markdown("---")

    st.subheader("2. Selezione Algoritmo")
    algoritmo = st.radio("Algoritmo di Clustering:", ["K-Means", "DBSCAN"], index=0)

    st.markdown("---")

    st.subheader(f"3. Parametri {algoritmo}")
    parametri_algo_selezionato = {}
    if algoritmo == "K-Means":
        parametri_algo_selezionato['n_clusters'] = st.slider("Numero di Cluster (K)", 2, 10, 6)
        parametri_algo_selezionato['init_method'] = st.selectbox("Metodo di Inizializzazione", ["k-means++", "random"])
        parametri_algo_selezionato['max_iter'] = st.slider("Max Iterazioni", 100, 500, 300) # Nuovo parametro per iterazioni
        # kmeans_random_state_param non è più direttamente un parametro dell'algoritmo, ma lo usiamo nel seed globale
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

# --- Genera e Scala i Dati ---
with st.spinner('Generazione dati clienti e scaling...'):
    df_originale, X_scalato_ds, caratteristiche_numeriche = genera_dati_retail(n_campioni, livello_rumore_ds, stato_casuale_ds)

# --- Riduci la Dimensionalità ---
with st.spinner(f'Riduzione dimensionalità con {dim_reduction}...'):
    # X_ridotto avrà sempre 2 colonne se il metodo è PCA o t-SNE e i dati lo consentono.
    # Altrimenti, potrebbe avere 1 colonna o essere vuoto.
    X_ridotto_ds, metodo_riduzione_etichetta = riduci_dimensioni(X_scalato_ds, dim_reduction, stato_casuale_ds)

# --- Esegui il Clustering ---
with st.spinner(f'Esecuzione clustering con {algoritmo}...'):
    etichette_pred, modello_cluster, centri_cluster, inertia_val = esegui_clustering(
        X_scalato_ds, algoritmo, parametri_algo_selezionato, stato_casuale_ds # Usiamo stato_casuale_ds come seed globale
    )
df_originale['etichetta_cluster'] = etichette_pred

# --- Calcola le Metriche ---
metriche_risultati = calcola_metriche(X_scalato_ds, etichette_pred)

# --- Visualizzazione dei Cluster con Plotly ---
st.header(f"🚀 Risultati del Clustering: {algoritmo}")

col_grafico, col_metriche = st.columns([2, 1])

with col_grafico:
    st.subheader("Grafico dei Cluster Rilevati")

    if X_ridotto_ds.shape[1] < 2:
        st.warning("Impossibile generare un grafico 2D con meno di 2 dimensioni. Riprova con un altro metodo di riduzione o più caratteristiche.")
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

        # Aggiungi i centroidi per K-Means, se disponibili e visualizzabili in 2D
        if algoritmo == "K-Means" and centri_cluster is not None and centri_cluster.shape[1] >= 2:
            # Dobbiamo proiettare i centroidi nello spazio ridotto per visualizzarli correttamente
            # Qui si assume che X_ridotto_ds sia il risultato della proiezione di X_scalato_ds
            # e che il 'riduttore' (PCA/t-SNE) sia lo stesso usato.
            # Per semplicità, proiettiamo i centroidi nello stesso modo:
            # (ATTENZIONE: t-SNE non ha una trasformazione diretta per nuovi punti/centri)
            if dim_reduction == "PCA":
                # Ottieni il riduttore usato da @st.cache_data
                scaler = StandardScaler() # Re-inizializza per evitare problemi di cache tra chiamate
                scaler.fit(df_originale[caratteristiche_numeriche])
                X_scaled_for_pca = scaler.transform(df_originale[caratteristiche_numeriche])

                pca_model = PCA(n_components=min(X_scaled_for_pca.shape[1], 2), random_state=stato_casuale_ds)
                pca_model.fit(X_scaled_for_pca) # Fit sulla base dei dati scalati

                centri_ridotti = pca_model.transform(centri_cluster)

                if centri_ridotti.shape[1] >= 2:
                    fig.add_trace(px.scatter(x=centri_ridotti[:, 0], y=centri_ridotti[:, 1],
                                             marker=dict(symbol='star', size=15, color='red', line=dict(width=2, color='DarkSlateGrey')),
                                             mode='markers', name='Centroidi K-Means'
                                             ).data[0])
            # Per t-SNE i centroidi non sono significativi nello spazio ridotto in questo modo diretto.
            # Per "Caratteristiche Originali", i centroidi sono già nello spazio originale,
            # ma stiamo visualizzando solo le prime 2 caratteristiche.
            elif dim_reduction == "Caratteristiche Originali":
                 if centri_cluster.shape[1] >= 2:
                    fig.add_trace(px.scatter(x=centri_cluster[:, 0], y=centri_cluster[:, 1],
                                             marker=dict(symbol='star', size=15, color='red', line=dict(width=2, color='DarkSlateGrey')),
                                             mode='markers', name='Centroidi K-Means'
                                             ).data[0])

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


st.markdown("---")
st.header("📈 Approfondimento sui Cluster")

tab_profilazione, tab_distribuzione, tab_dati = st.tabs([
    "🎯 Profilazione Dettagliata",
    "📊 Distribuzione Caratteristiche",
    "📋 Dati Completii"
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

        # Aggiungi un grafico a radar (o a ragnatela) per i profili
        if len(profili_numerici) > 0 and len(caratteristiche_numeriche) > 2:
            fig_radar = px.line_polar(profili_numerici.reset_index().melt(id_vars=['etichetta_cluster'], var_name='Caratteristica', value_name='Valore Medio'),
                                     r='Valore Medio', theta='Caratteristica', line_group='etichetta_cluster', color='etichetta_cluster',
                                     line_close=True, title="Profili Radar dei Cluster (Medie Scalate)")
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Non abbastanza cluster o caratteristiche per creare un grafico radar.")


        # Profilazione per caratteristiche categoriche
        st.subheader("Distribuzione di Genere per Cluster")
        if 'Genere' in df_originale.columns:
            genere_dist = pd.crosstab(df_clusterizzato_senza_rumore['etichetta_cluster'], df_clusterizzato_senza_rumore['Genere'], normalize='index').round(2)
            st.dataframe(genere_dist)
            fig_genere = px.bar(genere_dist.reset_index().melt(id_vars=['etichetta_cluster']),
                                x='etichetta_cluster', y='value', color='Genere',
                                title="Distribuzione Genere per Cluster", barmode='group')
            st.plotly_chart(fig_genere)

        st.subheader("Distribuzione Membro Carta Fedeltà per Cluster")
        if 'Membro Carta Fedeltà' in df_originale.columns:
            carta_fedelta_dist = pd.crosstab(df_clusterizzato_senza_rumore['etichetta_cluster'], df_clusterizzato_senza_rumore['Membro Carta Fedeltà'], normalize='index').round(2)
            st.dataframe(carta_fedelta_dist)
            fig_carta_fedelta = px.bar(carta_fedelta_dist.reset_index().melt(id_vars=['etichetta_cluster']),
                                      x='etichetta_cluster', y='value', color='Membro Carta Fedeltà',
                                      title="Distribuzione Membro Carta Fedeltà per Cluster", barmode='group')
            st.plotly_chart(fig_carta_fedelta)
    else:
        st.info("Nessun cluster valido trovato (tutti i punti sono rumore o non ci sono cluster).")


with tab_distribuzione:
    st.subheader("Esplorazione delle Distribuzioni delle Caratteristiche")
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

with tab_dati:
    st.subheader("Dati di Esempio con Assegnazione Cluster")
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
                         title="Matrice di Correlazione")
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")
st.caption("Creato con Streamlit per scopi didattici e dimostrativi.")
