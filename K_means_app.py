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
    algoritmo_selezionato = st.radio("Algoritmo:", ["K-Means", "DBSCAN"])

    # Selezione features
    features_disponibili = [
        "Età", "Reddito Annuale", "Visite Mensili",
        "Spesa Media", "% Acquisti Online",
        "Frequenza Acquisti", "Valore Carrello"
    ]
    features_selezionate = st.multiselect(
        "Seleziona caratteristiche per il clustering:",
        features_disponibili,
        default=["Età", "Reddito Annuale", "Visite Mensili"],
        max_selections=5
    )

    if algoritmo_selezionato == "K-Means":
        num_cluster = st.slider("Numero di cluster:", 2, 10, 4)
        max_iterazioni = st.slider("Massime iterazioni:", 1, 30, 10) # Aumentato range per più visibilità
        random_state_kmeans = st.number_input("Seed per riproducibilità (K-Means):", min_value=0, value=42)

        # Visualizzazione evoluzione cluster
        mostra_evoluzione_cluster = st.checkbox("Mostra evoluzione iterazioni (K-Means)")
        if mostra_evoluzione_cluster:
            num_iter_viz = st.slider("Numero di iterazioni da visualizzare:", 1, max_iterazioni, min(5, max_iterazioni))
    else: # DBSCAN
        epsilon = st.slider("Raggio (epsilon):", 0.1, 2.0, 0.5, step=0.05) # Aumentato range
        min_campioni = st.slider("Minimo campioni:", 2, 20, 5)

    riduzione_dimensionale_metodo = st.selectbox("Metodo di riduzione dimensionale:", ["PCA", "t-SNE"])

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
    st.warning("Seleziona almeno 2 caratteristiche per il clustering!")
    st.stop()

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(df[features_selezionate])

# Riduzione dimensionale
if riduzione_dimensionale_metodo == "PCA":
    riduttore = PCA(n_components=2)
    X_ridotto = riduttore.fit_transform(X)
    varianza_spiegata = riduttore.explained_variance_ratio_.sum() * 100
    metodo_riduzione_str = f"PCA (Varianza: {varianza_spiegata:.1f}%)"
else: # t-SNE
    # Per t-SNE, è bene impostare un random_state per riproducibilità
    riduttore = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate='auto', init='random')
    X_ridotto = riduttore.fit_transform(X)
    metodo_riduzione_str = "t-SNE"

# Clustering
modello_clustering = None # Inizializza a None
labels = np.array([]) # Inizializza labels a un array vuoto
centers = None

if algoritmo_selezionato == "K-Means":
    modello_clustering = KMeans(n_clusters=num_cluster, max_iter=max_iterazioni, n_init='auto', random_state=random_state_kmeans)
    labels = modello_clustering.fit_predict(X)
    centers = scaler.inverse_transform(modello_clustering.cluster_centers_)

    # Visualizzazione evoluzione iterazioni K-Means
    if mostra_evoluzione_cluster:
        st.subheader("Risultati K-Means con diverse iterazioni massime")
        st.info("Nota: Ogni grafico mostra il risultato finale di un'esecuzione K-Means interrotta al numero specificato di iterazioni, non l'evoluzione passo-passo di una singola esecuzione.")

        figs = []
        # Per mostrare l'evoluzione, addestriamo un nuovo modello K-Means per ogni iterazione
        # Questo NON mostra l'evoluzione di UN SINGOLO modello, ma il risultato di modelli
        # che si fermano a X iterazioni. Per una vera evoluzione, servirebbe un callback
        # che scikit-learn non espone direttamente.
        for i in range(1, num_iter_viz + 1):
            # Usiamo n_init=1 e init='random' per simulare un'inizializzazione specifica per ogni iterazione
            # e mostrare la convergenza in modo più evidente.
            model_temp = KMeans(n_clusters=num_cluster, max_iter=i, n_init=1, init='random', random_state=random_state_kmeans)
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
        cols = st.columns(num_iter_viz)
        for i, (col, fig) in enumerate(zip(cols, figs), 1):
            with col:
                st.pyplot(fig)
                # Aggiungi pulsante per scaricare l'immagine
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=100)
                st.download_button(
                    label=f"Scarica iterazione {i}",
                    data=buf.getvalue(), # Usa getvalue() per BytesIO
                    file_name=f"kmeans_iterazione_{i}.png",
                    mime="image/png"
                )
elif algoritmo_selezionato == "DBSCAN":
    modello_clustering = DBSCAN(eps=epsilon, min_samples=min_campioni)
    labels = modello_clustering.fit_predict(X)
    centers = None # DBSCAN non ha centroidi espliciti

# Calcolo metriche con gestione degli errori
silhouette_score_value = None
unique_labels = np.unique(labels)
# Il Silhouette Score richiede almeno 2 cluster e non tutti i punti nel cluster di rumore (-1)
if len(unique_labels) > 1 and (len(unique_labels) > 2 or -1 not in unique_labels):
    try:
        # Se DBSCAN genera un cluster -1 (rumore), lo escludiamo per il calcolo del silhouette score
        # in quanto non rappresenta un cluster vero e proprio.
        if -1 in unique_labels:
            X_filtered = X[labels != -1]
            labels_filtered = labels[labels != -1]
            if len(np.unique(labels_filtered)) > 1: # Assicurati che ci siano ancora almeno 2 cluster dopo il filtraggio
                 silhouette_score_value = silhouette_score(X_filtered, labels_filtered)
        else:
            silhouette_score_value = silhouette_score(X, labels)
    except ValueError:
        silhouette_score_value = None
else:
    silhouette_score_value = None


# Creazione DataFrame per plotting
plot_df = pd.DataFrame({
    "Componente_1": X_ridotto[:, 0],
    "Componente_2": X_ridotto[:, 1],
    "Cluster": labels.astype(str) # Converti in stringa per plotly, specialmente per -1 in DBSCAN
})

# Aggiunta delle features selezionate per l'hover
for feature in features_selezionate:
    plot_df[feature] = df[feature]

# Visualizzazione
tab1, tab2, tab3 = st.tabs(["📊 Risultati", "📈 Analisi", "❓ Guida"])

with tab1:
    st.header("Risultati Clustering")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Plot cluster con gestione degli errori
        try:
            fig = px.scatter(
                plot_df,
                x="Componente_1",
                y="Componente_2",
                color="Cluster",
                title=f"Risultati Clustering ({metodo_riduzione_str})",
                labels={
                    "Componente_1": "Componente 1",
                    "Componente_2": "Componente 2",
                    "Cluster": "Cluster"
                },
                hover_data=features_selezionate,
                # Aggiungi un tooltip per i cluster di rumore di DBSCAN
                color_discrete_map={'-1': 'grey'} if algoritmo_selezionato == "DBSCAN" else None
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Errore nella creazione del grafico: {str(e)}")
            # Fallback a matplotlib
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_ridotto[:, 0], X_ridotto[:, 1], c=labels, cmap='viridis')
            ax.set_title(f"Risultati Clustering ({metodo_riduzione_str})")
            ax.set_xlabel("Componente 1")
            ax.set_ylabel("Componente 2")
            plt.colorbar(scatter, ax=ax, label='Cluster')
            st.pyplot(fig)

    with col2:
        st.subheader("Metriche")
        if silhouette_score_value is not None:
            st.metric("Silhouette Score",
                     f"{silhouette_score_value:.3f}",
                     help="Valore tra -1 e 1: più alto è meglio. I valori negativi indicano che i punti potrebbero essere stati assegnati al cluster sbagliato, mentre i valori vicini a 0 indicano cluster sovrapposti. Per DBSCAN, i punti di rumore (-1) vengono esclusi dal calcolo.")
        else:
            st.metric("Silhouette Score",
                     "N/A",
                     help="Non calcolabile: richiede almeno 2 cluster validi (non solo rumore o un singolo cluster).")

        st.subheader("Distribuzione Cluster")
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        st.bar_chart(cluster_counts)

        if algoritmo_selezionato == "DBSCAN" and '-1' in cluster_counts.index.astype(str):
            st.info("Nota: Il cluster '-1' in DBSCAN rappresenta i punti di rumore (outlier).")

        st.write(f"Caratteristiche utilizzate: {', '.join(features_selezionate)}")

with tab2:
    st.header("Analisi Dettagliata")

    df["Cluster"] = labels # Assegna le etichette dei cluster al DataFrame originale

    st.subheader("Statistiche per Cluster")
    # Filtra il cluster -1 per le statistiche se è presente in DBSCAN
    if algoritmo_selezionato == "DBSCAN" and -1 in df["Cluster"].unique():
        st.dataframe(
            df[df["Cluster"] != -1].groupby("Cluster")[features_selezionate]
            .agg(["mean", "median", "std"])
            .style.background_gradient(cmap="Blues")
        )
        st.info("Le statistiche medie per il cluster '-1' (rumore) non sono incluse in questa tabella, poiché non rappresenta un cluster tradizionale.")
    else:
        st.dataframe(
            df.groupby("Cluster")[features_selezionate]
            .agg(["mean", "median", "std"])
            .style.background_gradient(cmap="Blues")
        )

    st.subheader("Distribuzione Caratteristiche per Cluster")
    caratteristica_selezionata = st.selectbox("Seleziona caratteristica:", features_selezionate)
    fig_box = px.box(df, x="Cluster", y=caratteristica_selezionata, color="Cluster", title=f"Distribuzione di '{caratteristica_selezionata}' per Cluster")
    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.header("Guida all'Uso")

    st.markdown("""
    ### Come utilizzare questo strumento:
    1.  **Seleziona le caratteristiche** che vuoi usare per il clustering (max 5) nella sidebar.
    2.  **Scegli l'algoritmo** preferito nella sidebar:
        * **K-Means**: adatto per cluster sferici di dimensioni simili. Richiede di specificare il numero di cluster (`Numero di cluster`).
        * **DBSCAN**: ideale per cluster di forma arbitraria e per l'identificazione di punti di rumore (outlier). Richiede `Raggio (epsilon)` (distanza massima tra due campioni per essere considerati nella stessa vicinanza) e `Minimo campioni` (numero di campioni in una vicinanza per un punto per essere considerato un punto core).
    3.  **Regola i parametri** specifici dell'algoritmo selezionato nella sidebar.
    4.  L'analisi si aggiornerà automaticamente ad ogni modifica dei parametri.
    """)

    st.subheader("Interpretazione dei Risultati:")
    st.markdown("""
    * **Silhouette Score**: È una metrica che misura la qualità del clustering. Varia tra -1 e 1.
        * **Valori vicino a +1**: Indicano che i punti sono ben separati e formano cluster distinti.
        * **Valori vicino a 0**: Suggeriscono che i cluster si sovrappongono o che i punti si trovano al limite tra due cluster.
        * **Valori vicino a -1**: Indicano che i punti potrebbero essere stati assegnati al cluster sbagliato.
        * Per DBSCAN, i punti di rumore (cluster '-1') non vengono inclusi nel calcolo del Silhouette Score.
    * **Grafici dei Cluster**: Mostrano la distribuzione dei dati nel nuovo spazio a 2 dimensioni dopo la riduzione dimensionale (PCA o t-SNE). Cluster simili indicano segmenti di clienti con comportamenti o caratteristiche simili.
    * **Distribuzione Cluster**: Un istogramma che mostra quanti punti ci sono in ogni cluster.
    * **Statistiche per Cluster**: Le tabelle mostrano le medie, mediane e deviazioni standard delle caratteristiche selezionate per ogni cluster. Questo ti aiuta a capire il "profilo" di ogni segmento di clientela (es. "Clienti giovani ad alto reddito", "Anziani con bassa frequenza di acquisto").
    * **Cluster '-1' (DBSCAN)**: Se utilizzi DBSCAN, potresti vedere un cluster etichettato '-1'. Questi sono i **punti di rumore (outlier)** che non sono stati assegnati a nessun cluster denso.
    """)

    st.subheader("Descrizione Caratteristiche")
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
st.markdown("Progetto di Data Science - Università IULM | [GitHub](https://github.com/)") # Sostituisci con il tuo link GitHub reale
