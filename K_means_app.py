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
import time # Per l'animazione

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

    # Nuovi slider per generazione dati (come nello screenshot)
    st.subheader("Generazione Dati Simulati")
    num_clienti = st.slider("Numero di clienti:", 100, 2000, 1000)
    # Nota: Il numero di features è un po' ridondante dato che le selezioniamo
    # Ma lo includo per replicare lo screenshot, anche se avrà meno impatto reale
    num_features_generazione = st.slider("Numero massimo di features per generazione (solo visivo):", 2, 10, 7) # Corrisponde alle nostre 7 feature

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
        num_cluster = st.slider("Numero di cluster (K):", 2, 10, 4)
        max_iterazioni = st.slider("Massime iterazioni dell'algoritmo:", 1, 50, 20) # Aumentato range per più visibilità
        random_seed_kmeans = st.number_input("Random seed:", min_value=0, value=42)

        # Visualizzazione evoluzione cluster (come nello screenshot)
        mostra_evoluzione_cluster = st.checkbox("Mostra iterazioni dell'algoritmo")
        if mostra_evoluzione_cluster:
            # Qui non abbiamo uno slider per "Numero massimo di iterazioni" separato
            # perché è già 'max_iterazioni' sopra. Manteniamo coerenza.
            st.markdown(f"Numero massimo di iterazioni: **{max_iterazioni}**")

    else: # DBSCAN
        epsilon = st.slider("Raggio (epsilon):", 0.1, 2.0, 0.5, step=0.05) # Aumentato range
        min_campioni = st.slider("Minimo campioni:", 2, 20, 5)


# Generazione dati simulati
@st.cache_data
def genera_dati_dinamici(n_clienti_generati):
    np.random.seed(42) # Usiamo un seed fisso per la riproducibilità della generazione base
    n = n_clienti_generati

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

df = genera_dati_dinamici(num_clienti)

# Verifica che siano selezionate almeno 2 features
if len(features_selezionate) < 2:
    st.warning("Seleziona almeno 2 caratteristiche per il clustering!")
    st.stop()

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(df[features_selezionate])

# Riduzione dimensionale
riduzione_dimensionale_metodo_str = "" # Inizializza
if st.session_state.get('riduzione_dimensionale_metodo', 'PCA') == "PCA": # Per mantenere stato se possibile
    riduttore = PCA(n_components=2)
    X_ridotto = riduttore.fit_transform(X)
    varianza_spiegata = riduttore.explained_variance_ratio_.sum() * 100
    riduzione_dimensionale_metodo_str = f"PCA (Varianza: {varianza_spiegata:.1f}%)"
else: # t-SNE
    riduttore = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate='auto', init='random') # Aggiunto random_state
    X_ridotto = riduttore.fit_transform(X)
    riduzione_dimensionale_metodo_str = "t-SNE"

# Store the selected reduction method in session_state to persist choice for plotting
st.session_state['riduzione_dimensionale_metodo'] = riduzione_dimensionale_metodo_str


# Clustering
modello_clustering = None
labels = np.array([])
centers = None
convergenza_iterazioni = max_iterazioni # Default, sarà aggiornato per K-Means

if algoritmo_selezionato == "K-Means":
    # Pre-calcoliamo tutte le iterazioni per poi navigarle con lo slider
    iterazioni_dati = []
    # Nota: Per replicare la visualizzazione dello screenshot, addestriamo un modello
    # che si ferma ad ogni iterazione. Questo NON è l'andamento di un singolo
    # algoritmo K-Means passo-passo, ma il risultato di K-Means dopo N iterazioni.
    # Una vera animazione dei centroidi richiede un'implementazione più custom.
    
    # Per K-Means, dobbiamo catturare gli stati intermedi se vogliamo un'animazione vera.
    # Scikit-learn KMeans non espone i centroidi ad ogni iterazione.
    # Possiamo solo addestrare un nuovo modello e fermarlo dopo 'i' iterazioni.
    # Oppure, una soluzione più complessa, implementare K-Means a mano.
    # Per ora, simuliamo come nell'esempio dello screenshot, mostrando il risultato
    # di un modello che si ferma a 'i' iterazioni.

    st.session_state['kmeans_states'] = {} # Per memorizzare labels e centers per ogni iterazione
    
    # Eseguo K-Means passo dopo passo per catturare lo stato, ma ogni step è una nuova esecuzione.
    # Questo è un compromesso per mostrare l'effetto delle iterazioni con scikit-learn.
    # N.B.: init='random' e n_init=1 per avere un comportamento più prevedibile per ogni 'step' fittizio.
    # Il random_state per la singola iterazione è importante per la riproducibilità.
    
    # Esegui l'algoritmo completo una volta per ottenere il numero di iterazioni reali
    modello_finale_kmeans = KMeans(n_clusters=num_cluster, max_iter=max_iterazioni, n_init='auto', random_state=random_seed_kmeans)
    modello_finale_kmeans.fit(X)
    convergenza_iterazioni = modello_finale_kmeans.n_iter_
    
    # Ora, per la visualizzazione iterativa, creiamo modelli che si fermano a un certo punto
    for i in range(1, max_iterazioni + 1):
        temp_kmeans = KMeans(n_clusters=num_cluster, max_iter=i, n_init=1, init='random', random_state=random_seed_kmeans)
        temp_labels = temp_kmeans.fit_predict(X)
        temp_centers = temp_kmeans.cluster_centers_ # Questi sono i centri *scalati*
        
        # Salviamo i risultati per l'iterazione corrente
        st.session_state['kmeans_states'][i] = {
            'labels': temp_labels,
            'centers': temp_centers
        }

    # Per il risultato finale mostrato fuori dall'animazione
    labels = modello_finale_kmeans.labels_
    centers = scaler.inverse_transform(modello_finale_kmeans.cluster_centers_)


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
        if -1 in unique_labels:
            X_filtered = X[labels != -1]
            labels_filtered = labels[labels != -1]
            if len(np.unique(labels_filtered)) > 1:
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
    "Cluster": labels.astype(str)
})

for feature in features_selezionate:
    plot_df[feature] = df[feature]

# Visualizzazione
tab1, tab2, tab3 = st.tabs(["📊 Risultati", "📈 Analisi", "❓ Guida"])

with tab1:
    st.header("Risultati Clustering")

    if algoritmo_selezionato == "K-Means" and mostra_evoluzione_cluster:
        st.subheader("Evoluzione delle Iterazioni dell'Algoritmo K-Means")
        st.info(f"L'algoritmo converge in {convergenza_iterazioni} iterazioni.")

        # Slider per selezionare l'iterazione
        iterazione_corrente = st.slider(
            "Seleziona iterazione:",
            1,
            max_iterazioni, # Max slider value
            max_iterazioni # Default value, shows final state
        )

        # Pulsante Avvia Animazione
        if st.checkbox("Avvia Animazione"):
            # Placeholder per il grafico animato
            grafico_placeholder = st.empty()
            for i in range(1, max_iterazioni + 1):
                temp_labels_anim = st.session_state['kmeans_states'][i]['labels']
                temp_centers_anim_scaled = st.session_state['kmeans_states'][i]['centers']
                temp_centers_anim = scaler.inverse_transform(temp_centers_anim_scaled) if temp_centers_anim_scaled is not None else None

                plot_df_anim = pd.DataFrame({
                    "Componente_1": X_ridotto[:, 0],
                    "Componente_2": X_ridotto[:, 1],
                    "Cluster": temp_labels_anim.astype(str)
                })

                fig_anim = px.scatter(
                    plot_df_anim,
                    x="Componente_1",
                    y="Componente_2",
                    color="Cluster",
                    title=f"Iterazione {i}/{max_iterazioni} ({riduzione_dimensionale_metodo_str})",
                    labels={"Componente_1": "Prima Componente Principale", "Componente_2": "Seconda Componente Principale", "Cluster": "Cluster"},
                    hover_data=features_selezionate
                )
                if temp_centers_anim is not None:
                    centers_reduced_anim = riduttore.transform(temp_centers_anim_scaled) # Trasforma i centri scalati
                    fig_anim.add_scatter(x=centers_reduced_anim[:, 0], y=centers_reduced_anim[:, 1],
                                    mode='markers', marker=dict(symbol='x', size=15, color='black'),
                                    name='Centroidi', showlegend=True)

                with grafico_placeholder:
                    st.plotly_chart(fig_anim, use_container_width=True)
                time.sleep(0.5) # Pausa tra un frame e l'altro
            st.success("Animazione completata!")

        # Visualizzazione singola iterazione selezionata (se non in animazione)
        if not st.session_state.get('animazione_in_corso', False): # per non mostrare due grafici contemporaneamente
            selected_labels = st.session_state['kmeans_states'][iterazione_corrente]['labels']
            selected_centers_scaled = st.session_state['kmeans_states'][iterazione_corrente]['centers']
            selected_centers = scaler.inverse_transform(selected_centers_scaled) if selected_centers_scaled is not None else None

            plot_df_selected = pd.DataFrame({
                "Componente_1": X_ridotto[:, 0],
                "Componente_2": X_ridotto[:, 1],
                "Cluster": selected_labels.astype(str)
            })

            fig_selected = px.scatter(
                plot_df_selected,
                x="Componente_1",
                y="Componente_2",
                color="Cluster",
                title=f"Iterazione {iterazione_corrente}/{max_iterazioni} ({riduzione_dimensionale_metodo_str})",
                labels={"Componente_1": "Prima Componente Principale", "Componente_2": "Seconda Componente Principale", "Cluster": "Cluster"},
                hover_data=features_selezionate
            )
            if selected_centers is not None:
                # Trasforma i centroidi scalati con lo stesso riduttore
                centers_reduced = riduttore.transform(selected_centers_scaled)
                fig_selected.add_scatter(x=centers_reduced[:, 0], y=centers_reduced[:, 1],
                                mode='markers', marker=dict(symbol='x', size=15, color='black'),
                                name='Centroidi', showlegend=True)

            st.plotly_chart(fig_selected, use_container_width=True)

    else: # K-Means senza evoluzione o DBSCAN
        col1, col2 = st.columns([2, 1])

        with col1:
            try:
                fig = px.scatter(
                    plot_df,
                    x="Componente_1",
                    y="Componente_2",
                    color="Cluster",
                    title=f"Risultati Clustering ({riduzione_dimensionale_metodo_str})",
                    labels={
                        "Componente_1": "Prima Componente Principale",
                        "Componente_2": "Seconda Componente Principale",
                        "Cluster": "Cluster"
                    },
                    hover_data=features_selezionate,
                    color_discrete_map={'-1': 'grey'} if algoritmo_selezionato == "DBSCAN" else None
                )
                # Aggiungi centroidi per K-Means nel grafico finale
                if algoritmo_selezionato == "K-Means" and centers is not None:
                    # I centri devono essere scalati prima di essere ridotti dimensionalmente
                    centers_scaled = scaler.transform(centers)
                    centers_reduced = riduttore.transform(centers_scaled) # Trasforma i centri scalati
                    fig.add_scatter(x=centers_reduced[:, 0], y=centers_reduced[:, 1],
                                    mode='markers', marker=dict(symbol='x', size=15, color='black'),
                                    name='Centroidi', showlegend=True)

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Errore nella creazione del grafico: {str(e)}")
                # Fallback a matplotlib
                fig, ax = plt.subplots()
                scatter = ax.scatter(X_ridotto[:, 0], X_ridotto[:, 1], c=labels, cmap='viridis')
                ax.set_title(f"Risultati Clustering ({riduzione_dimensionale_metodo_str})")
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
    1.  **Generazione Dati Simulati**: Controlla la dimensione del dataset generato.
    2.  **Seleziona le caratteristiche** che vuoi usare per il clustering (max 5) nella sidebar.
    3.  **Scegli l'algoritmo** preferito nella sidebar:
        * **K-Means**: adatto per cluster sferici di dimensioni simili. Richiede di specificare il numero di cluster (`Numero di cluster`). Puoi anche impostare un `Random seed` per risultati riproducibili.
        * **DBSCAN**: ideale per cluster di forma arbitraria e per l'identificazione di punti di rumore (outlier). Richiede `Raggio (epsilon)` (distanza massima tra due campioni per essere considerati nella stessa vicinanza) e `Minimo campioni` (numero di campioni in una vicinanza per un punto per essere considerato un punto core).
    4.  **Regola i parametri** specifici dell'algoritmo selezionato nella sidebar.
    5.  L'analisi si aggiornerà automaticamente ad ogni modifica dei parametri.
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
        "%"
        "Acquisti Online": "Percentuale di acquisti fatti online",
        "Frequenza Acquisti": "Numero di acquisti mensili",
        "Valore Carrello": "Valore medio del carrello in €"
    }

    for feat in features_selezionate:
        st.markdown(f"**{feat}**: {descrizioni.get(feat, '')}")

# Footer
st.markdown("---")
st.markdown("Progetto di Data Science - Università IULM | [GitHub](https://github.com/)") # Sostituisci con il tuo link GitHub reale
