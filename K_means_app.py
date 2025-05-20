import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# --- NEW CUSTOM DATA GENERATOR BASED ON USER'S SCRIPT ---
def generate_custom_customer_data(n_customers, random_seed, bb_ratio=0.15):
    np.random.seed(random_seed)

    n_bb = int(n_customers * bb_ratio)
    n_non_bb = n_customers - n_bb

    # Bodybuilder
    bb_data = pd.DataFrame({
        'ID_Cliente': [f'BB{i+1:04d}' for i in range(n_bb)], # Renamed customer_id to ID_Cliente
        'is_bodybuilder': 1,
        'protein_spending': np.random.normal(180, 30, n_bb),
        'supplements_spending': np.random.normal(100, 20, n_bb),
        'carb_spending': np.random.normal(80, 15, n_bb),
        'total_visits_per_month': np.random.randint(8, 16, n_bb),
        'avg_basket_size': np.random.normal(25, 5, n_bb),
        'purchase_time_slot': np.random.choice([1, 2], n_bb, p=[0.3, 0.7]) # 1: PM, 2: Sera
    })

    # Non bodybuilder
    non_bb_data = pd.DataFrame({
        'ID_Cliente': [f'NB{i+1:04d}' for i in range(n_non_bb)], # Renamed customer_id to ID_Cliente
        'is_bodybuilder': 0,
        'protein_spending': np.random.normal(60, 20, n_non_bb),
        'supplements_spending': np.random.normal(10, 5, n_non_bb),
        'carb_spending': np.random.normal(100, 20, n_non_bb),
        'total_visits_per_month': np.random.randint(3, 10, n_non_bb),
        'avg_basket_size': np.random.normal(15, 5, n_non_bb),
        'purchase_time_slot': np.random.choice([0, 1, 2], n_non_bb, p=[0.4, 0.4, 0.2]) # 0: AM, 1: PM, 2: Sera
    })

    df = pd.concat([bb_data, non_bb_data]).sample(frac=1, random_state=random_seed).reset_index(drop=True) # Shuffle data

    # Pulizia (no valori negativi)
    cols_to_clip = ['protein_spending', 'supplements_spending', 'carb_spending', 'avg_basket_size']
    for col in cols_to_clip:
        if col in df.columns: # Ensure column exists before clipping
            df[col] = df[col].clip(lower=0)
            
    return df

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="MarketPro: Individuazione Clienti Target")

# --- Titolo e Introduzione ---
st.title("🎯 MarketPro: Individuazione Avanzata Cluster Clienti")
st.markdown("""
Benvenuti nell'analisi avanzata di **MarketPro**! Utilizzeremo i dati simulati dei clienti per identificare specifici pattern di acquisto.
L'obiettivo è segmentare la clientela per strategie di marketing mirate, con un focus sull'individuazione di nicchie di valore come **bodybuilder/appassionati di fitness**.
Esploreremo come **K-Means** e **DBSCAN** possono aiutarci, basandoci sulle feature da voi selezionate.
""")

st.info("💡 **Obiettivo Specifico**: Isolare gruppi di clienti (es. bodybuilder) analizzando le loro abitudini di acquisto. Ad esempio, i bodybuilder dovrebbero mostrare alta spesa in proteine e integratori, e bassa in altre categorie (non modellate qui ma implicite).")

# --- Lista di tutte le feature disponibili per il clustering (escluso ID) ---
ALL_POSSIBLE_FEATURES_FOR_CLUSTERING = [
    'protein_spending', 'supplements_spending', 'carb_spending',
    'total_visits_per_month', 'avg_basket_size', 'purchase_time_slot',
    'is_bodybuilder' # Ground truth label, use with caution for discovery
]
ALL_POSSIBLE_FEATURES_FOR_CLUSTERING.sort()


# --- Sidebar per Controlli Globali ---
with st.sidebar:
    st.header("⚙️ Impostazioni Simulazione Dati")
    n_samples = st.slider("Numero di Clienti Simulati", 100, 2000, 500, help="Quanti profili cliente generare.")
    random_state_data = st.slider("Seed per Generazione Dati", 0, 100, 42, help="Controlla la riproducibilità dei dati generati.")

    st.header("📊 Selezione Feature per Clustering")
    st.markdown("Seleziona da 1 a 10 features per l'analisi. L'ID Cliente è escluso.")

    default_features = ['protein_spending', 'supplements_spending', 'carb_spending', 'total_visits_per_month']
    default_selection = [f for f in default_features if f in ALL_POSSIBLE_FEATURES_FOR_CLUSTERING]

    selected_features = st.multiselect(
        "Scegli le feature per l'analisi di clustering:",
        options=ALL_POSSIBLE_FEATURES_FOR_CLUSTERING,
        default=default_selection,
        max_selections=10, # Max 10 features as per original request
        help="Seleziona le feature per il clustering. 'ID_Cliente' è escluso."
    )

    if not selected_features:
        st.warning("Per favore, seleziona almeno una feature per il clustering.")
        st.stop()

    if 'is_bodybuilder' in selected_features:
        st.warning("⚠️ Hai selezionato 'is_bodybuilder' per il clustering. Questa è la variabile target (ground truth). Includerla aiuterà l'algoritmo a 'trovare' i bodybuilder facilmente, ma non rappresenta uno scenario di scoperta reale. Per una vera scoperta, deselezionala.")


    features_for_clustering = selected_features

    st.markdown("---")
    st.subheader("💡 Suggerimento Algoritmo:")
    num_feat = len(features_for_clustering)
    if num_feat <= 4:
        st.markdown(f"Con **{num_feat}** feature(s), **K-Means** potrebbe essere un buon punto di partenza se ti aspetti cluster sferici e conosci approssimativamente il numero di segmenti (K).")
        st.markdown("**DBSCAN** è utile se cerchi cluster di forme arbitrarie o vuoi identificare automaticamente outliers/nicchie dense.")
    else:
        st.markdown(f"Con **{num_feat}** feature(s), la dimensionalità aumenta. **DBSCAN** può essere più efficace nell'identificare cluster basati sulla densità e forme complesse, oltre a gestire il rumore.")
        st.markdown("**K-Means** è ancora utilizzabile, ma la 'maledizione della dimensionalità' potrebbe renderne i risultati meno intuitivi. Assicurati che K sia scelto con cura.")
    st.markdown("*Questo è solo un suggerimento generico. La scelta migliore dipende dalla natura specifica dei tuoi dati e dagli obiettivi.*")
    st.markdown("---")

    st.header("🔬 Scegli l'Algoritmo di Clustering")
    algoritmo_scelto = st.radio(
        "Quale algoritmo vuoi esplorare?",
        ("K-Means", "DBSCAN")
    )

    st.markdown("---")

    if algoritmo_scelto == "K-Means":
        st.subheader("Parametri K-Means")
        # Suggest K=2 if 'is_bodybuilder' is not used, as we know there are two main groups
        suggested_k = 2 if 'is_bodybuilder' not in features_for_clustering else len(features_for_clustering) + 1
        k_clusters = st.slider("Numero di Gruppi (K)", 2, 10, suggested_k, help="Il numero di cluster che K-Means cercherà di formare.")
        kmeans_random_state = st.slider("Seed per K-Means", 0, 100, 1, help="Controlla l'inizializzazione dei centroidi.")
        st.write("*(K-Means divide i dati in K cluster compatti)*")

    elif algoritmo_scelto == "DBSCAN":
        st.subheader("Parametri DBSCAN")
        eps = st.slider("Epsilon (eps)", 0.1, 3.0, 0.8, step=0.05, help="Raggio massimo per considerare i punti come 'vicini'.") # Adjusted range for new data
        min_samples_dbscan = st.slider("Min Samples", 2, 30, 5, help="Numero minimo di punti per formare un cluster denso.")
        st.write("*(DBSCAN raggruppa punti in base alla densità e identifica il rumore)*")

    st.markdown("---")
    st.caption("App sviluppata per MarketPro - Analisi Dati Clienti")

# --- Generazione Dati ---
# customer_df_full will contain ID_Cliente and all features generated by generate_custom_customer_data
customer_df_full = generate_custom_customer_data(n_samples, random_state_data)

# customer_data_for_clustering will contain only the features selected by the user
customer_data_for_clustering = customer_df_full[features_for_clustering].copy()


st.subheader("📊 Panoramica dei Dati Clienti Simulati")
st.write(f"Dataset generato con {n_samples} clienti. Di seguito un estratto con tutte le colonne generate:")
st.dataframe(customer_df_full.head())

st.write(f"Per il clustering verranno usate solo le **{len(features_for_clustering)} feature selezionate**: {', '.join(features_for_clustering)}")


# --- Controlli per la Visualizzazione del Grafico ---
st.markdown("---")
st.header("📈 Visualizzazione Cluster (Grafico 2D)")
if len(features_for_clustering) >= 2:
    st.write("Scegli due feature (tra quelle selezionate per il clustering) da visualizzare nel grafico a dispersione.")
    col_plot_x, col_plot_y = st.columns(2)
    
    # Default plot axes - try to pick meaningful ones if available
    default_x_plot = 'protein_spending' if 'protein_spending' in features_for_clustering else features_for_clustering[0]
    default_y_plot_options = [f for f in ['supplements_spending', 'avg_basket_size', 'carb_spending'] if f in features_for_clustering and f != default_x_plot]
    default_y_plot = default_y_plot_options[0] if default_y_plot_options else (features_for_clustering[1] if len(features_for_clustering) > 1 else features_for_clustering[0])


    plot_x_axis = col_plot_x.selectbox(
        "Feature per l'asse X:",
        options=features_for_clustering,
        index=features_for_clustering.index(default_x_plot)
    )
    plot_y_axis = col_plot_y.selectbox(
        "Feature per l'asse Y:",
        options=features_for_clustering,
        index=features_for_clustering.index(default_y_plot)
    )
    if plot_x_axis == plot_y_axis:
        st.warning("Per una visualizzazione significativa, scegli due feature diverse per l'asse X e Y.")
elif len(features_for_clustering) == 1:
    plot_x_axis = features_for_clustering[0]
    plot_y_axis = None
    st.info(f"Hai selezionato solo una feature ('{plot_x_axis}'). Verrà visualizzata come istogramma/densità se il clustering viene eseguito.")
else:
    st.error("Errore: nessuna feature selezionata per il clustering.") # Should be caught by earlier check
    st.stop()


# Scalatura dei dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data_for_clustering)
# Create a DataFrame from scaled data with correct column names for plotting
customer_df_scaled = pd.DataFrame(X_scaled, columns=features_for_clustering)


# --- Pulsante per Eseguire il Clustering ---
st.markdown("---")
if st.button(f"🚀 Esegui {algoritmo_scelto} Clustering!", type="primary"):
    if not plot_x_axis or (len(features_for_clustering) >=2 and not plot_y_axis):
        st.error("Seleziona le feature per gli assi X e Y del grafico prima di eseguire il clustering.")
    else:
        st.subheader(f"✨ Risultati del Clustering con {algoritmo_scelto}")

        labels = []
        cluster_centers_scaled = None
        cluster_centers_original = None
        n_clusters_found = 0
        inertia = None

        if algoritmo_scelto == "K-Means":
            kmeans = KMeans(n_clusters=k_clusters, random_state=kmeans_random_state, n_init='auto')
            labels = kmeans.fit_predict(X_scaled)
            cluster_centers_scaled = kmeans.cluster_centers_
            cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)
            n_clusters_found = len(set(labels))
            inertia = kmeans.inertia_

        elif algoritmo_scelto == "DBSCAN":
            dbscan = DBSCAN(eps=eps, min_samples=min_samples_dbscan)
            labels = dbscan.fit_predict(X_scaled)
            unique_labels = set(labels)
            n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)

        customer_df_full['Cluster'] = labels # Add cluster labels to the main df for analysis

        # --- Visualizzazione dei Cluster ---
        fig, ax = plt.subplots(figsize=(12, 8))

        if plot_y_axis is None and plot_x_axis is not None: # 1D plot
            unique_plot_labels = sorted(list(set(labels)))
            for i, label_val in enumerate(unique_plot_labels):
                data_to_plot = customer_df_scaled.loc[labels == label_val, plot_x_axis]
                if not data_to_plot.empty:
                    color_val = plt.cm.viridis(i / max(1, len(unique_plot_labels)-1)) if label_val != -1 else (0.5,0.5,0.5,0.6)
                    ax.hist(data_to_plot, bins=20, alpha=0.7, label=f'Cluster {label_val}' if label_val != -1 else 'Rumore (-1)', color=color_val, density=True)
            ax.set_title(f'Distribuzione di {plot_x_axis} (Scalata) per Cluster ({algoritmo_scelto})')
            ax.set_xlabel(f'{plot_x_axis} (Scalata)')
            ax.set_ylabel('Densità')
        elif plot_x_axis and plot_y_axis: # 2D Scatter Plot
            unique_plot_labels = sorted(list(set(labels)))
            num_actual_clusters = len(unique_plot_labels) -1 if -1 in unique_plot_labels else len(unique_plot_labels)
            
            # Create a color mapping
            cluster_colors = plt.cm.viridis(np.linspace(0, 1, max(1, num_actual_clusters)))
            color_map_dict = {}
            color_idx = 0
            for lbl in unique_plot_labels:
                if lbl == -1:
                    color_map_dict[lbl] = (0.5, 0.5, 0.5, 0.6) # Grey for noise
                else:
                    color_map_dict[lbl] = cluster_colors[color_idx]
                    color_idx +=1
            
            for label_val in unique_plot_labels:
                mask = (labels == label_val)
                current_color = color_map_dict[label_val]
                marker_style = 'x' if label_val == -1 else 'o'
                point_size = 60 if label_val == -1 else 100
                plot_label = f'Rumore/Outlier (-1)' if label_val == -1 else f'Cluster {label_val}'

                ax.scatter(customer_df_scaled.loc[mask, plot_x_axis],
                           customer_df_scaled.loc[mask, plot_y_axis],
                           c=[current_color], marker=marker_style, s=point_size, label=plot_label, alpha=0.8, edgecolors='w' if label_val !=-1 else 'none')

            if algoritmo_scelto == "K-Means" and cluster_centers_scaled is not None:
                idx_x = features_for_clustering.index(plot_x_axis)
                idx_y = features_for_clustering.index(plot_y_axis)
                ax.scatter(cluster_centers_scaled[:, idx_x],
                           cluster_centers_scaled[:, idx_y],
                           marker='X', s=350, c='red', label='Centroidi', edgecolors='black', zorder=10)

            ax.set_title(f'Cluster Clienti con {algoritmo_scelto} ({plot_x_axis} vs {plot_y_axis})')
            ax.set_xlabel(f'{plot_x_axis} (Scalata)')
            ax.set_ylabel(f'{plot_y_axis} (Scalata)')
        
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.markdown("---")
        col_metrics, col_details = st.columns(2)

        with col_metrics:
            st.subheader("Metriche e Riepilogo")
            if algoritmo_scelto == "K-Means":
                st.metric(label="Numero di Cluster Richiesti (K)", value=k_clusters)
                st.metric(label="Numero di Cluster Trovati Effettivi", value=n_clusters_found)
                if inertia is not None:
                    st.metric(label="Inerzia (WCSS)", value=f"{inertia:.2f}", help="Somma delle distanze quadrate dai centroidi: minore è, meglio è.")
            elif algoritmo_scelto == "DBSCAN":
                st.metric(label="Numero di Cluster Trovati", value=n_clusters_found)
                n_noise = list(labels).count(-1)
                st.metric(label="Punti Identificati come Rumore", value=n_noise)
                if n_clusters_found == 0 and n_noise > 0 and n_noise < len(labels):
                    st.warning("⚠️ Nessun cluster denso trovato. Prova a regolare `eps` (aumentare) o `min_samples` (diminuire).")
                elif n_clusters_found == 0 and n_noise == len(labels):
                     st.warning("⚠️ Tutti i punti sono rumore. Aumenta `eps` o diminuisci `min_samples`.")
                elif n_clusters_found == 0 and n_noise == 0:
                    st.warning("⚠️ Nessun cluster o rumore. Controlla dati/parametri.")

        with col_details:
            st.subheader(f"Analisi dei Cluster ({algoritmo_scelto})")
            
            # Create df_for_summary from customer_df_full which includes ID_Cliente and all original features + Cluster label
            df_for_summary = customer_df_full.copy()
            
            # For DBSCAN, often we only want to summarize actual clusters
            if algoritmo_scelto == "DBSCAN":
                if n_clusters_found > 0 :
                    df_for_summary_display = df_for_summary[df_for_summary['Cluster'] != -1]
                else: # No clusters found, maybe all noise
                    df_for_summary_display = pd.DataFrame() # Empty to show warning
            else: # K-Means
                df_for_summary_display = df_for_summary

            if not df_for_summary_display.empty and 'Cluster' in df_for_summary_display:
                # Show means of features used for clustering
                st.write("Profilo medio (per le feature usate nel clustering) di ogni cluster:")
                cluster_summary = df_for_summary_display.groupby('Cluster')[features_for_clustering].mean().round(2)
                st.dataframe(cluster_summary)

                # If 'is_bodybuilder' was NOT used for clustering, show its distribution in clusters
                if 'is_bodybuilder' in customer_df_full.columns and 'is_bodybuilder' not in features_for_clustering:
                    st.write("Distribuzione 'is_bodybuilder' (0=No, 1=Si) nei cluster trovati:")
                    # Ensure 'is_bodybuilder' is numeric for mean calculation
                    df_for_summary_display['is_bodybuilder'] = pd.to_numeric(df_for_summary_display['is_bodybuilder'], errors='coerce')
                    bodybuilder_dist = df_for_summary_display.groupby('Cluster')['is_bodybuilder'].agg(['mean', 'count', 'sum']).rename(
                        columns={'mean': '% Bodybuilder', 'count': 'N. Clienti nel Cluster', 'sum': 'N. Bodybuilder nel Cluster'}
                    )
                    bodybuilder_dist['% Bodybuilder'] = (bodybuilder_dist['% Bodybuilder']*100).round(1).astype(str) + '%'
                    st.dataframe(bodybuilder_dist)


                st.markdown("""
                **Interpretazione per MarketPro**:
                Analizza le medie dei cluster. Per il **cluster Bodybuilder/Fitness**, cerca un profilo con:
                * Alta `protein_spending` e `supplements_spending`.
                * Visite frequenti (`total_visits_per_month`).
                * Altre caratteristiche distintive basate sulle feature selezionate.
                """)
                if algoritmo_scelto == "K-Means" and cluster_centers_original is not None:
                    st.write("Centroidi dei Cluster (valori originali delle feature usate per il clustering):")
                    centroids_df = pd.DataFrame(cluster_centers_original, columns=features_for_clustering).round(2)
                    centroids_df.index.name = "Centroide K-Means"
                    st.dataframe(centroids_df)
            else:
                st.warning("Nessun cluster valido trovato per mostrare il riepilogo.")
else:
    st.markdown("<p style='text-align: center; font-style: italic;'>Clicca su 'Esegui Clustering!' per visualizzare i risultati.</p>", unsafe_allow_html=True)

# --- Sezioni Didattiche (invariate) ---
st.markdown("---")
st.header("📚 Approfondimenti sugli Algoritmi di Clustering")

with st.expander("🔍 K-Means: Quando i Gruppi sono 'Compatti' e il loro Numero è Conosciuto"):
    st.subheader("Cos'è il K-Means?")
    st.markdown("""
    Il **K-Means** è un algoritmo di clustering basato sui centroidi. Il suo obiettivo è partizionare `N` osservazioni in `K` cluster, dove ogni osservazione appartiene al cluster con il centroide (il centroide è la media dei punti nel cluster) più vicino.

    **Come funziona (iterativamente):**
    1.  **Inizializzazione**: Vengono scelti `K` centroidi iniziali casualmente o con tecniche più sofisticate (es. K-Means++).
    2.  **Assegnazione**: Ogni punto dati viene assegnato al centroide più vicino. Questo definisce i `K` cluster iniziali.
    3.  **Aggiornamento**: I centroidi vengono ricalcolati come la media (il "baricentro") di tutti i punti assegnati al loro rispettivo cluster.
    4.  **Iterazione**: I passi 2 e 3 vengono ripetuti finché i centroidi non si muovono più significativamente (convergenza) o viene raggiunto un numero massimo di iterazioni.

    **Punti di Forza:**
    * Semplice e veloce, specialmente su dataset di grandi dimensioni con un basso numero di K.
    * Facile da interpretare: i cluster hanno un centro ben definito.
    * Converge sempre.

    **Punti di Debolezza:**
    * Richiede di specificare il numero di cluster `K` in anticipo (può essere difficile da stimare).
    * Assume cluster di forma sferica/globulare e dimensioni simili.
    * Sensibile agli outlier (punti anomali) che possono distorcere i centroidi.
    * Può dare risultati diversi a seconda dell'inizializzazione dei centroidi (mitigato da più run con `n_init`).
    * Può faticare con cluster di densità molto diverse.
    """)

    st.subheader("Quando usare K-Means per MarketPro?")
    st.markdown("""
    K-Means può essere usato per MarketPro se:
    * **Hai un'ipotesi sul numero di segmenti di clienti** che vuoi identificare (`K`). Ad esempio, se sai che ci sono circa 2-3 tipi principali di clienti.
    * **Ti aspetti che i segmenti siano distinguibili e relativamente compatti** nelle feature selezionate.
    * **Vuoi centroidi chiari** per definire ogni segmento.

    **Sfida con questo dataset:** Il gruppo "bodybuilder" è definito da valori più alti in certe spese. K-Means (con K=2 e senza usare `is_bodybuilder` come feature) dovrebbe essere in grado di separare bene i due gruppi principali se le feature distintive sono selezionate.
    """)

with st.expander("🔬 DBSCAN: Scoprire Gruppi di Densità e Rilevare Anomali"):
    st.subheader("Cos'è DBSCAN?")
    st.markdown("""
    **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) è un algoritmo di clustering basato sulla **densità**. A differenza di K-Means, non richiede di specificare il numero di cluster in anticipo e può identificare cluster di forme arbitrarie, oltre a classificare i punti di rumore (outlier).

    **Come funziona (concetti chiave):**
    DBSCAN si basa su due parametri principali:
    * **`Epsilon (eps)` ($\epsilon$)**: Il raggio massimo del vicinato da considerare attorno a un punto.
    * **`min_samples` (MinPts)**: Il numero minimo di punti che devono trovarsi all'interno del raggio `eps` di un punto affinché quel punto sia considerato un **core point**.

    L'algoritmo classifica i punti come Core, Border, o Noise Point. I cluster vengono formati espandendosi dai core point ai punti densamente connessi.

    **Punti di Forza:**
    * Non richiede di specificare il numero di cluster.
    * Può trovare cluster di forma arbitraria.
    * Robusto agli outlier.

    **Punti di Debolezza:**
    * Sensibile alla scelta di `eps` e `min_samples`.
    * Non gestisce bene cluster di densità molto variabile.
    * La "maledizione della dimensionalità" può rendere difficile definire la densità.
    """)

    st.subheader("Quando usare DBSCAN per MarketPro (Individuazione Bodybuilder)?")
    st.markdown("""
    DBSCAN è utile se:
    * **Non sai quanti segmenti di clienti esistono**.
    * **Sospetti che il cluster dei bodybuilder sia un gruppo denso** ma la sua "forma" nello spazio delle feature potrebbe non essere sferica, o potrebbe esserci rumore.
    * **Vuoi identificare clienti anomali (outlier)**.

    **Sfida con questo dataset:** Trovare i giusti `eps` e `min_samples` sarà cruciale. Se i bodybuilder formano un gruppo denso e separato, DBSCAN dovrebbe identificarli. Potrebbe anche etichettare alcuni clienti "intermedi" come rumore. La standardizzazione dei dati (già fatta) è molto importante per DBSCAN.
    """)
