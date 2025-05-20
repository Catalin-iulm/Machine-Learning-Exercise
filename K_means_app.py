import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
# from matplotlib.colors import ListedColormap # Not strictly needed with current cmap usage

# --- (Placeholder) IMPORTA IL NUOVO GENERATORE ---
# Replace this with your actual generator logic from db_generator.py
# Ensure your generator can create 'ID_Cliente' and all features in ALL_POSSIBLE_FEATURES_FOR_CLUSTERING
def generate_advanced_customer_db(n_samples, random_state_data, selected_features_plus_id_eta):
    """
    Placeholder for your advanced customer database generator.
    It should generate 'ID_Cliente', 'Eta', and any other features
    that might be selected by the user from ALL_POSSIBLE_FEATURES_FOR_CLUSTERING.
    """
    np.random.seed(random_state_data)
    data = {'ID_Cliente': [f'CUST_{i:04d}' for i in range(n_samples)]}

    # Define a base set of features it can generate
    possible_cols_to_generate = [
        'Eta', 'Spesa_Proteine_Settimanale', 'Spesa_Carbo_Complessi_Settimanale',
        'Spesa_JunkFood_Settimanale', 'Frequenza_Reparto_SportSalute',
        'Varieta_Prodotti_Proteici', 'Spesa_Frutta_Verdura_Settimanale',
        'Spesa_Dolci_Settimanale', 'Visite_Supermercato_Mese',
        'Spesa_Media_Scontrino', 'Acquisto_Integratori', 'Uso_App_Fedelta'
    ]

    # Generate 'Eta'
    data['Eta'] = np.random.randint(18, 75, size=n_samples)

    # Generate other features with some plausible distributions
    # This is a very simplified generation logic.
    # Your actual generator will have more sophisticated logic for bodybuilder profiles etc.
    if 'Spesa_Proteine_Settimanale' in selected_features_plus_id_eta:
        data['Spesa_Proteine_Settimanale'] = np.random.gamma(2, 10, n_samples) + 5 # Skewed towards lower values, base 5
        # Simulate bodybuilders
        bodybuilder_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False) # 10% bodybuilders
        data['Spesa_Proteine_Settimanale'][bodybuilder_indices] = np.random.gamma(10, 8, len(bodybuilder_indices)) + 30

    if 'Spesa_JunkFood_Settimanale' in selected_features_plus_id_eta:
        data['Spesa_JunkFood_Settimanale'] = np.random.gamma(1.5, 8, n_samples)
        if 'Spesa_Proteine_Settimanale' in data and len(bodybuilder_indices) > 0 : # Reduce junk food for bodybuilders
             data['Spesa_JunkFood_Settimanale'][bodybuilder_indices] = np.random.gamma(1, 3, len(bodybuilder_indices))


    if 'Spesa_Carbo_Complessi_Settimanale' in selected_features_plus_id_eta:
        data['Spesa_Carbo_Complessi_Settimanale'] = np.random.gamma(2, 8, n_samples)
        if 'Spesa_Proteine_Settimanale' in data and len(bodybuilder_indices) > 0 :
            data['Spesa_Carbo_Complessi_Settimanale'][bodybuilder_indices] = np.random.gamma(5, 6, len(bodybuilder_indices)) + 10


    if 'Frequenza_Reparto_SportSalute' in selected_features_plus_id_eta:
        data['Frequenza_Reparto_SportSalute'] = np.random.randint(0, 10, n_samples)
        if 'Spesa_Proteine_Settimanale' in data and len(bodybuilder_indices) > 0 :
            data['Frequenza_Reparto_SportSalute'][bodybuilder_indices] = np.random.randint(5, 15, len(bodybuilder_indices))


    if 'Varieta_Prodotti_Proteici' in selected_features_plus_id_eta:
        data['Varieta_Prodotti_Proteici'] = np.random.randint(1, 8, n_samples)
        if 'Spesa_Proteine_Settimanale' in data and len(bodybuilder_indices) > 0 :
            data['Varieta_Prodotti_Proteici'][bodybuilder_indices] = np.random.randint(4, 12, len(bodybuilder_indices))


    # Generate other dummy features if they are in selected_features_plus_id_eta
    for feature in possible_cols_to_generate:
        if feature not in data and feature in selected_features_plus_id_eta: # Check if it was selected
            if "Spesa" in feature:
                data[feature] = np.random.uniform(5, 50, n_samples)
            elif "Frequenza" in feature or "Visite" in feature:
                data[feature] = np.random.randint(0, 20, n_samples)
            elif "Acquisto" in feature or "Uso" in feature: # Boolean-like
                data[feature] = np.random.randint(0, 2, n_samples)
            else:
                data[feature] = np.random.rand(n_samples) * 10


    df = pd.DataFrame(data)
    # Ensure all selected features are present, fill with 0 if any was missed by the simple generator logic above
    for f in selected_features_plus_id_eta:
        if f not in df.columns and f != 'ID_Cliente':
            df[f] = 0
    return df[selected_features_plus_id_eta] # Return only ID, Eta and selected features

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="MarketPro: Individuazione Clienti Target")

# --- Titolo e Introduzione ---
st.title("🎯 MarketPro: Individuazione Avanzata Cluster Clienti")
st.markdown("""
Benvenuti nell'analisi avanzata di **MarketPro**! Utilizzeremo i dati aggregati delle tessere fedeltà per identificare clienti con specifici pattern di acquisto.
L'obiettivo è segmentare la clientela per strategie di marketing mirate, con un focus sull'individuazione di nicchie di valore come **bodybuilder/appassionati di fitness**.
Esploreremo come **K-Means** e **DBSCAN** possono aiutarci, basandoci sulle feature da voi selezionate.
""")

st.info("💡 **Obiettivo Specifico**: Isolare gruppi di clienti (es. bodybuilder) analizzando le loro abitudini di acquisto e caratteristiche. Ad esempio, i bodybuilder potrebbero mostrare alta spesa in proteine, carboidrati complessi e bassa in junk food.")

# --- Lista di tutte le feature disponibili per il clustering (escluso ID) ---
ALL_POSSIBLE_FEATURES_FOR_CLUSTERING = [
    'Eta', 'Spesa_Proteine_Settimanale', 'Spesa_Carbo_Complessi_Settimanale',
    'Spesa_JunkFood_Settimanale', 'Frequenza_Reparto_SportSalute',
    'Varieta_Prodotti_Proteici', 'Spesa_Frutta_Verdura_Settimanale',
    'Spesa_Dolci_Settimanale', 'Visite_Supermercato_Mese',
    'Spesa_Media_Scontrino', 'Acquisto_Integratori', # Esempio: 0 o 1
    'Uso_App_Fedelta' # Esempio: 0 o 1
]
# Ordinare per chiarezza nella selectbox
ALL_POSSIBLE_FEATURES_FOR_CLUSTERING.sort()


# --- Sidebar per Controlli Globali ---
with st.sidebar:
    st.header("⚙️ Impostazioni Simulazione Dati")
    n_samples = st.slider("Numero di Clienti Simulati", 100, 2000, 700, help="Quanti profili cliente generare.")
    random_state_data = st.slider("Seed per Generazione Dati", 0, 100, 42, help="Controlla la riproducibilità dei dati generati.")

    st.header("📊 Selezione Feature per Clustering")
    st.markdown("L'età (`Eta`) è inclusa di default. Seleziona da 0 a 9 features aggiuntive (massimo 10 totali per il clustering).")

    # 'Eta' is always a candidate. User selects others.
    default_features = ['Eta', 'Spesa_Proteine_Settimanale', 'Spesa_JunkFood_Settimanale', 'Spesa_Carbo_Complessi_Settimanale']
    # Ensure defaults are in the main list
    default_selection = [f for f in default_features if f in ALL_POSSIBLE_FEATURES_FOR_CLUSTERING]


    selected_features = st.multiselect(
        "Scegli le feature per l'analisi di clustering:",
        options=ALL_POSSIBLE_FEATURES_FOR_CLUSTERING,
        default=default_selection,
        max_selections=10,
        help="Seleziona da 1 a 10 feature. 'ID_Cliente' è escluso dal clustering."
    )

    if not selected_features:
        st.warning("Per favore, seleziona almeno una feature per il clustering.")
        st.stop()
    elif 'Eta' not in selected_features:
         st.warning("Si consiglia di includere 'Eta' tra le feature selezionate per un'analisi più completa.")


    features_for_clustering = selected_features # This list will be used

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

    # Controlli dinamici per l'algoritmo scelto
    if algoritmo_scelto == "K-Means":
        st.subheader("Parametri K-Means")
        k_clusters = st.slider("Numero di Gruppi (K)", 2, 10, 5, help="Il numero di cluster che K-Means cercherà di formare.")
        kmeans_random_state = st.slider("Seed per K-Means", 0, 100, 1, help="Controlla l'inizializzazione dei centroidi. Un valore fisso garantisce riproducibilità.")
        st.write("*(K-Means divide i dati in K cluster compatti)*")

    elif algoritmo_scelto == "DBSCAN":
        st.subheader("Parametri DBSCAN")
        eps = st.slider("Epsilon (eps)", 0.1, 2.5, 0.5, step=0.05, help="Raggio massimo per considerare i punti come 'vicini'. Adattalo in base alla scala e densità dei dati.")
        min_samples_dbscan = st.slider("Min Samples", 2, 30, 5, help="Numero minimo di punti in un neighborhood per formare un cluster denso.") # Renamed to avoid conflict
        st.write("*(DBSCAN raggruppa punti in base alla densità e identifica il rumore)*")

    st.markdown("---")
    st.caption("App sviluppata per MarketPro - Analisi Dati Clienti")

# --- Generazione Dati ---
# Features to generate: ID, Eta, and all selected for clustering
features_to_generate_in_df = ['ID_Cliente'] + features_for_clustering
# Remove duplicates if 'Eta' was already in features_for_clustering
features_to_generate_in_df = sorted(list(set(features_to_generate_in_df)))


customer_df_full = generate_advanced_customer_db(n_samples, random_state_data, features_to_generate_in_df)
# Ensure 'Eta' is numeric if it's not already
if 'Eta' in customer_df_full.columns:
    customer_df_full['Eta'] = pd.to_numeric(customer_df_full['Eta'], errors='coerce').fillna(0)


# Prepare data for clustering (only selected features, no ID)
customer_data_for_clustering = customer_df_full[features_for_clustering].copy()

st.subheader("📊 Panoramica dei Dati Clienti Simulati (Feature Selezionate)")
st.write(f"Ecco un estratto del dataset con le {len(features_for_clustering)} feature selezionate per il clustering (più ID_Cliente):")
st.dataframe(customer_df_full[['ID_Cliente'] + features_for_clustering].head())


# --- Controlli per la Visualizzazione del Grafico ---
st.markdown("---")
st.header("📈 Visualizzazione Cluster (Grafico 2D)")
if len(features_for_clustering) >= 2:
    st.write("Scegli due feature (tra quelle selezionate per il clustering) da visualizzare nel grafico a dispersione.")
    col_plot_x, col_plot_y = st.columns(2)
    plot_x_axis = col_plot_x.selectbox(
        "Feature per l'asse X:",
        options=features_for_clustering,
        index=0
    )
    plot_y_axis = col_plot_y.selectbox(
        "Feature per l'asse Y:",
        options=features_for_clustering,
        index=1 if len(features_for_clustering) > 1 else 0 # Ensure different default if possible
    )
    if plot_x_axis == plot_y_axis:
        st.warning("Per una visualizzazione significativa, scegli due feature diverse per l'asse X e Y.")
elif len(features_for_clustering) == 1:
    plot_x_axis = features_for_clustering[0]
    plot_y_axis = None # Only 1D, can't do 2D scatter
    st.info(f"Hai selezionato solo una feature ('{plot_x_axis}'). Verrà visualizzata come istogramma se il clustering viene eseguito.")
else: # Should not happen due to guard above
    st.error("Errore: nessuna feature selezionata per il clustering.")
    st.stop()


# Scalatura dei dati (solo le feature per il clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data_for_clustering)
customer_df_scaled = pd.DataFrame(X_scaled, columns=features_for_clustering)


# --- Pulsante per Eseguire il Clustering ---
st.markdown("---")
if st.button(f"🚀 Esegui {algoritmo_scelto} Clustering!", type="primary"):
    if not plot_x_axis or (len(features_for_clustering) >=2 and not plot_y_axis):
        st.error("Seleziona le feature per gli assi X e Y del grafico prima di eseguire il clustering.")
    else:
        st.subheader(f"✨ Risultati del Clustering con {algoritmo_scelto}")

        labels = []
        cluster_centers_scaled = None # Centroids in scaled space
        cluster_centers_original = None # Centroids in original scale
        n_clusters_found = 0
        inertia = None

        if algoritmo_scelto == "K-Means":
            kmeans = KMeans(n_clusters=k_clusters, random_state=kmeans_random_state, n_init='auto') # n_init='auto' is new default
            labels = kmeans.fit_predict(X_scaled)
            cluster_centers_scaled = kmeans.cluster_centers_ # Centroidi nello spazio scalato
            cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled) # Riporta i centroidi alla scala originale
            n_clusters_found = len(set(labels))
            inertia = kmeans.inertia_

        elif algoritmo_scelto == "DBSCAN":
            dbscan = DBSCAN(eps=eps, min_samples=min_samples_dbscan)
            labels = dbscan.fit_predict(X_scaled)
            unique_labels = set(labels)
            n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
            # DBSCAN doesn't have explicit centroids in the same way K-Means does.
            # We can calculate them as the mean of points in each cluster if needed for interpretation.

        # Aggiungi le label al DataFrame originale per l'interpretazione
        # Use a temporary df to avoid modifying customer_df_full if button is pressed multiple times
        customer_df_with_clusters = customer_df_full.copy()
        customer_df_with_clusters['Cluster'] = labels

        # --- Visualizzazione dei Cluster ---
        fig, ax = plt.subplots(figsize=(12, 8))

        # Handle case with only one feature (histogram)
        if plot_y_axis is None and plot_x_axis is not None:
            if len(set(labels)) > 0:
                for label_val in sorted(set(labels)):
                    data_to_plot = customer_df_scaled.loc[labels == label_val, plot_x_axis]
                    if not data_to_plot.empty:
                        ax.hist(data_to_plot, bins=15, alpha=0.7, label=f'Cluster {label_val}' if label_val != -1 else 'Rumore (-1)')
                ax.set_title(f'Distribuzione di {plot_x_axis} (Scalata) per Cluster ({algoritmo_scelto})')
                ax.set_xlabel(f'{plot_x_axis} (Scalata)')
                ax.set_ylabel('Frequenza')
            else:
                ax.text(0.5, 0.5, "Nessun dato da plottare.", horizontalalignment='center', verticalalignment='center')

        # Standard 2D Scatter Plot
        elif plot_x_axis and plot_y_axis:
            # Crea una colormap dinamica
            n_unique_labels = len(set(labels))
            # Escludi il colore per il rumore se presente
            if -1 in labels and n_unique_labels > 1 : # check if there are actual clusters besides noise
                cmap_base = plt.cm.get_cmap('viridis', n_unique_labels -1 if n_unique_labels > 1 else 1)
                colors = [cmap_base(i) for i in range(n_unique_labels -1 if n_unique_labels > 1 else 1)]
                colors.append((0.5, 0.5, 0.5, 0.6)) # Grigio per rumore
            elif n_unique_labels > 0 :
                cmap_base = plt.cm.get_cmap('viridis', n_unique_labels if n_unique_labels > 0 else 1)
                colors = [cmap_base(i) for i in range(n_unique_labels if n_unique_labels > 0 else 1)]
            else: # No labels or only noise
                colors = [(0.5, 0.5, 0.5, 0.6)] # Default to grey

            color_map_dict = {label: colors[i % len(colors)] for i, label in enumerate(sorted(set(labels)))}


            for i, label in enumerate(sorted(set(labels))):
                mask = (labels == label)
                current_color = color_map_dict[label]

                if label == -1: # Rumore
                    ax.scatter(customer_df_scaled.loc[mask, plot_x_axis],
                               customer_df_scaled.loc[mask, plot_y_axis],
                               c=[current_color], marker='x', s=60, label=f'Rumore/Outlier (-1)', alpha=0.7, edgecolors='none')
                else:
                    ax.scatter(customer_df_scaled.loc[mask, plot_x_axis],
                               customer_df_scaled.loc[mask, plot_y_axis],
                               c=[current_color], marker='o', s=100, label=f'Cluster {label}', alpha=0.8, edgecolors='w')

            if algoritmo_scelto == "K-Means" and cluster_centers_scaled is not None:
                # Plot centroids only if K-Means was used and they exist
                # Ensure plot_x_axis and plot_y_axis are in features_for_clustering
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
                st.metric(label="Numero di Cluster Trovati Effettivi", value=n_clusters_found) # Could be less if some clusters are empty
                if inertia is not None:
                    st.metric(label="Inerzia (WCSS)", value=f"{inertia:.2f}",
                              help="Misura la compattezza dei cluster (somma delle distanze quadrate dai centroidi): minore è, meglio è.")
            elif algoritmo_scelto == "DBSCAN":
                st.metric(label="Numero di Cluster Trovati", value=n_clusters_found)
                n_noise = list(labels).count(-1)
                st.metric(label="Punti Identificati come Rumore (Outlier)", value=n_noise)
                if n_clusters_found == 0 and n_noise > 0 and n_noise < len(labels):
                    st.warning("⚠️ Nessun cluster denso trovato, solo rumore e forse alcuni punti non classificati. Prova a regolare `eps` (aumentare) o `min_samples` (diminuire).")
                elif n_clusters_found == 0 and n_noise == len(labels):
                     st.warning("⚠️ Tutti i punti sono stati classificati come rumore. Prova ad aumentare `eps` significativamente o diminuire `min_samples`.")
                elif n_clusters_found == 0 and n_noise == 0: # Should not happen with DBSCAN unless X_scaled is empty
                    st.warning("⚠️ Nessun cluster o rumore trovato. Controlla i dati di input o i parametri.")


        with col_details:
            st.subheader(f"Analisi dei Cluster ({algoritmo_scelto})")
            st.write("Profilo medio per ogni cluster identificato (valori originali, escluso rumore per DBSCAN):")

            # Use the df with cluster assignments, use original feature names for groupby
            df_for_summary = customer_df_with_clusters.copy()

            if algoritmo_scelto == "DBSCAN":
                # Exclude noise points from summary, unless no clusters were found
                if n_clusters_found > 0:
                    df_for_summary = df_for_summary[df_for_summary['Cluster'] != -1]
                elif n_clusters_found == 0 and list(labels).count(-1) < len(labels): # No dense clusters, but not all noise
                     st.write("Nessun cluster denso identificato per il riepilogo. Mostrando medie dei punti non rumorosi (se presenti).")
                     df_for_summary = df_for_summary[df_for_summary['Cluster'] != -1] # Show non-noise if any exist

            if not df_for_summary.empty and 'Cluster' in df_for_summary and df_for_summary['Cluster'].nunique() > 0 :
                # Ensure we group by features used for clustering
                cluster_summary = df_for_summary.groupby('Cluster')[features_for_clustering].mean().round(2)
                st.dataframe(cluster_summary)

                st.markdown("""
                **Interpretazione per MarketPro**:
                Analizza le medie dei cluster per identificare segmenti di clientela.
                Per il **cluster Bodybuilder/Fitness**, cerca un profilo con:
                * Alta `Spesa_Proteine_Settimanale` (se selezionata)
                * Alta `Spesa_Carbo_Complessi_Settimanale` (se selezionata)
                * Bassa `Spesa_JunkFood_Settimanale` (se selezionata)
                * Alta `Frequenza_Reparto_SportSalute` (se selezionata)
                * Alta `Varieta_Prodotti_Proteici` (se selezionata)
                * Età (`Eta`) potrebbe variare, ma spesso più giovani adulti.
                * Altre feature rilevanti che hai selezionato.
                """)
                if algoritmo_scelto == "K-Means" and cluster_centers_original is not None:
                    st.write("Centroidi dei Cluster (valori originali):")
                    centroids_df = pd.DataFrame(cluster_centers_original, columns=features_for_clustering).round(2)
                    centroids_df.index.name = "Centroide del Cluster"
                    st.dataframe(centroids_df)


            elif list(labels).count(-1) == len(labels) and algoritmo_scelto == "DBSCAN":
                 st.warning("Tutti i punti sono stati classificati come rumore. Nessun riepilogo di cluster da mostrare.")
            else:
                st.warning("Nessun cluster valido trovato per mostrare il riepilogo (potrebbe essere tutto rumore o nessun cluster formato).")
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
    * **Hai un'ipotesi sul numero di segmenti di clienti** che vuoi identificare (`K`).
    * **Ti aspetti che i segmenti siano distinguibili e relativamente compatti** nelle feature selezionate (es. i bodybuilder formano un gruppo con valori simili per spesa proteica, junk food, etc.).
    * **La velocità è importante** per dataset molto grandi.
    * **Vuoi centroidi chiari** per definire ogni segmento.

    **Sfida in questo scenario:** Se il cluster "bodybuilder" è una nicchia piccola e non perfettamente sferica rispetto alle altre, o se il `K` scelto non è ottimale, K-Means potrebbe non isolarlo nettamente o fonderlo con altri gruppi. È anche sensibile alla scala delle features, quindi la standardizzazione (già implementata) è cruciale.
    """)

with st.expander("🔬 DBSCAN: Scoprire Gruppi di Densità e Rilevare Anomali"):
    st.subheader("Cos'è DBSCAN?")
    st.markdown("""
    **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) è un algoritmo di clustering basato sulla **densità**. A differenza di K-Means, non richiede di specificare il numero di cluster in anticipo e può identificare cluster di forme arbitrarie, oltre a classificare i punti di rumore (outlier).

    **Come funziona (concetti chiave):**
    DBSCAN si basa su due parametri principali:
    * **`Epsilon (eps)` ($\epsilon$)**: Il raggio massimo del vicinato da considerare attorno a un punto. Due punti sono vicini se la loro distanza è minore o uguale a $\epsilon$.
    * **`min_samples` (MinPts)**: Il numero minimo di punti che devono trovarsi all'interno del raggio `eps` di un punto (incluso il punto stesso) affinché quel punto sia considerato un **core point**.

    L'algoritmo classifica i punti come:
    1.  **Core Point**: Un punto con almeno `min_samples` punti nel suo $\epsilon$-vicinato. I core point sono il cuore dei cluster.
    2.  **Border Point**: Un punto che non è un core point, ma si trova nell' $\epsilon$-vicinato di un core point. I border point appartengono a un cluster ma sono ai suoi margini.
    3.  **Noise Point (Outlier)**: Un punto che non è né un core point né un border point. Questi punti non appartengono a nessun cluster denso.

    I cluster vengono formati espandendosi dai core point ai punti densamente connessi.

    **Punti di Forza:**
    * Non richiede di specificare il numero di cluster.
    * Può trovare cluster di forma arbitraria (non solo sferici).
    * Robusto agli outlier, che vengono identificati come rumore.
    * I parametri `eps` e `min_samples` hanno un significato fisico comprensibile.

    **Punti di Debolezza:**
    * Può essere sensibile alla scelta di `eps` e `min_samples`. Trovare i valori ottimali può richiedere sperimentazione.
    * Non gestisce bene cluster di densità molto variabile (un `eps` e `min_samples` globale potrebbe non funzionare per tutti).
    * Può essere computazionalmente più intensivo di K-Means su dataset molto grandi, specialmente con implementazioni non ottimizzate o `eps` grandi.
    * La "maledizione della dimensionalità" può rendere difficile definire la densità in spazi ad alta dimensionalità.
    """)

    st.subheader("Quando usare DBSCAN per MarketPro (Individuazione Bodybuilder)?")
    st.markdown("""
    DBSCAN è particolarmente promettente per MarketPro, specialmente per individuare nicchie come i bodybuilder, se:
    * **Non sai quanti segmenti di clienti esistono** o se il loro numero varia.
    * **Sospetti che il cluster dei bodybuilder sia un gruppo denso** con abitudini di acquisto specifiche, ma la sua "forma" nello spazio delle feature potrebbe non essere sferica.
    * **Vuoi identificare clienti anomali (outlier)** che non rientrano in nessun segmento definito.
    * **Il cluster dei bodybuilder potrebbe essere di dimensioni diverse** rispetto ad altri segmenti di clientela.

    **Sfida in questo scenario:** La scelta di `eps` e `min_samples` è cruciale.
    * Un `eps` troppo piccolo potrebbe frammentare i cluster o classificare troppi punti come rumore.
    * Un `eps` troppo grande potrebbe unire cluster distinti.
    * `min_samples` influenza la "densità minima" richiesta. Per nicchie piccole ma dense, un `min_samples` più basso potrebbe essere necessario, ma attenzione a non creare cluster da piccole fluttuazioni casuali.
    La standardizzazione dei dati (già fatta) è importante perché `eps` è una misura di distanza.
    """)
