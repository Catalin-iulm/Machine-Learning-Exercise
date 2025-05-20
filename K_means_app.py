import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="Visualizzatore Algoritmi di Clustering per Marketing Supermercato")

# --- Titolo e Introduzione ---
st.title("🔬 Visualizzatore Interattivo di Algoritmi di Clustering per il Marketing al Supermercato")
st.markdown("""
Questa applicazione ti permette di esplorare il funzionamento degli algoritmi di clustering **K-Means** e **DBSCAN**
su un dataset simulato di clienti di un supermercato, con caratteristiche realistiche (Età, Sesso, Reddito, Comportamento Online).
Modifica i parametri dell'algoritmo per vedere quali segmenti di clienti puoi identificare e come ottimizzare le tue strategie di marketing!
""")

# --- Funzione per Generare Dati Sintetici per Marketing (Supermercato) ---
def generate_customer_data(n_samples, random_state):
    np.random.seed(random_state)

    ages = np.random.randint(18, 71, n_samples)
    genders = np.random.choice(['Uomo', 'Donna'], n_samples)

    # Simulate income based on age
    incomes = np.zeros(n_samples)
    for i, age in enumerate(ages):
        if 18 <= age <= 25: # Young adults
            incomes[i] = np.random.normal(20000, 4000) # Lower income
        elif 26 <= age <= 40: # Early/Mid career
            incomes[i] = np.random.normal(40000, 8000) # Medium income
        elif 41 <= age <= 55: # Established career
            incomes[i] = np.random.normal(60000, 12000) # Higher income
        else: # 56-70 (Approaching/in retirement)
            incomes[i] = np.random.normal(45000, 10000) # Income might stabilize or slightly decrease

    # Simulate website/app visits based on age
    visits = np.zeros(n_samples)
    for i, age in enumerate(ages):
        if 18 <= age <= 35: # Younger, more tech-savvy
            visits[i] = np.random.normal(12, 5) # High frequency
        elif 36 <= age <= 55: # Mid-age, moderate tech use
            visits[i] = np.random.normal(7, 3) # Medium frequency
        else: # 56-70 (Older, less frequent online)
            visits[i] = np.random.normal(3, 2) # Low frequency

    # Simulate average time on site/app based on age
    time_on_site = np.zeros(n_samples)
    for i, age in enumerate(ages):
        if 18 <= age <= 35:
            time_on_site[i] = np.random.normal(25, 8) # Longer time
        elif 36 <= age <= 55:
            time_on_site[i] = np.random.normal(15, 6) # Medium time
        else:
            time_on_site[i] = np.random.normal(7, 3) # Shorter time

    # Ensure positive values and reasonable limits for all features
    incomes = np.clip(incomes, 10000, 150000).astype(int)
    visits = np.clip(visits, 1, 30).astype(int)
    time_on_site = np.clip(time_on_site, 2, 60).astype(int) # Time in minutes

    # Create DataFrame for unscaled data
    df_unscaled = pd.DataFrame({
        'Età': ages,
        'Sesso': genders,
        'Reddito Medio Annuo (€)': incomes,
        'Frequenza Visite al Sito/App (n/mese)': visits,
        'Tempo Medio Permanenza sul Sito/App (min)': time_on_site
    })
    
    # Select features for clustering and scale them
    # Sesso cannot be directly used in numerical clustering without one-hot encoding,
    # and for 2D visualization it's best to stick to numerical features.
    X_unscaled_numerical = df_unscaled[['Età', 'Reddito Medio Annuo (€)', 'Frequenza Visite al Sito/App (n/mese)', 'Tempo Medio Permanenza sul Sito/App (min)']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled_numerical)
    
    return X_scaled, df_unscaled, X_unscaled_numerical # Return numerical unscaled data too

# --- Mapping per Nomi Features Marketing ---
# These are the actual column names in the df_unscaled numerical part
feature_names_mapping = {
    "Età": "Età",
    "Reddito Medio Annuo (€)": "Reddito Medio Annuo (€)",
    "Frequenza Visite al Sito/App (n/mese)": "Frequenza Visite al Sito/App (n/mese)",
    "Tempo Medio Permanenza sul Sito/App (min)": "Tempo Medio Permanenza sul Sito/App (min)"
}

# --- Sidebar per Controlli Globali ---
with st.sidebar:
    st.header("⚙️ Configurazione Esperimento")

    st.subheader("1. Generazione Dataset 'Clienti'")
    n_samples_data = st.slider("Numero di 'Clienti' Simulati", 500, 3000, 1500, step=100) # Increased max samples
    random_state_ds = st.slider("Seed Generazione Dati (per riproducibilità)", 0, 100, 42)
    st.markdown("---")

    st.subheader("2. Scegli le Caratteristiche per il Grafico")
    plot_feature_x = st.selectbox("Caratteristica Asse X:", list(feature_names_mapping.keys()), index=0) # Default to Età
    plot_feature_y = st.selectbox("Caratteristica Asse Y:", list(feature_names_mapping.keys()), index=2) # Default to Frequenza Visite

    if plot_feature_x == plot_feature_y:
        st.warning("Seleziona caratteristiche diverse per l'asse X e Y per una visualizzazione significativa.")
        st.stop() # Stop execution if axes are the same

    st.markdown("---")

    st.subheader("3. Scegli l'Algoritmo di Clustering")
    algoritmo_scelto = st.radio(
        "Algoritmo:", ("K-Means", "DBSCAN"), horizontal=True
    )
    st.markdown("---")

    st.subheader(f"4. Parametri {algoritmo_scelto}")
    if algoritmo_scelto == "K-Means":
        k_clusters_param = st.slider("Numero di Segmenti (K) da Trovare", 1, 10, 4, # Changed default K to 4 for better fit with new data
                                     help="Quanti segmenti di clienti l'algoritmo K-Means cercherà.")
        kmeans_random_state_param = st.slider("Seed K-Means (per inizializzazione)", 0, 100, 1,
                                             help="Controlla l'inizializzazione dei centroidi per la riproducibilità. Cambialo per vedere diverse configurazioni iniziali.")
    elif algoritmo_scelto == "DBSCAN":
        eps_param = st.slider("Epsilon (eps) - Raggio di Vicinato", 0.05, 2.0, 0.5, step=0.01,
                             help="Distanza massima per considerare due 'clienti' vicini. Dato che i dati sono scalati, 0.5-1.0 è un buon punto di partenza.")
        min_samples_param = st.slider("Min Samples - Densità Minima", 1, 50, 15, # Adjusted min_samples for more structured data
                                     help="Numero minimo di 'clienti' in un vicinato per formare un segmento denso. I punti sotto questa soglia potrebbero essere considerati rumore.")

# --- Generazione Dati ---
X_data_scaled_full, df_data_unscaled, X_data_unscaled_numerical = generate_customer_data(n_samples_data, random_state_ds)

# Extract indices for plotting features from the numerical array
idx_x = list(feature_names_mapping.keys()).index(plot_feature_x)
idx_y = list(feature_names_mapping.keys()).index(plot_feature_y)

# Get the specific scaled features for plotting
X_data_scaled_plot = X_data_scaled_full[:, [idx_x, idx_y]]


# --- Contesto Marketing per il Dataset ---
st.markdown("---")
st.subheader("💡 Contesto Marketing del Dataset 'Clienti Simulati del Supermercato':")
st.info("""
    Questo dataset simula una base clienti di un supermercato, con caratteristiche come **Età**, **Sesso**, **Reddito Medio Annuo**,
    **Frequenza Visite al Sito/App** e **Tempo Medio di Permanenza sul Sito/App**.
    I dati sono stati generati per riflettere correlazioni realistiche (es. i più giovani tendono a essere più online),
    creando implicitamente dei segmenti di clienti, come:
    * **Giovani Clienti Digitali:** Età più bassa, reddito variabile (spesso inferiore), alta attività online.
    * **Famiglie/Professionisti:** Età media, reddito medio/alto, attività online bilanciata.
    * **Clienti Senior/Tradizionali:** Età più avanzata, reddito da pensione, minore propensione all'online.

    L'obiettivo del clustering è **identificare questi segmenti nascosti** per sviluppare strategie di marketing più efficaci:
    promozioni mirate, personalizzazione dell'esperienza in-store vs. online, gestione della fedeltà e acquisizione di nuovi clienti.
""")
st.markdown("---")

# --- Visualizzazione del Dataset Completo ---
with st.expander("📊 Visualizza il Dataset Completo dei 'Clienti' (Dati Originali)"):
    st.write(f"Ecco un'anteprima dei primi 10 'clienti' del dataset simulato con le loro caratteristiche originali:")
    st.dataframe(df_data_unscaled.head(10))
    st.write(f"**Numero totale di clienti simulati:** {len(df_data_unscaled)}")
    
    # Add download button for the CSV
    csv_buffer = io.StringIO()
    df_data_unscaled.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Scarica il Dataset Completo (CSV)",
        data=csv_buffer.getvalue(),
        file_name="clienti_supermercato_simulati.csv",
        mime="text/csv"
    )
    st.info("Nota: Gli algoritmi di clustering lavorano sulle **caratteristiche numeriche scalate** (Età, Reddito, Frequenza, Tempo) per garantire che tutte abbiano la stessa importanza, indipendentemente dalla loro scala originale. La colonna 'Sesso' è inclusa solo per contesto e non viene usata direttamente nel clustering bidimensionale, ma sarebbe rilevante in un'analisi multidimensionale.")

st.header(f"🚀 Esecuzione e Risultati: {algoritmo_scelto}")

# --- Esecuzione Clustering ---
labels_pred = []
cluster_centers_coords_plot = None
n_clusters_found_val = 0
inertia_val = None
n_noise_points = 0
silhouette_avg = None

if algoritmo_scelto == "K-Means":
    kmeans_model = KMeans(n_clusters=k_clusters_param, random_state=kmeans_random_state_param, n_init='auto')
    labels_pred = kmeans_model.fit_predict(X_data_scaled_full) # Use full scaled data for clustering
    
    # Centroids are in the full feature space, need to project to the 2D plot space
    cluster_centers_coords_full = kmeans_model.cluster_centers_
    cluster_centers_coords_plot = cluster_centers_coords_full[:, [idx_x, idx_y]] # Extract for plotting
    
    n_clusters_found_val = len(set(labels_pred))
    inertia_val = kmeans_model.inertia_
elif algoritmo_scelto == "DBSCAN":
    dbscan_model = DBSCAN(eps=eps_param, min_samples=min_samples_param)
    labels_pred = dbscan_model.fit_predict(X_data_scaled_full) # Use full scaled data for clustering
    unique_labels_set = set(labels_pred)
    n_clusters_found_val = len(unique_labels_set) - (1 if -1 in unique_labels_set else 0)
    n_noise_points = list(labels_pred).count(-1)

# Calcola Silhouette Score se ci sono cluster validi (più di 1 cluster e non tutti rumore)
if len(set(labels_pred)) > 1 and (len(set(labels_pred)) > 1 or (algoritmo_scelto == "DBSCAN" and -1 not in set(labels_pred))):
    try:
        silhouette_avg = silhouette_score(X_data_scaled_full, labels_pred) # Use full scaled data for silhouette calculation
    except ValueError: # Happens if only one cluster is found, or all points are noise
        silhouette_avg = None
else:
    silhouette_avg = None # Cannot compute for 1 cluster or all noise

# --- Visualizzazione dei Cluster ---
col1_plot, col2_metrics = st.columns([2,1])

with col1_plot:
    st.subheader(f"Grafico dei Segmenti di Clienti: '{plot_feature_x}' vs '{plot_feature_y}' (Dati Scalati)")
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 7))

    # Colormap dinamica
    unique_plot_labels = sorted(list(set(labels_pred)))
    num_actual_clusters_for_cmap = len(unique_plot_labels) -1 if -1 in unique_plot_labels else len(unique_plot_labels)
    
    cluster_colors_palette = plt.cm.viridis(np.linspace(0, 1, max(1, num_actual_clusters_for_cmap)))
    color_map_for_plot = {}
    color_idx_plot = 0
    for lbl_plot in unique_plot_labels:
        if lbl_plot == -1:
            color_map_for_plot[lbl_plot] = (0.5, 0.5, 0.5, 0.7) # Grigio per rumore (outlier)
        else:
            if color_idx_plot < len(cluster_colors_palette):
                color_map_for_plot[lbl_plot] = cluster_colors_palette[color_idx_plot]
            else:
                color_map_for_plot[lbl_plot] = (np.random.rand(), np.random.rand(), np.random.rand(), 0.8)
            color_idx_plot +=1
    
    for label_val_plot in unique_plot_labels:
        mask_plot = (labels_pred == label_val_plot)
        current_color_plot = color_map_for_plot.get(label_val_plot, (0,0,0,1))
        marker_style_plot = 'x' if label_val_plot == -1 else 'o'
        point_size_plot = 40 if label_val_plot == -1 else 60
        plot_legend_label = f'Clienti Rumore/Outlier (-1)' if label_val_plot == -1 else f'Segmento {label_val_plot}'

        ax_cluster.scatter(X_data_scaled_plot[mask_plot, 0], X_data_scaled_plot[mask_plot, 1],
                           facecolor=current_color_plot, marker=marker_style_plot, s=point_size_plot,
                           label=plot_legend_label, alpha=0.8,
                           edgecolor='k' if label_val_plot !=-1 else 'none', linewidth=0.5 if label_val_plot !=-1 else 0)

    if algoritmo_scelto == "K-Means" and cluster_centers_coords_plot is not None:
        ax_cluster.scatter(cluster_centers_coords_plot[:, 0], cluster_centers_coords_plot[:, 1],
                           marker='P', s=250, facecolor='red', label='Centroide Segmento',
                           edgecolor='black', linewidth=1.5, zorder=10)

    # Dynamic Axis Labels for scaled plot
    ax_cluster.set_title(f'Segmentazione con {algoritmo_scelto}')
    ax_cluster.set_xlabel(f"{plot_feature_x} (Scalata)")
    ax_cluster.set_ylabel(f"{plot_feature_y} (Scalata)")
    ax_cluster.legend(loc='best', fontsize='small')
    ax_cluster.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_cluster)

with col2_metrics:
    st.subheader("Metriche dei Segmenti")
    if algoritmo_scelto == "K-Means":
        st.metric(label="Numero di Segmenti (K) Richiesti", value=k_clusters_param)
        st.metric(label="Numero di Segmenti Trovati", value=n_clusters_found_val)
        if inertia_val is not None:
            st.metric(label="Inerzia (WCSS)", value=f"{inertia_val:.2f}",
                      help="Somma delle distanze quadrate dai centroidi: minore è, più compatti sono i segmenti.")
    elif algoritmo_scelto == "DBSCAN":
        st.metric(label="Numero di Segmenti Trovati", value=n_clusters_found_val)
        st.metric(label="Clienti Rumorosi (Outliers)", value=n_noise_points)

    if silhouette_avg is not None:
        st.metric(label="Silhouette Score", value=f"{silhouette_avg:.3f}",
                  help="Misura quanto bene i clienti sono raggruppati all'interno del proprio segmento e separati dagli altri (-1 a +1). Più alto è, meglio definiti sono i segmenti.")
    else:
        st.info("Silhouette Score non calcolabile (es. un solo segmento trovato o tutti i clienti sono rumore).")

    st.markdown("---")
    st.write("**Conteggio Clienti per Segmento:**")
    if len(labels_pred) > 0:
        counts = pd.Series(labels_pred).value_counts().sort_index()
        counts.index = counts.index.map(lambda x: 'Clienti Rumore (-1)' if x == -1 else f'Segmento {x}')
        st.dataframe(counts.rename("Numero di Clienti"))
    else:
        st.write("Nessun cliente clusterizzato.")


# --- Sezioni Didattiche ---
st.markdown("---")
st.header("📚 Approfondimenti sugli Algoritmi di Clustering per il Marketing")

with st.expander("🔍 K-Means: Come Segmenta i Clienti?"):
    st.markdown("""
    **K-Means** mira a partizionare i tuoi clienti in `K` segmenti distinti, assegnando ogni cliente al segmento con il "centroide" (il cliente medio di quel segmento) più simile.

    **Pensalo così:**
    1.  **Inizializzazione**: Scegli `K` punti iniziali che rappresentano potenziali "clienti tipo".
    2.  **Assegnazione**: Ogni cliente viene assegnato al "cliente tipo" più vicino, formando così `K` segmenti.
    3.  **Aggiornamento**: Il "cliente tipo" di ogni segmento viene ricalcolato come la media di tutti i clienti al suo interno.
    4.  **Iterazione**: I passaggi 2 e 3 vengono ripetuti finché i segmenti non si stabilizzano.

    **Quando usarlo nel Marketing?**
    * Quando hai già un'idea di **quanti segmenti** vuoi identificare (es. 3 segmenti: Alto Valore, Medio Valore, Basso Valore).
    * Per trovare segmenti di clienti basati su metriche come **frequenza di acquisto, spesa media, età, ecc.** quando i segmenti sono abbastanza "sferici" e ben separati.
    * È **veloce** e scalabile per grandi basi clienti.

    **Limiti per il Marketing:**
    * Devi **specificare `K`** in anticipo, e la scelta di `K` può essere difficile.
    * Assume che i segmenti abbiano una **forma sferica** e dimensioni simili, il che non sempre è vero per i comportamenti complessi dei clienti.
    * Sensibile ai **clienti outlier**, che possono spostare i centroidi.
    """)

with st.expander("🔬 DBSCAN: Come Identifica i Segmenti e gli Outlier?"):
    st.markdown("""
    **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) raggruppa i clienti che sono vicini tra loro in base a una stima di densità. È ottimo per trovare segmenti di forma arbitraria e, cosa fondamentale per il marketing, **identificare esplicitamente i clienti "rumore" o "outlier"**.

    **Concetti chiave per il Marketing:**
    * **`Epsilon (eps)` ($\epsilon$)**: La distanza massima per considerare due clienti "vicini". Pensa a quanto devono essere simili due clienti per essere considerati parte dello stesso gruppo denso.
    * **`Min Samples (MinPts)`**: Il numero minimo di clienti che devono essere vicini tra loro per formare un "nucleo" di un segmento.

    **Tipi di Clienti identificati:**
    1.  **Core Point (Cliente Nucleo)**: Un cliente che ha un numero sufficiente di altri clienti nel suo raggio $\epsilon$. Questi sono i "rappresentanti" centrali di un segmento.
    2.  **Border Point (Cliente di Confine)**: Un cliente che non è un cliente nucleo, ma è nel raggio $\epsilon$ di un cliente nucleo. Questi sono ai margini di un segmento.
    3.  **Noise Point (Cliente Rumore/Outlier)**: Un cliente che non è né nucleo né di confine. Questi sono i clienti "insoliti" o "anomali" che non rientrano in nessun segmento denso.

    **Quando usarlo nel Marketing?**
    * Quando non sai **quanti segmenti** esistono nella tua base clienti.
    * Per trovare **segmenti di clienti con forme e dimensioni complesse** (es. clienti che seguono un percorso di acquisto a "U" o a "S").
    * Per individuare facilmente i **clienti outlier** (es. acquirenti fraudolenti, clienti con comportamenti estremamente inusuali) che necessitano di attenzione speciale o esclusione da certe campagne.
    * Utile per segmentare dati geolocalizzati o pattern di navigazione web.

    **Limiti per il Marketing:**
    * La performance dipende molto dalla scelta di `eps` e `MinPts`. Trovare i valori giusti può richiedere sperimentazione e conoscenza del dominio.
    * Può faticare con segmenti di **densità molto diverse** (es. un segmento di clienti ad alta frequenza molto compatto e un segmento di clienti occasionali molto sparsi).
    """)

st.markdown("---")
st.caption("Applicazione didattica per visualizzare algoritmi di clustering per scopi di marketing. Creato con Streamlit.")
