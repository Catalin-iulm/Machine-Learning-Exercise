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
su un dataset simulato di clienti di un supermercato, con caratteristiche realistiche e **cluster "nascosti"**
per una scoperta di segmenti meno ovvia.
Modifica i parametri per vedere quali pattern di clienti emergono!
""")

# --- Funzione per Generare Dati Sintetici per Marketing (Supermercato) ---
def generate_customer_data(n_samples, random_state):
    np.random.seed(random_state)

    # Define prototypes for different customer types, aiming for non-obvious clusters
    # [Age, Income, Visits, Time_on_site]
    prototypes = [
        # 1. Young, Tech-Savvy, Moderate Income (e.g., students/early career, active online)
        [22, 25000, 25, 45],
        # 2. Established Professionals, High Income, Moderate Digital (e.g., busy families)
        [40, 60000, 10, 15],
        # 3. Older, Lower Income, Less Digital (e.g., retirees, traditional shoppers)
        [65, 30000, 3, 5],
        # 4. "Digital Seniors": Older, Moderate Income, Surprisingly Active Online
        [60, 40000, 15, 30],
        # 5. "High-Roller Digital": Any age, Very High Income, High Digital Engagement (but maybe less frequent visits)
        [35, 90000, 8, 35],
        # 6. "Budget-Conscious Online": Young/Middle, Lower Income, High Frequency Online (coupon hunting)
        [30, 20000, 18, 20]
    ]

    # Standard deviations for each feature to add variability (adjust as needed for more/less distinct clusters)
    # [Age_std, Income_std, Visits_std, Time_std]
    stds = [
        [4, 5000, 7, 10], # Prototype 1
        [6, 10000, 4, 7], # Prototype 2
        [5, 8000, 2, 3],  # Prototype 3
        [7, 9000, 6, 10], # Prototype 4
        [10, 15000, 5, 12],# Prototype 5
        [5, 4000, 6, 8]   # Prototype 6
    ]

    data = []
    num_prototypes = len(prototypes)
    
    # Distribute samples among prototypes, slightly biased towards more common ones
    samples_per_proto = [n_samples // num_prototypes] * num_prototypes
    # Adjust for remainder
    for i in range(n_samples % num_prototypes):
        samples_per_proto[i] += 1

    for i, proto in enumerate(prototypes):
        num_current_samples = samples_per_proto[i]
        
        ages = np.random.normal(proto[0], stds[i][0], num_current_samples)
        incomes = np.random.normal(proto[1], stds[i][1], num_current_samples)
        visits = np.random.normal(proto[2], stds[i][2], num_current_samples)
        time_on_site = np.random.normal(proto[3], stds[i][3], num_current_samples)
        
        # Ensure values are within realistic bounds
        ages = np.clip(ages, 18, 70).astype(int)
        incomes = np.clip(incomes, 10000, 180000).astype(int)
        visits = np.clip(visits, 1, 50).astype(int)
        time_on_site = np.clip(time_on_site, 1, 90).astype(int)

        genders = np.random.choice(['Uomo', 'Donna'], num_current_samples)
        
        for j in range(num_current_samples):
            data.append([ages[j], genders[j], incomes[j], visits[j], time_on_site[j]])

    df_unscaled = pd.DataFrame(data, columns=['Età', 'Sesso', 'Reddito Medio Annuo (€)', 'Frequenza Visite al Sito/App (n/mese)', 'Tempo Medio Permanenza sul Sito/App (min)'])
    
    # Shuffle the DataFrame to mix the prototype-generated samples
    df_unscaled = df_unscaled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Select numerical features for clustering and scale them
    numerical_features = ['Età', 'Reddito Medio Annuo (€)', 'Frequenza Visite al Sito/App (n/mese)', 'Tempo Medio Permanenza sul Sito/App (min)']
    X_unscaled_numerical = df_unscaled[numerical_features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled_numerical)
    
    return X_scaled, df_unscaled, numerical_features # Return numerical features list for consistency

# --- Mapping per Nomi Features Marketing ---
# This is now generated dynamically from the numerical_features list
# as returned by generate_customer_data
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
    n_samples_data = st.slider("Numero di 'Clienti' Simulati", 500, 3000, 1500, step=100)
    random_state_ds = st.slider("Seed Generazione Dati (per riproducibilità)", 0, 100, 42)
    st.markdown("---")

    st.subheader("2. Scegli le Caratteristiche per il Grafico")
    # Dynamically get keys from the feature_names_mapping dictionary
    available_features = list(feature_names_mapping.keys())
    plot_feature_x = st.selectbox("Caratteristica Asse X:", available_features, index=0) # Default to Età
    plot_feature_y = st.selectbox("Caratteristica Asse Y:", available_features, index=2) # Default to Frequenza Visite

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
        # Start with a K that might reveal some of the prototypes, e.g., 4 or 5
        k_clusters_param = st.slider("Numero di Segmenti (K) da Trovare", 1, 10, 5,
                                     help="Quanti segmenti di clienti l'algoritmo K-Means cercherà.")
        kmeans_random_state_param = st.slider("Seed K-Means (per inizializzazione)", 0, 100, 1,
                                             help="Controlla l'inizializzazione dei centroidi per la riproducibilità. Cambialo per vedere diverse configurazioni iniziali.")
    elif algoritmo_scelto == "DBSCAN":
        # Adjusted default eps and min_samples for more structured data
        eps_param = st.slider("Epsilon (eps) - Raggio di Vicinato", 0.05, 2.0, 0.5, step=0.01,
                             help="Distanza massima per considerare due 'clienti' vicini. Dato che i dati sono scalati internamente, un valore tra 0.4 e 0.8 è spesso un buon punto di partenza per questi dati.")
        min_samples_param = st.slider("Min Samples - Densità Minima", 1, 50, 10,
                                     help="Numero minimo di 'clienti' in un vicinato per formare un segmento denso. I punti sotto questa soglia potrebbero essere considerati rumore.")

# --- Generazione Dati ---
X_data_scaled_full, df_data_unscaled, numerical_features_list = generate_customer_data(n_samples_data, random_state_ds)


# --- Contesto Marketing per il Dataset ---
st.markdown("---")
st.subheader("💡 Contesto Marketing del Dataset 'Clienti Simulati del Supermercato':")
st.info("""
    Questo dataset simula una base clienti di un supermercato, con caratteristiche come **Età**, **Sesso**, **Reddito Medio Annuo**,
    **Frequenza Visite al Sito/App** e **Tempo Medio di Permanenza sul Sito/App**.
    I dati sono stati generati per creare **segmenti di clienti con interazioni più complesse e meno ovvie** tra le variabili,
    permettendo agli algoritmi di clustering di **scoprire pattern nascosti**.
    Puoi aspettarti di trovare combinazioni interessanti come:
    * **Giovani Digitali con Stili di Vita Diversi:** Alcuni con reddito basso (studenti), altri con reddito più alto (startup/influencer).
    * **Senior Connessi:** Età avanzata, ma sorprendentemente attivi e con una buona permanenza online.
    * **Clienti Ad Alto Reddito Ma Poco Frequenti:** Spesa elevata, ma meno tempo sul sito.

    L'obiettivo del clustering è **identificare questi segmenti emergenti** per sviluppare strategie di marketing più efficaci e mirate,
    andando oltre le segmentazioni più tradizionali e superficiali.
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
        file_name="clienti_supermercato_simulati_nascosti.csv",
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
    
    # Centroids are in the full feature space (scaled), need to inverse transform them for plotting on unscaled axes
    temp_scaler = StandardScaler()
    # Fit the scaler again on the *original numerical data* to ensure inverse_transform works correctly
    temp_scaler.fit(df_data_unscaled[numerical_features_list].values) 
    
    cluster_centers_coords_full_scaled = kmeans_model.cluster_centers_
    cluster_centers_coords_full_unscaled = temp_scaler.inverse_transform(cluster_centers_coords_full_scaled)
    
    # Get the indices of the selected features within the `numerical_features_list`
    idx_x_numerical = numerical_features_list.index(plot_feature_x)
    idx_y_numerical = numerical_features_list.index(plot_feature_y)

    # Extract the relevant two dimensions for plotting (unscaled)
    cluster_centers_coords_plot = cluster_centers_coords_full_unscaled[:, [idx_x_numerical, idx_y_numerical]]
    
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
    st.subheader(f"Grafico dei Segmenti di Clienti: '{plot_feature_x}' vs '{plot_feature_y}' (Dati Originali)")
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
    
    # Use unscaled data for plotting directly from the DataFrame
    plot_x_data = df_data_unscaled[plot_feature_x]
    plot_y_data = df_data_unscaled[plot_feature_y]

    for label_val_plot in unique_plot_labels:
        mask_plot = (labels_pred == label_val_plot)
        current_color_plot = color_map_for_plot.get(label_val_plot, (0,0,0,1))
        marker_style_plot = 'x' if label_val_plot == -1 else 'o'
        point_size_plot = 40 if label_val_plot == -1 else 60
        plot_legend_label = f'Clienti Rumore/Outlier (-1)' if label_val_plot == -1 else f'Segmento {label_val_plot}'

        ax_cluster.scatter(plot_x_data[mask_plot], plot_y_data[mask_plot], # Use unscaled data for plot
                           facecolor=current_color_plot, marker=marker_style_plot, s=point_size_plot,
                           label=plot_legend_label, alpha=0.8,
                           edgecolor='k' if label_val_plot !=-1 else 'none', linewidth=0.5 if label_val_plot !=-1 else 0)

    if algoritmo_scelto == "K-Means" and cluster_centers_coords_plot is not None:
        ax_cluster.scatter(cluster_centers_coords_plot[:, 0], cluster_centers_coords_plot[:, 1],
                           marker='P', s=250, facecolor='red', label='Centroide Segmento',
                           edgecolor='black', linewidth=1.5, zorder=10)

    # Dynamic Axis Labels for unscaled plot
    ax_cluster.set_title(f'Segmentazione con {algoritmo_scelto}')
    ax_cluster.set_xlabel(f"{plot_feature_x}") # No "(Scalata)"
    ax_cluster.set_ylabel(f"{plot_feature_y}") # No "(Scalata)"
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
