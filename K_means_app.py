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
su un dataset simulato di clienti di un supermercato, con **nuove feature e legami "nascosti"**
per scoprire segmenti di clienti inattesi e più ricchi di insight.
Modifica i parametri per vedere quali pattern di clienti emergono!
""")

# --- Funzione per Generare Dati Sintetici per Marketing (Supermercato) ---
def generate_customer_data(n_samples, random_state):
    np.random.seed(random_state)

    data = []
    
    # Define customer "archetypes" with more nuanced and interacting features
    # Each archetype is a tuple: (avg_age, avg_income, avg_visits_app, avg_time_on_site, 
    #                            avg_bio_freq, avg_promo_propensity, avg_in_store_time, avg_category_variety, avg_social_engagement)
    # Plus their respective standard deviations
    
    # Archetype 1: Young Urban, Tech-Savvy, Bio/Ethical Conscious, Moderate Income
    # Focus: Digital convenience, healthy choices, active online
    archetype_1_means = [28, 35000, 20, 40, 8, 50, 15, 10, 12]
    archetype_1_stds = [5, 8000, 6, 15, 3, 15, 5, 3, 5]

    # Archetype 2: Established Families, Value-Driven, In-Store Focused, Higher Income
    # Focus: Bulk buying, promos, less online engagement, physical shopping
    archetype_2_means = [42, 60000, 5, 10, 4, 80, 45, 15, 2]
    archetype_2_stds = [7, 12000, 3, 5, 2, 10, 10, 4, 1]

    # Archetype 3: Seniors, Less Digital, Routine Shoppers, Stable Income
    # Focus: Comfort, routine, familiar products, in-store interaction
    archetype_3_means = [68, 40000, 2, 5, 1, 30, 60, 5, 1]
    archetype_3_stds = [4, 7000, 1, 2, 1, 10, 15, 2, 1]
    
    # Archetype 4: "Digital Seniors" - Active Retirees, Moderate Income, High Online Engagement
    # Focus: Leisurely Browse online, good deals, home delivery
    archetype_4_means = [62, 45000, 15, 30, 5, 60, 20, 8, 8]
    archetype_4_stds = [6, 9000, 5, 10, 2, 15, 7, 3, 4]

    # Archetype 5: High-Net-Worth, Premium Buyers, Low Propensity for Promos, Any Age
    # Focus: Quality over price, wide variety, less frequent online visits but significant time
    archetype_5_means = [45, 100000, 7, 30, 10, 20, 30, 18, 5]
    archetype_5_stds = [12, 20000, 4, 10, 4, 10, 10, 5, 3]

    # Archetype 6: Budget-Conscious, Any Age, High Promo Propensity, High In-Store Time (couponing)
    # Focus: Seeking best deals, price sensitive, mix of digital and physical
    archetype_6_means = [35, 25000, 10, 15, 3, 95, 50, 12, 4]
    archetype_6_stds = [10, 5000, 4, 7, 2, 5, 12, 4, 2]


    archetypes = [
        (archetype_1_means, archetype_1_stds),
        (archetype_2_means, archetype_2_stds),
        (archetype_3_means, archetype_3_stds),
        (archetype_4_means, archetype_4_stds),
        (archetype_5_means, archetype_5_stds),
        (archetype_6_means, archetype_6_stds)
    ]

    num_archetypes = len(archetypes)
    samples_per_archetype = [n_samples // num_archetypes] * num_archetypes
    for i in range(n_samples % num_archetypes): # Distribute remainder
        samples_per_archetype[i] += 1
    
    for i, (means, stds) in enumerate(archetypes):
        num_current_samples = samples_per_archetype[i]
        
        ages = np.random.normal(means[0], stds[0], num_current_samples)
        incomes = np.random.normal(means[1], stds[1], num_current_samples)
        visits_app = np.random.normal(means[2], stds[2], num_current_samples)
        time_on_site = np.random.normal(means[3], stds[3], num_current_samples)
        freq_bio = np.random.normal(means[4], stds[4], num_current_samples)
        promo_propensity = np.random.normal(means[5], stds[5], num_current_samples)
        in_store_time = np.random.normal(means[6], stds[6], num_current_samples)
        category_variety = np.random.normal(means[7], stds[7], num_current_samples)
        social_engagement = np.random.normal(means[8], stds[8], num_current_samples)


        # Ensure values are within realistic bounds and positive
        ages = np.clip(ages, 18, 70).astype(int)
        incomes = np.clip(incomes, 10000, 200000).astype(int)
        visits_app = np.clip(visits_app, 0, 40).astype(int)
        time_on_site = np.clip(time_on_site, 0, 90).astype(int)
        freq_bio = np.clip(freq_bio, 0, 15).astype(int) # Max 15 bio purchases/month
        promo_propensity = np.clip(promo_propensity, 0, 100).astype(int) # Score 0-100
        in_store_time = np.clip(in_store_time, 5, 90).astype(int) # Min 5, Max 90 mins
        category_variety = np.clip(category_variety, 1, 25).astype(int) # From 1 to 25 categories
        social_engagement = np.clip(social_engagement, 0, 20).astype(int) # Likes/comments/month

        genders = np.random.choice(['Uomo', 'Donna'], num_current_samples)
        
        for j in range(num_current_samples):
            data.append([
                ages[j], genders[j], incomes[j], visits_app[j], time_on_site[j],
                freq_bio[j], promo_propensity[j], in_store_time[j], category_variety[j], social_engagement[j]
            ])

    df_unscaled = pd.DataFrame(data, columns=[
        'Età', 'Sesso', 'Reddito Medio Annuo (€)', 
        'Frequenza Visite App/Sito (n/mese)', 'Tempo Medio Permanenza App/Sito (min)',
        'Frequenza Acquisti Bio/Salutari (n/mese)', 'Propensione Offerte Speciali (punti)',
        'Tempo Medio Speso in Negozio (min)', 'Varietà Categorie Prodotti Acquistati',
        'Engagement Social Media Supermercato (likes/commenti/mese)'
    ])
    
    # Shuffle the DataFrame to mix the archetype-generated samples
    df_unscaled = df_unscaled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Select numerical features for clustering and scale them
    numerical_features = [col for col in df_unscaled.columns if col not in ['Sesso']]
    X_unscaled_numerical = df_unscaled[numerical_features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled_numerical)
    
    return X_scaled, df_unscaled, numerical_features # Return numerical features list for consistency

# --- Funzione per Analizzare le Caratteristiche dei Cluster (Cluster Profiling) ---
def analyze_cluster_characteristics(df_original, labels, numerical_features):
    df_with_labels = df_original.copy()
    df_with_labels['Cluster'] = labels

    cluster_profiles = {}
    for cluster_id in sorted(df_with_labels['Cluster'].unique()):
        if cluster_id == -1:
            cluster_name = "Rumore/Outlier"
        else:
            cluster_name = f"Segmento {cluster_id}"
        
        cluster_data = df_with_labels[df_with_labels['Cluster'] == cluster_id]
        
        profile = {
            'N° Clienti': len(cluster_data)
        }
        for feature in numerical_features:
            profile[f'Media {feature}'] = cluster_data[feature].mean()
            profile[f'Mediana {feature}'] = cluster_data[feature].median()
            profile[f'Std Dev {feature}'] = cluster_data[feature].std()
        
        cluster_profiles[cluster_name] = profile
    
    return cluster_profiles, df_with_labels

# --- Mapping per Nomi Features Marketing (pre-populate for sidebar) ---
# This will be dynamically generated later based on the actual numerical features from generate_customer_data
feature_names_mapping_placeholder = {
    "Età": "Età",
    "Reddito Medio Annuo (€)": "Reddito Medio Annuo (€)",
    "Frequenza Visite App/Sito (n/mese)": "Frequenza Visite App/Sito (n/mese)",
    "Tempo Medio Permanenza App/Sito (min)": "Tempo Medio Permanenza App/Sito (min)",
    "Frequenza Acquisti Bio/Salutari (n/mese)": "Frequenza Acquisti Bio/Salutari (n/mese)",
    "Propensione Offerte Speciali (punti)": "Propensione Offerte Speciali (punti)",
    "Tempo Medio Speso in Negozio (min)": "Tempo Medio Speso in Negozio (min)",
    "Varietà Categorie Prodotti Acquistati": "Varietà Categorie Prodotti Acquistati",
    "Engagement Social Media Supermercato (likes/commenti/mese)": "Engagement Social Media Supermercato (likes/commenti/mese)"
}


# --- Sidebar per Controlli Globali ---
with st.sidebar:
    st.header("⚙️ Configurazione Esperimento")

    st.subheader("1. Generazione Dataset 'Clienti'")
    n_samples_data = st.slider("Numero di 'Clienti' Simulati", 500, 5000, 2000, step=100) # Increased max samples
    random_state_ds = st.slider("Seed Generazione Dati (per riproducibilità)", 0, 100, 42)
    st.markdown("---")

    # --- Generazione Dati (needed here to get feature names for selectbox) ---
    # We call it once with a dummy number of samples to get the list of numerical features and df structure
    # This df_temp is only used for feature names, not for the actual clustering
    X_data_scaled_full_temp, df_data_unscaled_temp, numerical_features_list = generate_customer_data(10, random_state_ds)
    # Update feature_names_mapping with the actual generated numerical features
    feature_names_mapping = {feat: feat for feat in numerical_features_list}


    st.subheader("2. Scegli le Caratteristiche per il Grafico")
    available_features = list(feature_names_mapping.keys())
    # Try to set sensible defaults for visualization, e.g., income vs bio freq
    plot_feature_x = st.selectbox("Caratteristica Asse X:", available_features, index=available_features.index('Reddito Medio Annuo (€)'))
    plot_feature_y = st.selectbox("Caratteristica Asse Y:", available_features, index=available_features.index('Frequenza Acquisti Bio/Salutari (n/mese)'))

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
        # With 6 archetypes, maybe 5-7 clusters is a good starting point
        k_clusters_param = st.slider("Numero di Segmenti (K) da Trovare", 1, 10, 6,
                                     help="Quanti segmenti di clienti l'algoritmo K-Means cercherà. Prova a far variare per vedere come cambiano i raggruppamenti.")
        kmeans_random_state_param = st.slider("Seed K-Means (per inizializzazione)", 0, 100, 1,
                                             help="Controlla l'inizializzazione dei centroidi per la riproducibilità. Cambialo per vedere diverse configurazioni iniziali.")
    elif algoritmo_scelto == "DBSCAN":
        # Adjusted default eps and min_samples for potentially clearer clusters with more dimensions
        eps_param = st.slider("Epsilon (eps) - Raggio di Vicinato", 0.05, 3.0, 0.7, step=0.01, # Increased max eps
                             help="Distanza massima per considerare due 'clienti' vicini nel loro spazio multi-dimensionale scalato. Un valore tra 0.5 e 1.0 è un buon punto di partenza.")
        min_samples_param = st.slider("Min Samples - Densità Minima", 1, 100, 15, # Increased max min_samples
                                     help="Numero minimo di 'clienti' in un vicinato per formare un segmento denso. Più alto è, più rigorosi sono i cluster e più rumore può essere trovato.")

# --- Generazione Dati Finale (with chosen n_samples_data) ---
X_data_scaled_full, df_data_unscaled, numerical_features_list = generate_customer_data(n_samples_data, random_state_ds)


# --- Contesto Marketing per il Dataset ---
st.markdown("---")
st.subheader("💡 Contesto Marketing del Dataset 'Clienti Simulati del Supermercato':")
st.info("""
    Questo dataset simula una base clienti di un supermercato, ora con un set di feature più ampio e interconnesso:
    * **Età** e **Reddito Medio Annuo (€)**: I classici demografici.
    * **Frequenza Visite App/Sito (n/mese)** e **Tempo Medio Permanenza App/Sito (min)**: Comportamento digitale.
    * **Frequenza Acquisti Bio/Salutari (n/mese)**: Indica uno stile di vita e preferenze specifiche.
    * **Propensione Offerte Speciali (punti)**: Misura la sensibilità al prezzo e alle promozioni.
    * **Tempo Medio Speso in Negozio (min)**: Comportamento di acquisto fisico.
    * **Varietà Categorie Prodotti Acquistati**: Quanto un cliente esplora l'offerta del supermercato.
    * **Engagement Social Media Supermercato (likes/commenti/mese)**: Indica fedeltà e coinvolgimento con il brand online.

    I dati sono stati generati per creare **segmenti di clienti con interazioni più complesse e meno ovvie** tra le variabili.
    Questo significa che un "cliente tipo" potrebbe non essere definito solo dalla sua età o dal suo reddito,
    ma da una **combinazione di abitudini e preferenze**, rendendo i cluster non immediatamente visibili in un semplice grafico 2D.
    L'obiettivo è scoprire segmenti come:
    * **"Famiglie Consapevoli e Cacciatrici di Offerte":** Reddito medio-alto, attenti al bio, alta propensione alle offerte, tempo speso sia online che in negozio.
    * **"Senior Digitali e Sociali":** Età avanzata, ma molto attivi sull'app e sui social media del supermercato.
    * **"Professionisti Premium Online":** Alto reddito, acquisti di qualità, poco tempo in negozio, ma acquisti online efficienti e vari.

    Gli algoritmi di clustering sono gli strumenti ideali per **scoprire questi legami nascosti** e fornire insight preziosi
    per campagne marketing mirate, personalizzazione dell'esperienza e sviluppo prodotti/servizi.
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
    st.info("Nota: Gli algoritmi di clustering lavorano sulle **caratteristiche numeriche scalate** per garantire che tutte abbiano la stessa importanza. La colonna 'Sesso' è inclusa solo per contesto e non viene usata direttamente nel clustering multidimensionale, ma potrebbe essere utilizzata per analisi successive sui cluster trovati.")

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
    
    # Ensure there are enough colors even if num_actual_clusters_for_cmap is 0 or 1
    if num_actual_clusters_for_cmap == 0:
        cluster_colors_palette = [plt.cm.viridis(0.5)] # Single color for case no clusters
    else:
        cluster_colors_palette = plt.cm.viridis(np.linspace(0, 1, num_actual_clusters_for_cmap))
    
    color_map_for_plot = {}
    color_idx_plot = 0
    for lbl_plot in unique_plot_labels:
        if lbl_plot == -1:
            color_map_for_plot[lbl_plot] = (0.5, 0.5, 0.5, 0.7) # Grigio per rumore (outlier)
        else:
            if color_idx_plot < len(cluster_colors_palette):
                color_map_for_plot[lbl_plot] = cluster_colors_palette[color_idx_plot]
            else:
                color_map_for_plot[lbl_plot] = (np.random.rand(), np.random.rand(), np.random.rand(), 0.8) # Fallback for too many clusters
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
    ax_cluster.set_xlabel(f"{plot_feature_x}")
    ax_cluster.set_ylabel(f"{plot_feature_y}")
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

st.markdown("---")
st.header("🔬 Analisi Dettagliata dei Segmenti (Profili dei Cluster)")

if len(set(labels_pred)) > 0:
    cluster_profiles, df_with_labels = analyze_cluster_characteristics(df_data_unscaled, labels_pred, numerical_features_list)
    
    all_cluster_names = ["Tutti i Cluster"] + sorted([name for name in cluster_profiles.keys() if name != "Rumore/Outlier"])
    if "Rumore/Outlier" in cluster_profiles:
        all_cluster_names.append("Rumore/Outlier") # Add noise at the end

    cluster_to_show = st.selectbox("Seleziona un Segmento per Analizzare il Profilo Dettagliato:", all_cluster_names)

    if cluster_to_show == "Tutti i Cluster":
        st.subheader("Panoramica Comparativa delle Medie delle Caratteristiche per Tutti i Segmenti")
        
        # Prepare data for bar chart - Mean of each feature for each cluster
        cluster_means_df = pd.DataFrame()
        for c_name, profile_data in cluster_profiles.items():
            if c_name == "Rumore/Outlier": continue # Exclude noise from this comparative bar chart if preferred
            means_only = {k.replace('Media ', ''): v for k, v in profile_data.items() if 'Media ' in k}
            cluster_means_df = pd.concat([cluster_means_df, pd.DataFrame(means_only, index=[c_name])])
        
        if not cluster_means_df.empty:
            st.bar_chart(cluster_means_df.T) # Transpose to have features on y-axis and clusters as bars
            st.info("Il grafico a barre mostra la media di ciascuna caratteristica per ogni segmento, permettendo un confronto rapido. Puoi notare come alcuni segmenti si distinguano per valori alti o bassi in specifiche caratteristiche.")
        else:
            st.warning("Nessun segmento valido trovato per la comparazione (potrebbe esserci solo rumore).")

        st.subheader("Dettagli Riassuntivi di Tutti i Segmenti")
        summary_data = []
        for c_name, profile_data in cluster_profiles.items():
            row = {'Segmento': c_name, 'N° Clienti': profile_data['N° Clienti']}
            for feature in numerical_features_list:
                row[f'Media {feature}'] = f"{profile_data[f'Media {feature}']:.2f}"
            summary_data.append(row)
        st.dataframe(pd.DataFrame(summary_data).set_index('Segmento'))
        st.info("Questa tabella riassume le caratteristiche medie di tutti i segmenti trovati, inclusi i 'clienti rumore'.")

    else:
        st.subheader(f"Profilo Dettagliato del {cluster_to_show}")
        profile_data = cluster_profiles[cluster_to_show]
        
        st.write(f"**Numero di Clienti in questo Segmento:** {profile_data['N° Clienti']}")
        
        profile_df = pd.DataFrame({
            'Statistica': ['Media', 'Mediana', 'Deviazione Standard'],
            **{
                feature: [
                    profile_data[f'Media {feature}'],
                    profile_data[f'Mediana {feature}'],
                    profile_data[f'Std Dev {feature}']
                ] for feature in numerical_features_list
            }
        }).set_index('Statistica')
        st.dataframe(profile_df)
        st.info(f"""
        Questa tabella fornisce una visione approfondita delle caratteristiche dei clienti nel **{cluster_to_show}**.
        * La **Media** indica il valore tipico per la caratteristica.
        * La **Mediana** è meno sensibile agli outlier e mostra il valore centrale.
        * La **Deviazione Standard (Std Dev)** indica la variabilità all'interno del segmento per quella caratteristica: una Std Dev bassa significa che i clienti sono molto simili per quella caratteristica, una alta indica più eterogeneità.
        """)
        
        # Optional: Bar chart for the selected cluster's mean features
        st.subheader(f"Visualizzazione delle Medie delle Caratteristiche per il {cluster_to_show}")
        mean_features_for_single_cluster = {k.replace('Media ', ''): v for k, v in profile_data.items() if 'Media ' in k}
        # Create a DataFrame suitable for bar_chart, with a single row for the selected cluster
        df_for_bar_chart = pd.DataFrame([mean_features_for_single_cluster])
        st.bar_chart(df_for_bar_chart.T)
        st.info("Questo grafico a barre evidenzia i valori medi delle caratteristiche per il segmento selezionato. Osserva quali caratteristiche hanno valori particolarmente alti o bassi per questo segmento rispetto agli altri.")
else:
    st.info("Nessun cluster trovato. Assicurati di aver generato i dati e di aver impostato correttamente i parametri dell'algoritmo.")


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
    * Devi **specificare `K`** in anticipo, e la scelta di `K` può essere difficile, specialmente con cluster "nascosti".
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
    * La performance dipende molto dalla scelta di `eps` e `MinPts`. Trovare i valori giusti può richiedere sperimentazione e conoscenza del dominio, specialmente con cluster sovrapposti.
    * Può faticare con segmenti di **densità molto diverse** (es. un segmento di clienti ad alta frequenza molto compatto e un segmento di clienti occasionali molto sparsi).
    """)

st.markdown("---")
st.caption("Applicazione didattica per visualizzare algoritmi di clustering per scopi di marketing. Creato con Streamlit.")
