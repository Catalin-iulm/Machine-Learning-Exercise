import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="Visualizzatore Algoritmi di Clustering")

# --- Titolo e Introduzione ---
st.title("🔬 Visualizzatore Interattivo di Algoritmi di Clustering")
st.markdown("""
Questa applicazione ti permette di esplorare il funzionamento degli algoritmi di clustering **K-Means** e **DBSCAN**
su diversi tipi di dataset sintetici. Modifica i parametri del dataset e dell'algoritmo per vedere come cambiano i risultati!
""")

# --- Funzione per Generare Dati Sintetici ---
def generate_data(dataset_type, n_samples, noise, random_state, n_blobs_centers=3, blob_std=1.0):
    X = np.array([[]]) # Initialize X
    y_true = None # True labels, useful for some datasets but not directly used by clustering

    if dataset_type == "Blobs Ben Separati":
        X, y_true = make_blobs(n_samples=n_samples, centers=n_blobs_centers, cluster_std=0.6,
                               random_state=random_state)
    elif dataset_type == "Blobs Misti (Varianza Alta)":
        X, y_true = make_blobs(n_samples=n_samples, centers=n_blobs_centers, cluster_std=blob_std if blob_std > 0 else 1.5,
                               random_state=random_state)
    elif dataset_type == "Lune (Moons)":
        X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == "Cerchi Concentrici (Circles)":
        X, y_true = make_circles(n_samples=n_samples, factor=0.5, noise=noise, random_state=random_state)
    elif dataset_type == "Dati Anisotropi (Ellittici)": # Challenging for K-Means default
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso, _ = make_blobs(n_samples=n_samples, centers=n_blobs_centers, random_state=random_state, cluster_std=0.7)
        X = np.dot(X_aniso, transformation)
        y_true = None # True labels are harder to map after transformation for simple demo
    elif dataset_type == "Nessuna Struttura Evidente (Random)":
        X = np.random.rand(n_samples, 2) * 10 # Spread out points
        y_true = None
        
    # Standard Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_true


# --- Sidebar per Controlli Globali ---
with st.sidebar:
    st.header("⚙️ Configurazione Esperimento")

    st.subheader("1. Scegli il Dataset Sintetico")
    dataset_options = [
        "Blobs Ben Separati", "Blobs Misti (Varianza Alta)", "Lune (Moons)",
        "Cerchi Concentrici (Circles)", "Dati Anisotropi (Ellittici)", "Nessuna Struttura Evidente (Random)"
    ]
    dataset_type = st.selectbox("Tipo di Dataset:", dataset_options)

    n_samples_data = st.slider("Numero di Punti Dati", 100, 1500, 300, step=50)
    
    # Dataset specific parameters
    if "Blobs" in dataset_type:
        n_centers_blobs = st.slider("Numero di Centri (Blobs)", 2, 5, 3)
        if dataset_type == "Blobs Misti (Varianza Alta)":
             blob_std_param = st.slider("Deviazione Standard Blobs", 0.5, 3.0, 1.8, step=0.1)
        else:
            blob_std_param = 0.6 # Fixed for well-separated
    elif dataset_type in ["Lune (Moons)", "Cerchi Concentrici (Circles)"]:
        noise_level = st.slider("Livello di Rumore Dataset", 0.01, 0.3, 0.05, step=0.01)
    else: # For Anisotropic, Random
        noise_level = 0.05 # Dummy, not directly used by all generators in the same way
        n_centers_blobs = 3 # Used by anisotropic
        blob_std_param = 1.0 # Dummy

    random_state_ds = st.slider("Seed Generazione Dati", 0, 100, 42)
    st.markdown("---")

    st.subheader("2. Scegli l'Algoritmo di Clustering")
    algoritmo_scelto = st.radio(
        "Algoritmo:", ("K-Means", "DBSCAN"), horizontal=True
    )
    st.markdown("---")

    st.subheader(f"3. Parametri {algoritmo_scelto}")
    if algoritmo_scelto == "K-Means":
        k_clusters_param = st.slider("Numero di Cluster (K)", 1, 10, 3,
                                     help="Quanti gruppi l'algoritmo K-Means cercherà.")
        kmeans_random_state_param = st.slider("Seed K-Means (per inizializzazione)", 0, 100, 1,
                                              help="Controlla l'inizializzazione dei centroidi per la riproducibilità.")
    elif algoritmo_scelto == "DBSCAN":
        eps_param = st.slider("Epsilon (eps)", 0.05, 2.0, 0.2, step=0.01, # Adjusted range for scaled data
                              help="Raggio massimo per considerare i punti come vicini.")
        min_samples_param = st.slider("Min Samples", 1, 30, 5,
                                     help="Numero minimo di punti in un intorno per formare un cluster denso.")

# --- Generazione Dati ---
X_data, y_true_data = generate_data(dataset_type, n_samples_data,
                                    noise_level if 'noise_level' in locals() else 0.05,
                                    random_state_ds,
                                    n_centers_blobs if 'n_centers_blobs' in locals() else 3,
                                    blob_std_param if 'blob_std_param' in locals() else 1.0)

# --- Visualizzazione Dati Originali (Opzionale) ---
# with st.expander("Visualizza Dati Generati (Prima del Clustering)"):
#     fig_orig, ax_orig = plt.subplots(figsize=(8, 6))
#     ax_orig.scatter(X_data[:, 0], X_data[:, 1], c='gray', alpha=0.6)
#     ax_orig.set_title(f"Dataset: {dataset_type} (Dati Scalati)")
#     ax_orig.set_xlabel("Feature 1 (Scalata)")
#     ax_orig.set_ylabel("Feature 2 (Scalata)")
#     ax_orig.grid(True, linestyle='--', alpha=0.7)
#     st.pyplot(fig_orig)

st.header(f"🚀 Esecuzione e Risultati: {algoritmo_scelto}")

# --- Esecuzione Clustering ---
labels_pred = []
cluster_centers_coords = None
n_clusters_found_val = 0
inertia_val = None
n_noise_points = 0
silhouette_avg = None

if algoritmo_scelto == "K-Means":
    kmeans_model = KMeans(n_clusters=k_clusters_param, random_state=kmeans_random_state_param, n_init='auto')
    labels_pred = kmeans_model.fit_predict(X_data)
    cluster_centers_coords = kmeans_model.cluster_centers_
    n_clusters_found_val = len(set(labels_pred))
    inertia_val = kmeans_model.inertia_
elif algoritmo_scelto == "DBSCAN":
    dbscan_model = DBSCAN(eps=eps_param, min_samples=min_samples_param)
    labels_pred = dbscan_model.fit_predict(X_data)
    unique_labels_set = set(labels_pred)
    n_clusters_found_val = len(unique_labels_set) - (1 if -1 in unique_labels_set else 0)
    n_noise_points = list(labels_pred).count(-1)

# Calcola Silhouette Score se ci sono cluster validi (più di 1 cluster e non tutti rumore)
if len(set(labels_pred)) > 1 and (len(set(labels_pred)) > 1 or (algoritmo_scelto == "DBSCAN" and -1 not in set(labels_pred))):
    try:
        silhouette_avg = silhouette_score(X_data, labels_pred)
    except ValueError: # Happens if only one cluster is found, or all points are noise
        silhouette_avg = None
else:
    silhouette_avg = None # Cannot compute for 1 cluster or all noise

# --- Visualizzazione dei Cluster ---
col1_plot, col2_metrics = st.columns([2,1])

with col1_plot:
    st.subheader("Grafico dei Cluster")
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 7))

    # Colormap dinamica
    unique_plot_labels = sorted(list(set(labels_pred)))
    num_actual_clusters_for_cmap = len(unique_plot_labels) -1 if -1 in unique_plot_labels else len(unique_plot_labels)
    
    # Create a color mapping
    cluster_colors_palette = plt.cm.viridis(np.linspace(0, 1, max(1, num_actual_clusters_for_cmap)))
    color_map_for_plot = {}
    color_idx_plot = 0
    for lbl_plot in unique_plot_labels:
        if lbl_plot == -1:
            color_map_for_plot[lbl_plot] = (0.5, 0.5, 0.5, 0.7) # Grigio per rumore
        else:
            if color_idx_plot < len(cluster_colors_palette):
                color_map_for_plot[lbl_plot] = cluster_colors_palette[color_idx_plot]
            else: # Fallback if more labels than palette colors (should not happen with current logic)
                color_map_for_plot[lbl_plot] = (np.random.rand(), np.random.rand(), np.random.rand(), 0.8)
            color_idx_plot +=1
    
    for label_val_plot in unique_plot_labels:
        mask_plot = (labels_pred == label_val_plot)
        current_color_plot = color_map_for_plot.get(label_val_plot, (0,0,0,1)) # Default to black if label somehow missing
        marker_style_plot = 'x' if label_val_plot == -1 else 'o'
        point_size_plot = 40 if label_val_plot == -1 else 60
        plot_legend_label = f'Rumore (-1)' if label_val_plot == -1 else f'Cluster {label_val_plot}'

        ax_cluster.scatter(X_data[mask_plot, 0], X_data[mask_plot, 1],
                           facecolor=current_color_plot, marker=marker_style_plot, s=point_size_plot,
                           label=plot_legend_label, alpha=0.8,
                           edgecolor='k' if label_val_plot !=-1 else 'none', linewidth=0.5 if label_val_plot !=-1 else 0)

    if algoritmo_scelto == "K-Means" and cluster_centers_coords is not None:
        ax_cluster.scatter(cluster_centers_coords[:, 0], cluster_centers_coords[:, 1],
                           marker='P', s=250, facecolor='red', label='Centroidi K-Means',
                           edgecolor='black', linewidth=1.5, zorder=10)

    ax_cluster.set_title(f'Dataset: {dataset_type} - Clustering con {algoritmo_scelto}')
    ax_cluster.set_xlabel("Feature 1 (Scalata)")
    ax_cluster.set_ylabel("Feature 2 (Scalata)")
    ax_cluster.legend(loc='best', fontsize='small')
    ax_cluster.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_cluster)

with col2_metrics:
    st.subheader("Metriche e Riepilogo")
    if algoritmo_scelto == "K-Means":
        st.metric(label="Numero Cluster Richiesti (K)", value=k_clusters_param)
        st.metric(label="Numero Cluster Trovati", value=n_clusters_found_val)
        if inertia_val is not None:
            st.metric(label="Inerzia (WCSS)", value=f"{inertia_val:.2f}",
                      help="Somma delle distanze quadrate dai centroidi: minore è, meglio è.")
    elif algoritmo_scelto == "DBSCAN":
        st.metric(label="Numero Cluster Trovati", value=n_clusters_found_val)
        st.metric(label="Punti Rumorosi (Outliers)", value=n_noise_points)

    if silhouette_avg is not None:
        st.metric(label="Silhouette Score", value=f"{silhouette_avg:.3f}",
                  help="Misura la separazione dei cluster (-1 a +1). Più alto è, meglio definiti sono i cluster.")
    else:
        st.info("Silhouette Score non calcolabile (es. un solo cluster trovato o tutti i punti sono rumore).")

    st.markdown("---")
    st.write("**Conteggio Punti per Cluster:**")
    if len(labels_pred) > 0:
        counts = pd.Series(labels_pred).value_counts().sort_index()
        counts.index.name = "ID Cluster"
        st.dataframe(counts.rename("Numero Punti"))
    else:
        st.write("Nessun punto clusterizzato.")


# --- Sezioni Didattiche ---
st.markdown("---")
st.header("📚 Approfondimenti sugli Algoritmi")

with st.expander("🔍 K-Means: Come Funziona?"):
    st.markdown("""
    **K-Means** mira a partizionare N osservazioni in K cluster, assegnando ogni osservazione al cluster con il centroide (media dei punti del cluster) più vicino.

    **Passaggi principali:**
    1.  **Inizializzazione**: Scegli K centroidi iniziali (casualmente o in modo più intelligente).
    2.  **Assegnazione**: Assegna ogni punto dati al centroide più vicino.
    3.  **Aggiornamento**: Ricalcola i centroidi come la media dei punti assegnati a ciascun cluster.
    4.  **Iterazione**: Ripeti i passaggi 2 e 3 finché i centroidi non si stabilizzano.

    **Punti Chiave:**
    * **Devi specificare K** (il numero di cluster) in anticipo.
    * Assume cluster di forma **sferica/globulare** e di dimensioni simili.
    * Sensibile alla **posizione iniziale dei centroidi** (l'opzione `n_init` in scikit-learn esegue l'algoritmo più volte con diverse inizializzazioni e sceglie la migliore).
    * Sensibile agli **outlier**.
    * **Veloce** e scalabile per grandi dataset (con un K ragionevole).
    """)

with st.expander("🔬 DBSCAN: Come Funziona?"):
    st.markdown("""
    **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) raggruppa i punti che sono vicini tra loro in base a una stima di densità. Può trovare cluster di forma arbitraria e identificare il rumore.

    **Concetti chiave:**
    * **`Epsilon (eps)` ($\epsilon$)**: La distanza massima tra due campioni perché uno sia considerato nel vicinato dell'altro.
    * **`Min Samples (MinPts)`**: Il numero di campioni in un vicinato perché un punto sia considerato un "core point".

    **Tipi di Punti:**
    1.  **Core Point**: Un punto con almeno `MinPts` punti nel suo $\epsilon$-vicinato (incluso se stesso).
    2.  **Border Point**: Un punto che non è un core point, ma è nel $\epsilon$-vicinato di un core point.
    3.  **Noise Point (Outlier)**: Un punto che non è né core né border.

    **Funzionamento:**
    * L'algoritmo inizia da un punto arbitrario non visitato.
    * Se è un core point, viene creato un nuovo cluster. L'algoritmo espande il cluster visitando ricorsivamente tutti i punti densamente connessi.
    * Se è un border point, viene assegnato a un cluster (se nel vicinato di un core point di quel cluster), ma non viene usato per espandere ulteriormente il cluster.
    * Se è un noise point, viene etichettato come tale.

    **Punti Chiave:**
    * **Non devi specificare il numero di cluster.**
    * Può trovare cluster di **forma arbitraria**.
    * **Robusto agli outlier**, che vengono esplicitamente identificati.
    * La performance dipende dalla scelta di `eps` e `MinPts`. Trovare buoni valori può richiedere sperimentazione.
    * Può faticare con cluster di **densità molto diverse**.
    """)

st.markdown("---")
st.caption("Applicazione didattica per visualizzare algoritmi di clustering. Creato con Streamlit.")
