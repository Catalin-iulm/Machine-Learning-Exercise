import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="Visualizzatore Algoritmi di Clustering per Marketing")

# --- Titolo e Introduzione ---
st.title("🔬 Visualizzatore Interattivo di Algoritmi di Clustering per il Marketing")
st.markdown("""
Questa applicazione ti permette di esplorare il funzionamento degli algoritmi di clustering **K-Means** e **DBSCAN**
su diversi tipi di dataset sintetici, simulando scenari comuni nel marketing. Modifica i parametri del dataset e dell'algoritmo
per vedere come cambiano i risultati e quali insight puoi ottenere sui tuoi "clienti"!
""")

# --- Funzione per Generare Dati Sintetici ---
def generate_data(dataset_type, n_samples, noise, random_state, n_blobs_centers=3, blob_std=1.0):
    X = np.array([[]]) # Initialize X
    y_true = None # True labels, useful for some datasets but not directly used by clustering

    if dataset_type == "Clienti con Abitudini Chiare (Segmenti Distinti)":
        X, y_true = make_blobs(n_samples=n_samples, centers=n_blobs_centers, cluster_std=0.6,
                               random_state=random_state)
    elif dataset_type == "Clienti con Varianza Alta (Sovrapposizione)":
        X, y_true = make_blobs(n_samples=n_samples, centers=n_blobs_centers, cluster_std=blob_std if blob_std > 0 else 1.5,
                               random_state=random_state)
    elif dataset_type == "Clienti Fedeli vs. Nuovi (Comportamento su Piattaforme)":
        X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == "Livelli di Spesa/Engagement (Cerchi Concentrici)":
        X, y_true = make_circles(n_samples=n_samples, factor=0.5, noise=noise, random_state=random_state)
    elif dataset_type == "Clienti con Pattern Anisotropi (Comportamento Complesso)": # Challenging for K-Means default
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso, _ = make_blobs(n_samples=n_samples, centers=n_blobs_centers, random_state=random_state, cluster_std=0.7)
        X = np.dot(X_aniso, transformation)
        y_true = None # True labels are harder to map after transformation for simple demo
    elif dataset_type == "Dati di Mercato Senza Struttura Evidente (Random)":
        X = np.random.rand(n_samples, 2) * 10 # Spread out points
        y_true = None
        
    # Standard Scaler - Essential for many algorithms, especially K-Means for distance calculations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_true

# --- Mapping per Nomi Features Marketing ---
feature_names_mapping = {
    "Clienti con Abitudini Chiare (Segmenti Distinti)": {"x": "Frequenza Acquisti (Scalata)", "y": "Valore Medio Carrello (Scalata)"},
    "Clienti con Varianza Alta (Sovrapposizione)": {"x": "Interazioni Settimanali (Scalata)", "y": "Dimensione Media Carrello (Scalata)"},
    "Clienti Fedeli vs. Nuovi (Comportamento su Piattaforme)": {"x": "Tempo Totale su App/Sito (Scalata)", "y": "Numero Accessi Mensili (Scalata)"},
    "Livelli di Spesa/Engagement (Cerchi Concentrici)": {"x": "Spesa Totale Annua (Scalata)", "y": "Punti Fedeltà Guadagnati (Scalata)"},
    "Clienti con Pattern Anisotropi (Comportamento Complesso)": {"x": "Interazioni Social Media (Scalata)", "y": "Recensioni Prodotti Scritte (Scalata)"},
    "Dati di Mercato Senza Struttura Evidente (Random)": {"x": "Metrica Demografica A (Scalata)", "y": "Metrica Comportamentale B (Scalata)"}
}


# --- Sidebar per Controlli Globali ---
with st.sidebar:
    st.header("⚙️ Configurazione Esperimento")

    st.subheader("1. Scegli lo Scenario di Marketing (Dataset)")
    dataset_options = [
        "Clienti con Abitudini Chiare (Segmenti Distinti)",
        "Clienti con Varianza Alta (Sovrapposizione)",
        "Clienti Fedeli vs. Nuovi (Comportamento su Piattaforme)",
        "Livelli di Spesa/Engagement (Cerchi Concentrici)",
        "Clienti con Pattern Anisotropi (Comportamento Complesso)",
        "Dati di Mercato Senza Struttura Evidente (Random)"
    ]
    dataset_type = st.selectbox("Tipo di Dataset:", dataset_options)

    n_samples_data = st.slider("Numero di 'Clienti' (Punti Dati)", 100, 1500, 300, step=50)
    
    # Dataset specific parameters
    if "Clienti" in dataset_type and "Abitudini Chiare" in dataset_type or "Varianza Alta" in dataset_type:
        n_centers_blobs = st.slider("Numero di Segmenti/Centri Desiderati", 2, 5, 3)
        if "Varianza Alta" in dataset_type:
             blob_std_param = st.slider("Deviazione Standard dei Segmenti (Dispersion)", 0.5, 3.0, 1.8, step=0.1)
        else:
            blob_std_param = 0.6 # Fixed for well-separated
    elif dataset_type in ["Clienti Fedeli vs. Nuovi (Comportamento su Piattaforme)", "Livelli di Spesa/Engagement (Cerchi Concentrici)"]:
        noise_level = st.slider("Livello di Rumore nel Comportamento", 0.01, 0.3, 0.05, step=0.01)
    else: # For Anisotropic, Random
        noise_level = 0.05 # Dummy, not directly used by all generators in the same way
        n_centers_blobs = 3 # Used by anisotropic
        blob_std_param = 1.0 # Dummy

    random_state_ds = st.slider("Seed Generazione Dati (per riproducibilità)", 0, 100, 42)
    st.markdown("---")

    st.subheader("2. Scegli l'Algoritmo di Clustering")
    algoritmo_scelto = st.radio(
        "Algoritmo:", ("K-Means", "DBSCAN"), horizontal=True
    )
    st.markdown("---")

    st.subheader(f"3. Parametri {algoritmo_scelto}")
    if algoritmo_scelto == "K-Means":
        k_clusters_param = st.slider("Numero di Cluster (K) da Trovare", 1, 10, 3,
                                     help="Quanti segmenti di clienti l'algoritmo K-Means cercherà.")
        kmeans_random_state_param = st.slider("Seed K-Means (per inizializzazione)", 0, 100, 1,
                                             help="Controlla l'inizializzazione dei centroidi per la riproducibilità. Cambialo per vedere diverse configurazioni iniziali.")
    elif algoritmo_scelto == "DBSCAN":
        eps_param = st.slider("Epsilon (eps) - Raggio di Vicinato", 0.05, 2.0, 0.2, step=0.01, # Adjusted range for scaled data
                             help="Distanza massima per considerare due 'clienti' vicini.")
        min_samples_param = st.slider("Min Samples - Densità Minima", 1, 30, 5,
                                     help="Numero minimo di 'clienti' in un vicinato per formare un segmento denso. I punti sotto questa soglia potrebbero essere considerati rumore.")

# --- Generazione Dati ---
X_data, y_true_data = generate_data(dataset_type, n_samples_data,
                                    noise_level if 'noise_level' in locals() else 0.05,
                                    random_state_ds,
                                    n_centers_blobs if 'n_centers_blobs' in locals() else 3,
                                    blob_std_param if 'blob_std_param' in locals() else 1.0)

# --- Contesto Marketing per il Dataset Scelto ---
st.markdown("---")
st.subheader("💡 Contesto Marketing per lo Scenario Attuale:")

if dataset_type == "Clienti con Abitudini Chiare (Segmenti Distinti)":
    st.info("""
    Immagina questi punti come clienti di un'azienda. Gli algoritmi di clustering possono identificare **segmenti di clienti con abitudini di acquisto molto distinte**, basati sulla **frequenza dei loro acquisti** e il **valore medio del loro carrello**. Questo ti permette di creare campagne marketing altamente personalizzate: ad esempio, offerte esclusive per clienti ad alto valore, o incentivi per riattivare acquirenti occasionali.
    """)
elif dataset_type == "Clienti con Varianza Alta (Sovrapposizione)":
    st.info("""
    Qui i tuoi clienti mostrano comportamenti più eterogenei e i segmenti potrebbero **sovrapporsi**. L'analisi qui è cruciale per capire le sfumature e, magari, identificare clienti che potrebbero essere persuasi a passare da un segmento all'altro con le giuste sollecitazioni di marketing. Potresti vedere clienti che oscillano tra essere "grandi compratori occasionali" e "piccoli compratori frequenti".
    """)
elif dataset_type == "Clienti Fedeli vs. Nuovi (Comportamento su Piattaforme)":
    st.info("""
    Questo scenario simula due gruppi di clienti: **quelli più fedeli** che passano molto tempo e accedono spesso alla tua piattaforma, e **quelli più nuovi o meno impegnati** che hanno un'interazione più superficiale. Il clustering può aiutarti a differenziare le strategie di engagement: programmi fedeltà per i primi e campagne di onboarding o riattivazione per i secondi.
    """)
elif dataset_type == "Livelli di Spesa/Engagement (Cerchi Concentrici)":
    st.info("""
    Immagina tre o più "anelli" di clienti basati sulla loro **spesa totale annua** e i **punti fedeltà guadagnati**. I cluster qui indicano diversi livelli di engagement o valore per l'azienda: clienti occasionali, clienti di medio valore e clienti VIP. Questo è perfetto per definire livelli di servizio, promozioni esclusive o programmi di ricompensa differenziati.
    """)
elif dataset_type == "Clienti con Pattern Anisotropi (Comportamento Complesso)":
    st.info("""
    Questo dataset rappresenta clienti con **pattern di comportamento più complessi e meno "sferici"**, come ad esempio l'interazione sui social media e il numero di recensioni lasciate. Potrebbero esserci gruppi di "influencer passivi" o "recensori di nicchia". K-Means potrebbe faticare qui, mentre DBSCAN potrebbe rivelare forme di cluster più interessanti, utili per strategie di content marketing o influencer marketing.
    """)
elif dataset_type == "Dati di Mercato Senza Struttura Evidente (Random)":
    st.info("""
    Questo scenario indica che i tuoi dati attuali potrebbero **non avere segmenti distinti ben definiti** in base alle metriche scelte. In un contesto marketing, questo suggerisce che potresti dover raccogliere più dati, o esplorare diverse combinazioni di metriche, per trovare pattern significativi e segmenti azionabili.
    """)
st.markdown("---")

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
    st.subheader("Grafico dei Segmenti di Clienti")
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
            color_map_for_plot[lbl_plot] = (0.5, 0.5, 0.5, 0.7) # Grigio per rumore (outlier)
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
        plot_legend_label = f'Clienti Rumore/Outlier (-1)' if label_val_plot == -1 else f'Segmento {label_val_plot}'

        ax_cluster.scatter(X_data[mask_plot, 0], X_data[mask_plot, 1],
                           facecolor=current_color_plot, marker=marker_style_plot, s=point_size_plot,
                           label=plot_legend_label, alpha=0.8,
                           edgecolor='k' if label_val_plot !=-1 else 'none', linewidth=0.5 if label_val_plot !=-1 else 0)

    if algoritmo_scelto == "K-Means" and cluster_centers_coords is not None:
        ax_cluster.scatter(cluster_centers_coords[:, 0], cluster_centers_coords[:, 1],
                           marker='P', s=250, facecolor='red', label='Centroide Segmento',
                           edgecolor='black', linewidth=1.5, zorder=10)

    # Dynamic Axis Labels
    current_x_label = feature_names_mapping.get(dataset_type, {}).get("x", "Feature 1 (Scalata)")
    current_y_label = feature_names_mapping.get(dataset_type, {}).get("y", "Feature 2 (Scalata)")

    ax_cluster.set_title(f'Segmentazione Clienti: {dataset_type} con {algoritmo_scelto}')
    ax_cluster.set_xlabel(current_x_label)
    ax_cluster.set_ylabel(current_y_label)
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
    * Per identificare **segmenti di clienti con forme e dimensioni complesse** (es. clienti che seguono un percorso di acquisto a "U" o a "S").
    * Per individuare facilmente i **clienti outlier** (es. acquirenti fraudolenti, clienti con comportamenti estremamente inusuali) che necessitano di attenzione speciale o esclusione da certe campagne.
    * Utile per segmentare dati geolocalizzati o pattern di navigazione web.

    **Limiti per il Marketing:**
    * La performance dipende molto dalla scelta di `eps` e `MinPts`. Trovare i valori giusti può richiedere sperimentazione e conoscenza del dominio.
    * Può faticare con segmenti di **densità molto diverse** (es. un segmento di clienti ad alta frequenza molto compatto e un segmento di clienti occasionali molto sparsi).
    """)

st.markdown("---")
st.caption("Applicazione didattica per visualizzare algoritmi di clustering per scopi di marketing. Creato con Streamlit.")
