import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap # Per colori più vari

# --- IMPORTA IL NUOVO GENERATORE ---
from db_generator import generate_advanced_customer_db

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="MarketPro: Individuazione Bodybuilder")

# --- Titolo e Introduzione ---
st.title("💪 MarketPro: Individuazione del Cluster Bodybuilder/Fitness")
st.markdown("""... Benvenuti nell'analisi avanzata di **MarketPro**! Qui useremo i dati aggregati delle tessere fedeltà per identificare clienti con specifici pattern di acquisto.
Il nostro obiettivo principale è individuare il **cluster dei "bodybuilder" o "appassionati di fitness"**, un segmento di nicchia ma di alto valore.
Capiremo come **K-Means** e **DBSCAN** possono aiutarci in questo.
""")

st.info("💡 **Obiettivo Specifico**: Trovare i clienti che acquistano molti prodotti proteici, carboidrati complessi e pochi cibi processati, tipici di uno stile di vita orientato al fitness.")

# --- Sidebar per Controlli Globali ---
with st.sidebar:
    st.header("⚙️ Impostazioni Simulazione Dati")
    n_samples = st.slider("Numero di Clienti Simulati", 100, 2000, 700, help="Quanti profili cliente generare.")
    random_state_data = st.slider("Seed per Generazione Dati", 0, 100, 22, help="Controlla la riproducibilità dei dati generati.") # Cambiato seed per più evidenza
    st.header("🔬 Scegli l'Algoritmo di Clustering")
    algoritmo_scelto = st.radio(
        "Quale algoritmo vuoi esplorare?",
        ("K-Means", "DBSCAN")
    )
    
    st.markdown("---")
    
    # Controlli dinamici per l'algoritmo scelto
    if algoritmo_scelto == "K-Means":
        st.subheader("Parametri K-Means")
        k_clusters = st.slider("Numero di Gruppi (K)", 2, 8, 5, help="Il numero di cluster che K-Means cercherà di formare. Prova K=5 per includere i bodybuilder.")
        kmeans_random_state = st.slider("Seed per K-Means", 0, 100, 1, help="Controlla l'inizializzazione dei centroidi. 0 per casuale.")
        st.write("*(K-Means divide i dati in K cluster compatti)*")
    
    elif algoritmo_scelto == "DBSCAN":
        st.subheader("Parametri DBSCAN")
        eps = st.slider("Epsilon (eps)", 0.1, 1.5, 0.4, step=0.05, help="Raggio massimo per considerare i punti come 'vicini'.")
        min_samples = st.slider("Min Samples", 2, 25, 8, help="Numero minimo di punti in un neighborhood per formare un cluster denso.")
        st.write("*(DBSCAN raggruppa punti in base alla densità e identifica il rumore)*")
        
    st.markdown("---")
    st.caption("App sviluppata per MarketPro - Analisi Dati Clienti")

# --- Generazione e Visualizzazione Dati (Prima del Clustering) ---
# ----------- QUI LA NUOVA CHIAMATA -----------
customer_df = generate_advanced_customer_db(n_samples, random_state_data)
# ---------------------------------------------

st.subheader("📊 Panoramica dei Dati Clienti Simulati")
st.write("Ecco un estratto del dataset simulato, focalizzato su feature rilevanti per il fitness:")
st.dataframe(customer_df.head())

st.write("Per visualizzare i cluster e individuare i bodybuilder, useremo le due feature più distintive:")
st.markdown("- **'Spesa_Proteine_Settimanale (€)'**: Alto valore atteso per i bodybuilder.")
st.markdown("- **'Spesa_JunkFood_Settimanale (€)'**: Basso valore atteso per i bodybuilder.")

# Scalatura dei dati per garantire che tutte le feature contribuiscano equamente
features_for_clustering = [
    'Spesa_Proteine_Settimanale', 
    'Spesa_Carbo_Complessi_Settimanale', 
    'Spesa_JunkFood_Settimanale',
    'Frequenza_Reparto_SportSalute', 
    'Eta', 
    'Varieta_Prodotti_Proteici'
]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_df[features_for_clustering])
customer_df_scaled = pd.DataFrame(X_scaled, columns=features_for_clustering)

# --- Pulsante per Eseguire il Clustering ---
st.markdown("---")
if st.button(f"🚀 Esegui {algoritmo_scelto} Clustering!"):
    st.subheader(f"✨ Risultati del Clustering con {algoritmo_scelto}")
    
    labels = []
    cluster_centers = None
    n_clusters_found = 0

    if algoritmo_scelto == "K-Means":
        kmeans = KMeans(n_clusters=k_clusters, random_state=kmeans_random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_) # Riporta i centroidi alla scala originale
        n_clusters_found = len(set(labels))

    elif algoritmo_scelto == "DBSCAN":
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        unique_labels = set(labels)
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Aggiungi le label al DataFrame originale per l'interpretazione
    customer_df['Cluster'] = labels

    # --- Visualizzazione dei Cluster ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Crea una colormap dinamica
    n_unique_labels = len(set(labels))
    # Escludi il colore per il rumore se presente
    if -1 in labels:
        cmap = plt.cm.get_cmap('viridis', n_unique_labels - 1)
        colors = [cmap(i) for i in range(n_unique_labels -1)]
        # Aggiungi un colore specifico per il rumore
        colors.append((0.5, 0.5, 0.5, 0.6)) # Grigio per rumore
    else:
        cmap = plt.cm.get_cmap('viridis', n_unique_labels)
        colors = [cmap(i) for i in range(n_unique_labels)]
    
    # Plotta i punti per ogni cluster
    for i, label in enumerate(sorted(set(labels))):
        mask = (labels == label)
        if label == -1: # Rumore
            ax.scatter(customer_df_scaled.loc[mask, 'Spesa_Proteine_Settimanale'], 
                       customer_df_scaled.loc[mask, 'Spesa_JunkFood_Settimanale'], 
                       c=[colors[len(colors)-1]], marker='x', s=60, label=f'Rumore/Outlier (-1)', alpha=0.7, edgecolors='none')
        else:
            ax.scatter(customer_df_scaled.loc[mask, 'Spesa_Proteine_Settimanale'], 
                       customer_df_scaled.loc[mask, 'Spesa_JunkFood_Settimanale'], 
                       c=[colors[i]], marker='o', s=100, label=f'Cluster {label}', alpha=0.8, edgecolors='w')


    if cluster_centers is not None:
        # Trova gli indici delle feature per il plot
        idx_prot = features_for_clustering.index('Spesa_Proteine_Settimanale')
        idx_junk = features_for_clustering.index('Spesa_JunkFood_Settimanale')
        ax.scatter(scaler.transform(cluster_centers)[:, idx_prot], # Scala i centroidi per il plot
                   scaler.transform(cluster_centers)[:, idx_junk],
                   marker='X', s=300, c='red', label='Centroidi', edgecolors='black')

    ax.set_title(f'Cluster Clienti con {algoritmo_scelto} (Spesa Proteine vs Spesa Junk Food)')
    ax.set_xlabel('Spesa Proteine Settimanale (Scalata)')
    ax.set_ylabel('Spesa Junk Food Settimanale (Scalata)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    st.markdown("---")
    col_metrics, col_details = st.columns(2)

    with col_metrics:
        st.subheader("Metrica e Riepilogo")
        if algoritmo_scelto == "K-Means":
            st.metric(label="Numero di Cluster Trovati", value=n_clusters_found)
            st.metric(label="Inerzia (WCSS)", value=f"{kmeans.inertia_:.2f}", 
                      help="Misura la compattezza dei cluster: minore è, meglio è.")
        elif algoritmo_scelto == "DBSCAN":
            st.metric(label="Numero di Cluster Trovati", value=n_clusters_found)
            n_noise = list(labels).count(-1)
            st.metric(label="Punti Identificati come Rumore (Outlier)", value=n_noise)
            if n_clusters_found == 0 and n_noise > 0:
                st.warning("⚠️ Nessun cluster trovato, solo rumore. Prova a regolare `eps` o `min_samples`.")
            elif n_clusters_found == 0 and n_noise == 0:
                 st.warning("⚠️ Nessun cluster o rumore trovato. Dati troppo densi o parametri non adatti.")


    with col_details:
        st.subheader("Analisi dei Cluster per MarketPro")
        st.write("Ecco il profilo medio per ogni cluster identificato (valori originali):")
        
        # Filtra i cluster validi e calcola le medie
        valid_clusters_df = customer_df[customer_df['Cluster'] != -1] if algoritmo_scelto == "DBSCAN" else customer_df
        
        if not valid_clusters_df.empty:
            cluster_summary = valid_clusters_df.groupby('Cluster')[features_for_clustering].mean().round(2)
            st.dataframe(cluster_summary)

            st.markdown("""
            **Interpretazione per MarketPro**:
            Analizza i cluster per identificare quello che mostra:
            * **Alta `Spesa_Proteine_Settimanale`**
            * **Alta `Spesa_Carbo_Complessi_Settimanale`**
            * **Bassa `Spesa_JunkFood_Settimanale`**
            * **Alta `Frequenza_Reparto_SportSalute`**
            * **Alta `Varietà_Prodotti_Proteici`**
            
            Questo sarà il tuo cluster di **Bodybuilder/Appassionati di Fitness**!
            """)
        else:
            st.warning("Nessun cluster valido trovato per mostrare il riepilogo.")

# --- Sezioni Didattiche ---
st.markdown("---")
st.header("📚 Approfondimenti sugli Algoritmi di Clustering")

# --- Spiegazione K-Means ---
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
    * Semplice e veloce, ideale per dataset di grandi dimensioni.
    * Facile da interpretare: i cluster hanno un centro ben definito.
    
    **Punti di Debolezza:**
    * Richiede di specificare il numero di cluster `K` in anticipo.
    * Assume cluster di forma sferica e dimensioni simili.
    * Sensibile agli outlier (punti anomali) che possono distorcere i centroidi.
    * Può dare risultati diversi a seconda dell'inizializzazione dei centroidi.
    """)
    
    st.subheader("Quando usare K-Means per MarketPro (Individuazione Bodybuilder)?")
    st.markdown("""
    K-Means può essere usato per MarketPro per trovare i bodybuilder se:
    * **Ti aspetti che il gruppo dei bodybuilder sia uno dei `K` cluster principali**, ben separato e relativamente compatto in termini di abitudini di acquisto.
    * **Hai già un'idea di quanti segmenti principali** vuoi identificare nella tua clientela, incluso quello dei bodybuilder (es. 5 segmenti: bodybuilder, famiglie, risparmiatori, salutisti generici, amanti del junk food).
    * Non ti preoccupa che i **clienti "di transizione" o "non puri"** possano essere assegnati a un cluster vicino.

    **Sfida in questo scenario:** Se i bodybuilder sono una nicchia molto piccola e densa, K-Means potrebbe "diluirli" in cluster più grandi o non riuscire a isolarli perfettamente se K non è scelto con cura.
    """)

# --- Spiegazione DBSCAN ---
with st.expander("🔬 DBSCAN: Scoprire Gruppi di Densità e Rilevare Anomali"):
    st.subheader("Cos'è DBSCAN?")
    st.markdown("""
    **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) è un algoritmo di clustering basato sulla **densità**. A differenza di K-Means, non richiede di specificare il numero di cluster in anticipo e può identificare cluster di forme arbitrarie, oltre a classificare i punti di rumore (outlier).
    
    **Come funziona (concetti chiave):**
    DBSCAN si basa su due parametri:
    * **`Epsilon (eps)`**: Il raggio massimo del vicinato da considerare per un punto. Se un punto ha almeno `min_samples` punti nel suo vicinato (incluso se stesso), viene considerato un "core point" e può dare origine a un cluster.
    * **`min_samples`**: Il numero minimo di punti per formare un cluster denso.
    """)
