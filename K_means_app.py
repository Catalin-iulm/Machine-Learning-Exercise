import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="MarketPro: Clustering Clienti")

# --- Funzione per Generare Dati Simulati (Cashed per performance) ---
@st.cache_data
def generate_customer_data(n_samples, random_state_data):
    np.random.seed(random_state_data)

    # Simulo 5 segmenti principali di clienti
    # 1. Giovani/Single (Bassa spesa, Alta frequenza, Pochi freschi, Vicini)
    # 2. Famiglie (Alta spesa, Frequenza media, Freschi medi, Distanza media)
    # 3. Anziani (Spesa media, Bassa frequenza, Freschi bassi, Vicini)
    # 4. Amanti del Bio/Salute (Spesa alta, Frequenza media, Molti freschi, Distanza media)
    # 5. Acquirenti Occasionali/Offerte (Spesa molto bassa, Frequenza molto bassa, Pochi freschi, Distanza varia)

    data_points = []

    # Segmento 1: Giovani/Single (n_samples * 0.2)
    for _ in range(int(n_samples * 0.2)):
        data_points.append([
            np.random.normal(30, 5),    # Spesa Media Settimanale
            np.random.normal(8, 2),     # Frequenza Acquisti (volte/mese)
            np.random.normal(25, 3),    # Età
            np.random.normal(30, 10),   # Prodotti_Freschi_Ratio
            np.random.normal(2, 1)      # Distanza_Dal_Negozio
        ])
    
    # Segmento 2: Famiglie (n_samples * 0.3)
    for _ in range(int(n_samples * 0.3)):
        data_points.append([
            np.random.normal(120, 20),  # Spesa Media Settimanale
            np.random.normal(4, 1),     # Frequenza Acquisti
            np.random.normal(40, 5),    # Età
            np.random.normal(60, 10),   # Prodotti_Freschi_Ratio
            np.random.normal(5, 2)      # Distanza_Dal_Negozio
        ])

    # Segmento 3: Anziani (n_samples * 0.15)
    for _ in range(int(n_samples * 0.15)):
        data_points.append([
            np.random.normal(50, 10),   # Spesa Media Settimanale
            np.random.normal(2, 1),     # Frequenza Acquisti
            np.random.normal(65, 5),    # Età
            np.random.normal(40, 10),   # Prodotti_Freschi_Ratio
            np.random.normal(1, 0.5)    # Distanza_Dal_Negozio
        ])

    # Segmento 4: Amanti del Bio/Salute (n_samples * 0.15)
    for _ in range(int(n_samples * 0.15)):
        data_points.append([
            np.random.normal(90, 15),   # Spesa Media Settimanale
            np.random.normal(6, 1.5),   # Frequenza Acquisti
            np.random.normal(35, 7),    # Età
            np.random.normal(85, 5),    # Prodotti_Freschi_Ratio
            np.random.normal(4, 1.5)    # Distanza_Dal_Negozio
        ])

    # Segmento 5: Acquirenti Occasionali/Offerte (n_samples * 0.2)
    for _ in range(int(n_samples * 0.2)):
        data_points.append([
            np.random.normal(15, 7),    # Spesa Media Settimanale
            np.random.normal(1, 0.5),   # Frequenza Acquisti
            np.random.normal(30, 10),   # Età
            np.random.normal(20, 10),   # Prodotti_Freschi_Ratio
            np.random.normal(8, 3)      # Distanza_Dal_Negozio
        ])
        
    df = pd.DataFrame(data_points, columns=[
        'Spesa Media Settimanale (€)',
        'Frequenza Acquisti (volte/mese)',
        'Età (anni)',
        'Prodotti_Freschi_Ratio (%)',
        'Distanza_Dal_Negozio (km)'
    ])

    # Aggiungi un po' di rumore casuale generale per rendere i cluster meno perfetti
    df = df + np.random.normal(0, 3, df.shape)
    
    # Assicurati che non ci siano valori negativi non sensati
    df['Spesa Media Settimanale (€)'] = df['Spesa Media Settimanale (€)'].apply(lambda x: max(1, x))
    df['Frequenza Acquisti (volte/mese)'] = df['Frequenza Acquisti (volte/mese)'].apply(lambda x: max(0.5, x))
    df['Età (anni)'] = df['Età (anni)'].apply(lambda x: max(18, x))
    df['Prodotti_Freschi_Ratio (%)'] = df['Prodotti_Freschi_Ratio (%)'].apply(lambda x: min(100, max(0, x)))
    df['Distanza_Dal_Negozio (km)'] = df['Distanza_Dal_Negozio (km)'].apply(lambda x: max(0.1, x))

    return df

# --- Titolo e Introduzione ---
st.title("🛒 MarketPro: Segmentazione Clienti con Machine Learning")
st.markdown("""
Benvenuto nella nostra applicazione interattiva per la segmentazione clienti di **MarketPro**!
Utilizzeremo due algoritmi di clustering, **K-Means** e **DBSCAN**, per identificare gruppi omogenei di clienti basati sui loro comportamenti di acquisto.
Comprendere questi segmenti è fondamentale per strategie di marketing personalizzate e per ottimizzare l'offerta di prodotti.
""")

st.info("💡 **Obiettivo**: Identificare profili di clienti per Marketing Mirato, Gestione Scorte e Ottimizzazione del Punto Vendita.")

# --- Sidebar per Controlli Globali ---
with st.sidebar:
    st.header("⚙️ Impostazioni Simulazione Dati")
    n_samples = st.slider("Numero di Clienti Simulati", 100, 2000, 500, help="Quanti profili cliente generare.")
    random_state_data = st.slider("Seed per Generazione Dati", 0, 100, 42, help="Controlla la riproducibilità dei dati generati.")

    st.header("🔬 Scegli l'Algoritmo di Clustering")
    algoritmo_scelto = st.radio(
        "Quale algoritmo vuoi esplorare?",
        ("K-Means", "DBSCAN")
    )
    
    st.markdown("---")
    
    # Controlli dinamici per l'algoritmo scelto
    if algoritmo_scelto == "K-Means":
        st.subheader("Parametri K-Means")
        k_clusters = st.slider("Numero di Gruppi (K)", 2, 8, 4, help="Il numero di cluster che K-Means cercherà di formare.")
        kmeans_random_state = st.slider("Seed per K-Means", 0, 100, 0, help="Controlla l'inizializzazione dei centroidi. 0 per casuale.")
        st.write("*(K-Means divide i dati in K cluster compatti)*")
    
    elif algoritmo_scelto == "DBSCAN":
        st.subheader("Parametri DBSCAN")
        eps = st.slider("Epsilon (eps)", 0.1, 1.0, 0.3, step=0.05, help="Raggio massimo per considerare i punti come 'vicini'.")
        min_samples = st.slider("Min Samples", 2, 20, 5, help="Numero minimo di punti in un neighborhood per formare un cluster denso.")
        st.write("*(DBSCAN raggruppa punti in base alla densità e identifica il rumore)*")
        
    st.markdown("---")
    st.caption("App sviluppata per MarketPro - Analisi Dati Clienti")

# --- Generazione e Visualizzazione Dati (Prima del Clustering) ---
customer_df = generate_customer_data(n_samples, random_state_data)

st.subheader("📊 Panoramica dei Dati Clienti Simulati")
st.write("Ecco un estratto del dataset simulato che useremo per il clustering:")
st.dataframe(customer_df.head())

st.write("Per visualizzare i cluster, useremo le due feature più rappresentative per MarketPro:")
st.markdown("- **'Spesa Media Settimanale (€)'**: Indica il valore del cliente.")
st.markdown("- **'Frequenza Acquisti (volte/mese)'**: Indica l'engagement del cliente.")

# Scalatura dei dati per garantire che tutte le feature contribuiscano equamente
# Scaliamo solo le feature che useremo per il clustering
features_for_clustering = ['Spesa Media Settimanale (€)', 'Frequenza Acquisti (volte/mese)', 
                          'Età (anni)', 'Prodotti_Freschi_Ratio (%)', 'Distanza_Dal_Negozio (km)']
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
        
        # DBSCAN: -1 è il rumore
        unique_labels = set(labels)
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Aggiungi le label al DataFrame originale per l'interpretazione
    customer_df['Cluster'] = labels

    # --- Visualizzazione dei Cluster ---
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.get_cmap('viridis', n_clusters_found if n_clusters_found > 0 else 1) # Assicura un cmap valido anche con 0 cluster
    
    if algoritmo_scelto == "DBSCAN" and -1 in unique_labels:
        # Se c'è rumore in DBSCAN, plottalo per primo in grigio/nero
        noise_mask = (labels == -1)
        ax.scatter(customer_df_scaled.loc[noise_mask, 'Spesa Media Settimanale (€)'], 
                   customer_df_scaled.loc[noise_mask, 'Frequenza Acquisti (volte/mese)'], 
                   c='gray', marker='x', s=50, label='Rumore/Outlier (-1)', alpha=0.6)
        
        # Plotta i cluster veri
        for i in range(n_clusters_found):
            cluster_mask = (labels == i)
            ax.scatter(customer_df_scaled.loc[cluster_mask, 'Spesa Media Settimanale (€)'], 
                       customer_df_scaled.loc[cluster_mask, 'Frequenza Acquisti (volte/mese)'], 
                       c=[cmap(i)], label=f'Cluster {i}', s=100, alpha=0.8, edgecolors='w')
    else:
        # Per K-Means o DBSCAN senza rumore
        ax.scatter(customer_df_scaled['Spesa Media Settimanale (€)'], 
                   customer_df_scaled['Frequenza Acquisti (volte/mese)'], 
                   c=labels, cmap=cmap, s=100, alpha=0.8, edgecolors='w')

    if cluster_centers is not None:
        ax.scatter(cluster_centers[:, features_for_clustering.index('Spesa Media Settimanale (€)')],
                   cluster_centers[:, features_for_clustering.index('Frequenza Acquisti (volte/mese)')],
                   marker='X', s=300, c='red', label='Centroidi', edgecolors='black')

    ax.set_title(f'Cluster Clienti con {algoritmo_scelto} (Spesa vs Frequenza)')
    ax.set_xlabel('Spesa Media Settimanale (Scalata)')
    ax.set_ylabel('Frequenza Acquisti (Scalata)')
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
            Ogni riga rappresenta un segmento di clienti. Analizzando i valori medi per 'Spesa Media Settimanale', 'Frequenza Acquisti', 'Età', ecc.,
            il team marketing di MarketPro può definire "persona" per ogni cluster e creare strategie ad-hoc:
            
            * **Cluster con alta Spesa/Frequenza**: Programmi fedeltà premium, offerte personalizzate su prodotti di nicchia.
            * **Cluster con bassa Spesa/Frequenza (ma non rumore)**: Campagne di riattivazione, buoni sconto al primo acquisto, focus su prodotti base.
            * **Cluster con alta Prodotti_Freschi_Ratio**: Promozioni su frutta, verdura, biologico; eventi di show-cooking.
            * **Punti di Rumore (DBSCAN)**: Potrebbero essere clienti con comportamenti insoliti (es. acquisti sporadici ma molto grandi) o potenziali anomalie da investigare.
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
    
    st.subheader("Quando usare K-Means per MarketPro (Marketing)?")
    st.markdown("""
    K-Means è la scelta ideale per MarketPro quando:
    * **Hai già in mente un numero specifico di segmenti di clienti** che vuoi creare (es. "Voglio 3 tipi di clienti: High-Value, Medium-Value, Low-Value").
    * **I tuoi segmenti di clienti sono ragionevolmente ben separati e compatti** in termini di comportamento (es. gruppi di clienti con spesa media simile, età simile, ecc.).
    * **Non ti aspetti molti outlier** o clienti con comportamenti estremamente anomali.
    * Vuoi un **approccio diretto e interpretabile** per definire "persona" di marketing chiare.
    
    **Esempio per MarketPro:** Definire 4-5 segmenti per campagne email specifiche: "Amanti Bio", "Famiglie Convenienza", "Giovani Spendaccioni", ecc.
    """)

# --- Spiegazione DBSCAN ---
with st.expander("🔬 DBSCAN: Scoprire Gruppi di Densità e Rilevare Anomali"):
    st.subheader("Cos'è DBSCAN?")
    st.markdown("""
    **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) è un algoritmo di clustering basato sulla **densità**. A differenza di K-Means, non richiede di specificare il numero di cluster in anticipo e può identificare cluster di forme arbitrarie, oltre a classificare i punti di rumore (outlier).
    
    **Come funziona (concetti chiave):**
    DBSCAN si basa su due parametri:
    * **`Epsilon (eps)`**: Il raggio massimo del vicinato da considerare per un punto. Se un punto ha almeno `min_samples` punti entro questa distanza, è considerato un punto denso.
    * **`Min Samples (min_samples)`**: Il numero minimo di punti necessari all'interno del raggio `eps` di un punto per formare un'area densa.
    
    I punti vengono classificati in tre tipi:
    * **Punto Core (Core Point)**: Un punto che ha almeno `min_samples` altri punti (incluso se stesso) entro la distanza `eps`. Questi sono il "cuore" dei cluster.
    * **Punto di Bordo (Border Point)**: Un punto che non è un punto core, ma si trova all'interno del raggio `eps` di un punto core. Fa parte di un cluster ma è alla sua periferia.
    * **Punto di Rumore (Noise Point)**: Un punto che non è né un punto core né un punto di bordo. Questi sono considerati outlier e non appartengono a nessun cluster.
    
    **Punti di Forza:**
    * Non richiede di specificare il numero di cluster in anticipo.
    * Può trovare cluster di forme arbitrarie (non solo sferiche).
    * Identifica e gestisce automaticamente i punti di rumore/outlier.
    
    **Punti di Debolezza:**
    * Sensibile ai parametri `eps` e `min_samples`; la loro scelta può essere complessa.
    * Non funziona bene con cluster di densità molto diverse.
    * Più lento di K-Means su dataset molto grandi.
    """)
    
    st.subheader("Quando usare DBSCAN per MarketPro (Marketing)?")
    st.markdown("""
    DBSCAN è la scelta ideale per MarketPro quando:
    * **Non hai idea di quanti segmenti di clienti esistano** o se i segmenti hanno forme non sferiche (es. un gruppo di clienti molto fedeli che visitano frequentemente ma spendono poco e un altro che visita raramente ma spende molto).
    * **Vuoi identificare i clienti anomali o "rumore"** (outlier). Questo è cruciale per la rilevazione di frodi, l'identificazione di comportamenti di acquisto insoliti o per capire chi sono i clienti "one-shot" o i "non-clienti" reali.
    * **La densità di clienti varia molto** in diverse aree del tuo spazio di features (anche se questo è un limite di DBSCAN, la sua capacità di adattarsi a densità relative è un vantaggio rispetto a K-Means).
    
    **Esempio per MarketPro:** Trovare gruppi di clienti "insospettabili" con abitudini di acquisto uniche, o identificare singoli acquisti estremamente grandi o frequenti che potrebbero indicare comportamenti anomali (es. acquisti aziendali mascherati da privati, o persino frodi con carte fedeltà).
    """)

# --- Conclusione e Guida alla Scelta ---
st.markdown("---")
st.header("🎯 K-Means vs DBSCAN: Quale Scegliere per MarketPro?")
st.markdown("""
La scelta tra K-Means e DBSCAN per MarketPro dipende dagli obiettivi specifici della tua analisi di marketing:

* **Scegli K-Means se:**
    * Hai bisogno di un numero fisso e predefinito di segmenti clienti.
    * I tuoi segmenti sono generalmente compatti e sferici.
    * Vuoi un'interpretazione semplice e diretta dei centroidi come "prototipi" di cliente.
    * Il rumore nei dati non è una preoccupazione primaria o viene gestito a parte.

* **Scegli DBSCAN se:**
    * Non sai quanti segmenti esistono e vuoi che l'algoritmo li scopra.
    * Sospetti che i tuoi segmenti clienti abbiano forme irregolari o non sferiche.
    * È fondamentale identificare e separare i clienti "anomali" o "rumore" dal resto della clientela.
    * Le densità dei gruppi di clienti possono variare e vuoi che l'algoritmo si adatti a questo.

Entrambi gli algoritmi sono strumenti potenti per MarketPro per trasformare i dati grezzi in insight azionabili e strategie di marketing più efficaci!
""")
