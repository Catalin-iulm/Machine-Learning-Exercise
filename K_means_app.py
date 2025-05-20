import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap # Per colori più vari

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="MarketPro: Individuazione Bodybuilder")

# --- Funzione per Generare Dati Simulati (Cashed per performance) ---
@st.cache_data
def generate_customer_data_for_bodybuilders(n_samples, random_state_data):
    np.random.seed(random_state_data)

    data_points = []

    # Simulo 5 segmenti principali di clienti, inclusi i bodybuilder
    # 1. Bodybuilder/Fitness Enthusiasts (Niche, ~10% of customers)
    # 2. Standard Families (Largest segment, ~35%)
    # 3. Budget Shoppers (Cost-conscious, ~20%)
    # 4. Health-Conscious (General, not extreme fitness, ~20%)
    # 5. Convenience/Junk Food Consumers (~15%)

    # Features:
    # - Spesa_Proteine_Settimanale (€)
    # - Spesa_Carbo_Complessi_Settimanale (€)
    # - Spesa_JunkFood_Settimanale (€)
    # - Frequenza_Reparto_SportSalute (volte/mese)
    # - Età (anni)
    # - Varietà_Prodotti_Proteici (count)

    for i in range(n_samples):
        segment_choice = np.random.choice([
            "Bodybuilder", "Standard Family", "Budget Shopper", 
            "Health-Conscious", "Junk Food Consumer"
        ], p=[0.10, 0.35, 0.20, 0.20, 0.15])

        if segment_choice == "Bodybuilder":
            spesa_prot = np.random.normal(70, 15)  # Molto alta
            spesa_carbo = np.random.normal(50, 10) # Alta
            spesa_junk = np.random.normal(5, 3)    # Molto bassa
            freq_sport = np.random.normal(4, 1)    # Alta
            eta = np.random.normal(30, 5)          # Giovane/media
            varieta_prot = np.random.normal(15, 3) # Alta varietà
        elif segment_choice == "Standard Family":
            spesa_prot = np.random.normal(40, 10)
            spesa_carbo = np.random.normal(30, 8)
            spesa_junk = np.random.normal(20, 7)
            freq_sport = np.random.normal(1, 0.5)
            eta = np.random.normal(45, 7)
            varieta_prot = np.random.normal(8, 2)
        elif segment_choice == "Budget Shopper":
            spesa_prot = np.random.normal(20, 5)
            spesa_carbo = np.random.normal(15, 5)
            spesa_junk = np.random.normal(10, 5)
            freq_sport = np.random.normal(0.5, 0.3)
            eta = np.random.normal(35, 10)
            varieta_prot = np.random.normal(5, 1)
        elif segment_choice == "Health-Conscious":
            spesa_prot = np.random.normal(50, 10)
            spesa_carbo = np.random.normal(40, 10)
            spesa_junk = np.random.normal(8, 4)
            freq_sport = np.random.normal(2, 0.8)
            eta = np.random.normal(38, 8)
            varieta_prot = np.random.normal(12, 3)
        elif segment_choice == "Junk Food Consumer":
            spesa_prot = np.random.normal(15, 5)
            spesa_carbo = np.random.normal(10, 5)
            spesa_junk = np.random.normal(40, 10) # Molto alta
            freq_sport = np.random.normal(0.2, 0.1)
            eta = np.random.normal(28, 5)
            varieta_prot = np.random.normal(3, 1)
        
        data_points.append([
            max(0, spesa_prot), 
            max(0, spesa_carbo), 
            max(0, spesa_junk), 
            max(0, freq_sport), 
            max(18, eta), 
            max(1, varieta_prot)
        ])
    
    df = pd.DataFrame(data_points, columns=[
        'Spesa_Proteine_Settimanale (€)',
        'Spesa_Carbo_Complessi_Settimanale (€)',
        'Spesa_JunkFood_Settimanale (€)',
        'Frequenza_Reparto_SportSalute (volte/mese)',
        'Età (anni)',
        'Varietà_Prodotti_Proteici (count)'
    ])
    
    return df

# --- Titolo e Introduzione ---
st.title("💪 MarketPro: Individuazione del Cluster Bodybuilder/Fitness")
st.markdown("""
Benvenuti nell'analisi avanzata di **MarketPro**! Qui useremo i dati aggregati delle tessere fedeltà per identificare clienti con specifici pattern di acquisto.
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
customer_df = generate_customer_data_for_bodybuilders(n_samples, random_state_data)

st.subheader("📊 Panoramica dei Dati Clienti Simulati")
st.write("Ecco un estratto del dataset simulato, focalizzato su feature rilevanti per il fitness:")
st.dataframe(customer_df.head())

st.write("Per visualizzare i cluster e individuare i bodybuilder, useremo le due feature più distintive:")
st.markdown("- **'Spesa_Proteine_Settimanale (€)'**: Alto valore atteso per i bodybuilder.")
st.markdown("- **'Spesa_JunkFood_Settimanale (€)'**: Basso valore atteso per i bodybuilder.")

# Scalatura dei dati per garantire che tutte le feature contribuiscano equamente
features_for_clustering = customer_df.columns.tolist() # Tutte le colonne generate
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
            ax.scatter(customer_df_scaled.loc[mask, 'Spesa_Proteine_Settimanale (€)'], 
                       customer_df_scaled.loc[mask, 'Spesa_JunkFood_Settimanale (€)'], 
                       c=[colors[len(colors)-1]], marker='x', s=60, label=f'Rumore/Outlier (-1)', alpha=0.7, edgecolors='none')
        else:
            ax.scatter(customer_df_scaled.loc[mask, 'Spesa_Proteine_Settimanale (€)'], 
                       customer_df_scaled.loc[mask, 'Spesa_JunkFood_Settimanale (€)'], 
                       c=[colors[i]], marker='o', s=100, label=f'Cluster {label}', alpha=0.8, edgecolors='w')


    if cluster_centers is not None:
        # Trova gli indici delle feature per il plot
        idx_prot = features_for_clustering.index('Spesa_Proteine_Settimanale (€)')
        idx_junk = features_for_clustering.index('Spesa_JunkFood_Settimanale (€)')
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
    * **`Epsilon (eps)`**: Il raggio massimo del vicinato da considerare per un punto. Se un punto ha almeno `min_samples` punti entro questa distanza, è considerato un punto denso.
    * **`Min Samples (min_samples)`**: Il numero minimo di punti necessari all'interno del raggio `eps` di un punto per formare un'area densa.
    
    I punti vengono classificati in tre tipi:
    * **Punto Core (Core Point)**: Un punto che ha almeno `min_samples` altri punti (incluso se stesso) entro la distanza `eps`. Questi sono il "cuore" dei cluster.
    * **Punto di Bordo (Border Point)**: Un punto che non è un punto core, ma si trova all'interno del raggio `eps` di un punto core. Fa parte di un cluster ma è alla sua periferia.
    * **Punto di Rumore (Noise Point)**: Un punto che non è né un punto core né un punto di bordo. Questi sono considerati outlier e non appartengono a nessun cluster.
    
    **Punti di Forza:**
    * Non richiede di specificare il numero di cluster in anticipo.
    * Può trovare cluster di forme arbitrarie (non solo sferiche).
    * Identifica e gestisce automaticamente i punti di rumore/outlier, che in un contesto come questo potrebbero essere comportamenti di acquisto molto rari.
    
    **Punti di Debolezza:**
    * Sensibile ai parametri `eps` e `min_samples`; la loro scelta può essere complessa e richiede sperimentazione.
    * Non funziona bene con cluster di densità molto diverse.
    * Più lento di K-Means su dataset molto grandi.
    """)
    
    st.subheader("Quando usare DBSCAN per MarketPro (Individuazione Bodybuilder)?")
    st.markdown("""
    DBSCAN potrebbe essere una scelta **particolarmente interessante** per MarketPro per trovare i bodybuilder se:
    * Il gruppo dei bodybuilder è una **nicchia relativamente piccola ma molto densa** in termini di abitudini di acquisto specifiche (es. comprano tutti un certo tipo di prodotti con alta frequenza).
    * **Non hai un'idea precisa di quanti segmenti** la tua clientela generale sia composta, oltre a cercare un cluster specifico.
    * Vuoi che l'algoritmo **identifichi automaticamente gli "outlier"**, che potrebbero essere clienti con comportamenti di acquisto molto strani o rari, ma non necessariamente un cluster.
    * Ti aspetti che la **forma del cluster dei bodybuilder non sia necessariamente sferica**.

    **Vantaggio in questo scenario:** Se i bodybuilder sono un gruppo di nicchia, DBSCAN potrebbe isolarli più facilmente come un cluster separato, lasciando il resto della clientela come altri cluster o come "rumore", senza forzarli in un numero predefinito di K.
    """)

# --- Conclusione e Guida alla Scelta ---
st.markdown("---")
st.header("🎯 K-Means vs DBSCAN: Quale Scegliere per Individuare i Bodybuilder?")
st.markdown("""
Per MarketPro, la scelta tra K-Means e DBSCAN per individuare i bodybuilder dipende molto dalla **natura attesa di questo cluster** e dagli **obiettivi secondari dell'analisi**:

* **K-Means:** Utile se i bodybuilder sono un gruppo ben definito e **relativamente grande** (tra i K segmenti principali) e se la loro "densità" di acquisto è simile agli altri segmenti. Richiede una buona stima iniziale di `K`.
* **DBSCAN:** Potrebbe essere **più efficace per individuare una nicchia specifica e densa** come i bodybuilder, soprattutto se sono un gruppo **piuttosto piccolo** rispetto al resto della clientela. È eccellente per rilevare anche comportamenti "anomali" che non rientrano in nessun cluster. Potrebbe richiedere più tuning dei parametri (`eps`, `min_samples`).

**Il consiglio per MarketPro:**
* **Inizia con DBSCAN** se il tuo obiettivo primario è **isolare una nicchia specifica e potenziali anomalie**, senza preoccuparti del numero totale di cluster. È più flessibile per forme e dimensioni di cluster.
* **Considera K-Means** se vuoi segmentare **tutta la tua base clienti in un numero predefinito di gruppi** e credi che i bodybuilder rientrino naturalmente in uno di questi segmenti principali.

Sperimenta con entrambi gli algoritmi e i loro parametri per vedere quale ti dà i risultati più significativi e utilizzabili per le tue strategie di marketing!
""")
