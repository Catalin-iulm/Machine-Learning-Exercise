# kmeans_app.py migliorato
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np # Aggiunto per le operazioni numeriche

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="K-Means Didattico")

# --- 1. Titolo e Introduzione ---
st.title("🎯 Scopriamo il K-Means Clustering! 📊")
st.markdown("""
Benvenuto in questa app interattiva per esplorare l'algoritmo di **K-Means Clustering**!
Il K-Means è un algoritmo di Machine Learning non supervisionato che raggruppa punti dati simili in cluster.
L'obiettivo è dividere `n` osservazioni in `k` cluster, dove ogni osservazione appartiene al cluster con il centroide più vicino.
""")

st.info("💡 Usa i controlli nella sidebar a sinistra per personalizzare la simulazione.")

# --- 2. Controlli a sinistra (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Impostazioni del K-Means")
    
    # Controlli per la generazione dei dati
    st.subheader("Generazione Dati")
    n_punti = st.slider("Numero di Punti Dati", 50, 1000, 300, help="Quanti punti generare per la simulazione.")
    n_gruppi_reali = st.slider("Numero di Gruppi Reali (per generazione)", 2, 8, 4, help="Quanti cluster 'naturali' ci sono nei dati generati.")
    seed_generazione = st.slider("Seed per Generazione Dati (random_state)", 0, 100, 42, help="Controlla la riproducibilità della generazione dei dati.")
    
    # Controlli per l'algoritmo K-Means
    st.subheader("Algoritmo K-Means")
    n_gruppi_kmeans = st.slider("Numero di Gruppi da Trovare (K)", 2, 10, 4, help="Il valore 'K' che l'algoritmo K-Means cercherà di trovare.")
    seed_kmeans = st.slider("Seed per K-Means (random_state)", 0, 100, 0, help="Controlla la riproducibilità dell'inizializzazione di K-Means. Impostalo a 0 per casualità.")
    
    st.markdown("---")
    st.caption("App creata con ❤️ e Streamlit")

# --- 3. Genera Dati (Caching per performance) ---
@st.cache_data
def generate_data(n_samples, n_centers, random_state_data):
    X, y = make_blobs(n_samples=n_samples, centers=n_centers, cluster_std=1.0, random_state=random_state_data)
    return X, y

X, y_true = generate_data(n_punti, n_gruppi_reali, seed_generazione)

# --- 4. Mostra Dati Originali ---
st.subheader("📈 Dati Originali Generati")
st.write("Questi sono i punti dati generati casualmente. K-Means cercherà di trovare i gruppi nascosti in essi.")

fig_orig, ax_orig = plt.subplots(figsize=(8, 6))
ax_orig.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, edgecolors='w', s=50)
ax_orig.set_title("Punti Dati Generati")
ax_orig.set_xlabel("Feature 1")
ax_orig.set_ylabel("Feature 2")
ax_orig.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig_orig)

st.markdown("---")

# --- 5. Esegui K-Means e Mostra Risultati ---
st.subheader(f"✨ Risultati del K-Means (con K = {n_gruppi_kmeans})")

if st.button("▶️ Esegui K-Means e Analizza!", help="Clicca per far eseguire l'algoritmo K-Means sui dati."):
    st.spinner("Calcolo del K-Means in corso...")
    
    kmeans = KMeans(n_clusters=n_gruppi_kmeans, random_state=seed_kmeans, n_init=10) # n_init per robustezza
    gruppi_predetti = kmeans.fit_predict(X)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cluster Troppo dal K-Means")
        fig_kmeans, ax_kmeans = plt.subplots(figsize=(8, 6))
        ax_kmeans.scatter(X[:, 0], X[:, 1], c=gruppi_predetti, cmap='viridis', alpha=0.7, edgecolors='w', s=50)
        ax_kmeans.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                        s=300, c='red', marker='X', label="Centroidi")
        ax_kmeans.set_title(f"Punti Dati Clustered (K={n_gruppi_kmeans})")
        ax_kmeans.set_xlabel("Feature 1")
        ax_kmeans.set_ylabel("Feature 2")
        ax_kmeans.legend()
        ax_kmeans.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_kmeans)
        
    with col2:
        st.subheader("Metriche di Valutazione")
        st.metric(label="Inerzia (WCSS)", value=f"{kmeans.inertia_:.2f}", 
                  help="La somma delle distanze al quadrato dei punti dal centroide del loro cluster. Un valore più basso indica cluster più compatti.")
        
        st.markdown("""
        L'**Inerzia** (o Within-Cluster Sum of Squares - WCSS) misura quanto sono compatti i cluster.
        È la somma delle distanze al quadrato di ogni punto dal centroide del suo cluster.
        
        **Un valore di inerzia più basso indica cluster più densi e compatti.**
        """)

# --- 6. Sezione Didattica (Espandibile) ---
st.markdown("---")
st.header("📚 Approfondimenti sul K-Means")

with st.expander("🤔 Cos'è e come funziona il K-Means?"):
    st.markdown("""
    Il K-Means è un algoritmo iterativo che mira a minimizzare l'inerzia. Funziona così:
    
    1.  **Inizializzazione:** Vengono scelti `K` centroidi iniziali (posizioni casuali o con metodi più avanzati come K-Means++).
    2.  **Assegnazione:** Ogni punto dati viene assegnato al centroide più vicino. Questo forma i `K` cluster.
    3.  **Aggiornamento:** I centroidi vengono ricalcolati come la media (il "centro") di tutti i punti assegnati al loro rispettivo cluster.
    4.  **Iterazione:** I passi 2 e 3 vengono ripetuti finché i centroidi non si muovono più significativamente o viene raggiunto un numero massimo di iterazioni.
    
    L'algoritmo **converge** quando i cluster diventano stabili.
    """)

with st.expander("📏 Come scegliere il numero di gruppi 'K' (Metodo del Gomito)?"):
    st.markdown("""
    Uno dei maggiori interrogativi con K-Means è come scegliere il valore ottimale di `K` (il numero di cluster).
    Il **Metodo del Gomito (Elbow Method)** è una tecnica euristica comune:
    
    1.  Esegui K-Means per un intervallo di valori di `K` (es. da 1 a 10).
    2.  Calcola l'inerzia per ogni `K`.
    3.  Traccia un grafico dell'inerzia in funzione di `K`.
    
    Cerca un "gomito" nel grafico, cioè il punto in cui la diminuzione dell'inerzia inizia a rallentare significativamente. Questo punto è spesso considerato un buon compromesso per `K`.
    """)
    
    if st.button("Visualizza il Metodo del Gomito"):
        inertias = []
        k_values = range(1, 11) # Valori di K da testare
        
        # ProgressBar per feedback utente
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, k in enumerate(k_values):
            status_text.text(f"Calcolo per K = {k}...")
            kmeans_elbow = KMeans(n_clusters=k, random_state=seed_kmeans, n_init=10)
            kmeans_elbow.fit(X)
            inertias.append(kmeans_elbow.inertia_)
            progress_bar.progress((i + 1) / len(k_values))
        
        status_text.success("Calcolo completato!")
        progress_bar.empty() # Rimuovi la progress bar dopo il completamento

        fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
        ax_elbow.plot(k_values, inertias, marker='o', linestyle='-', color='purple')
        ax_elbow.set_title("Metodo del Gomito per la Scelta di K")
        ax_elbow.set_xlabel("Numero di Cluster (K)")
        ax_elbow.set_ylabel("Inerzia (WCSS)")
        ax_elbow.grid(True, linestyle='--', alpha=0.7)
        ax_elbow.set_xticks(k_values)
        st.pyplot(fig_elbow)
        st.write("Cerca un punto nel grafico dove la curva si piega bruscamente, formando un 'gomito'. Questo suggerisce un buon valore di K.")
