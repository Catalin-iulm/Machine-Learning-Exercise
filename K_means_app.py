import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # Per generare un dataset di esempio
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score # Per le metriche di performance

# 1. Generazione di un dataset di esempio (o caricamento di uno reale)
# Supponiamo di avere 3 variabili esplicative (features)
n_samples = 300
random_state = 42
X, y = make_blobs(n_samples=n_samples, n_features=3, centers=4, random_state=random_state, cluster_std=1.0)
# Per il clustering, ignoriamo 'y' che rappresenta le classi vere, ma 'make_blobs' le genera per coerenza

# 2. Preprocessing: Scaling delle feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Determinazione del numero ottimale di cluster (metodo "Elbow Method")
# Questo è un passo importante per determinare un buon 'K'
inertia = []
for i in range(1, 11): # Proviamo da 1 a 10 cluster
    kmeans = KMeans(n_clusters=i, random_state=random_state, n_init='auto')
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_) # La somma delle distanze al quadrato dei punti dal loro centroide

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
# plt.show() # Non lo mostriamo subito per Streamlit, ma in Jupyter/Colab lo faresti

# Dal grafico, si cerca il "gomito" che indica il K ottimale. Supponiamo K=4 per il nostro esempio.

# 4. Addestramento del modello K-Means con il K scelto
n_clusters = 4 # Dal metodo del gomito
kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
kmeans_model.fit(X_scaled)

# 5. Ottenere le assegnazioni dei cluster per ogni punto e i centroidi
cluster_labels = kmeans_model.labels_
centroids = kmeans_model.cluster_centers_

# --- Simulazione delle prime 10 iterazioni (manualmente per l'esempio concettuale) ---
# NB: Scikit-learn non espone direttamente i passaggi intermedi delle iterazioni K-Means.
# Per mostrare i passaggi, dovremmo implementare una versione semplificata del K-Means da zero
# o spiegare il concetto iterativo. Per questo esercizio, simuliamo il *concetto* delle iterazioni
# mostrando come l'algoritmo *idealmente* raffinerebbe i cluster.

print("\n--- Simulazione Concettuale dei Primi 10 Passaggi dell'Algoritmo K-Means ---")
print("*(Nota: Scikit-learn non espone direttamente le iterazioni intermedie. Questo è un esempio concettuale per illustrare il processo.)*")

# Per simulare le iterazioni, useremo un K-Means semplificato o spiegheremo il concetto.
# Per l'esercizio, possiamo spiegare cosa succede in ogni iterazione, mostrando immagini progressive.
# In un'implementazione Streamlit, potremmo avere un "slider" che visualizza i risultati
# dopo 'n' iterazioni se implementassimo il K-Means a basso livello.

# Ecco il concetto dei passaggi che spiegheresti:
# Passaggio 1: Inizializzazione dei centroidi (casuale o K-Means++)
# Passaggio 2: Assegnazione dei punti al centroide più vicino
# Passaggio 3: Ricalcolo dei centroidi
# Passaggio 4: Assegnazione dei punti al centroide più vicino (con i nuovi centroidi)
# Passaggio 5: Ricalcolo dei centroidi
# ... e così via fino alla convergenza (ad esempio, il 10° passaggio mostra uno stato quasi finale).

# Per il PowerPoint e la simulazione in Streamlit, potremmo visualizzare:
# - Stato iniziale (punti e centroidi casuali)
# - Dopo 1 iterazione (punti riassegnati, centroidi ricalcolati)
# - Dopo 2 iterazioni...
# - ...
# - Dopo 10 iterazioni (mostrando come si sono stabilizzati i cluster)

# Per visualizzare questo in Python (non le 10 iterazioni esatte, ma il risultato finale con un K scelto):
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d') # Per 3 variabili esplicative

# Scatter plot dei punti colorati in base al cluster assegnato
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=cluster_labels, cmap='viridis', s=50, alpha=0.6)

# Plot dei centroidi
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='X', s=200, color='red', label='Centroids')

ax.set_xlabel('Feature 1 (Scaled)')
ax.set_ylabel('Feature 2 (Scaled)')
ax.set_zlabel('Feature 3 (Scaled)')
ax.set_title(f'K-Means Clustering (K={n_clusters})')
ax.legend()
# plt.show() # Anche questo da usare in Streamlit

# --- Fine Simulazione Concettuale ---

# 6. Prestazioni (Metriche di Performance)
# Per il clustering, le metriche sono diverse dalla regressione/classificazione supervisionata.
# Non abbiamo etichette "vere" per confrontare, quindi usiamo metriche di "coerenza" interna.

# Silhouette Score: Misura quanto un oggetto è simile al proprio cluster (coesione) rispetto ad altri cluster (separazione).
# Il valore varia da -1 (assegnazione sbagliata) a +1 (cluster ben separati).
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"\nSilhouette Score: {silhouette_avg:.3f}")

# Altre metriche: Davies-Bouldin Index, Calinski-Harabasz Index.
# Non le implementiamo tutte qui, ma puoi menzionarle nel PowerPoint.

# 7. Struttura del Modello (Spiegazione)
# Descrizione del modello addestrato: numero di cluster, posizione dei centroidi.
# Possibili interpretazioni dei cluster trovati.

# 8. Simulazione (Interfaccia Streamlit)
# Questo sarà il cuore dell'applicazione Streamlit. Permetterà all'utente di:
# - Caricare un dataset (o usare quello predefinito).
# - Selezionare il numero di cluster (K).
# - Visualizzare il metodo del gomito.
# - Eseguire l'algoritmo e visualizzare i cluster risultanti.
# - Vedere le metriche di performance.
# - (Opzionale) Interagire con un "nuovo punto" e vedere a quale cluster verrebbe assegnato.

# 9. FAQ (Domande Frequenti)
# - Come scegliere il valore di K? (Elbow Method, Silhouette Score)
# - Quali sono i limiti del K-Means? (Sensibilità all'inizializzazione, forma dei cluster)
# - Quando usare K-Means?
# - K-Means è un algoritmo deterministico? (No, per l'inizializzazione casuale)

