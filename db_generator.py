import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap

# --- 1. Generazione Dati Sintetici Realistici ---
# Questi dati rappresentano 4 segmenti di clienti con caratteristiche distinte
np.random.seed(42) # Per garantire la riproducibilità dei dati

n_clienti_per_gruppo = 200
n_totale_clienti = n_clienti_per_gruppo * 4

# Gruppo 1: "Clienti High-Value" (spesa alta, acquisti frequenti)
spesa_1 = np.random.normal(loc=500, scale=80, size=n_clienti_per_gruppo)
acquisti_1 = np.random.normal(loc=15, scale=3, size=n_clienti_per_gruppo)

# Gruppo 2: "Clienti Budget" (spesa bassa, acquisti rari)
spesa_2 = np.random.normal(loc=100, scale=30, size=n_clienti_per_gruppo)
acquisti_2 = np.random.normal(loc=3, scale=1, size=n_clienti_per_gruppo)

# Gruppo 3: "Clienti Fedeli a Media Spesa" (spesa media, acquisti molto frequenti)
spesa_3 = np.random.normal(loc=250, scale=50, size=n_clienti_per_gruppo)
acquisti_3 = np.random.normal(loc=25, scale=5, size=n_clienti_per_gruppo)

# Gruppo 4: "Clienti Occasionali Alta Spesa" (spesa alta, acquisti rari ma grandi)
spesa_4 = np.random.normal(loc=450, scale=100, size=n_clienti_per_gruppo)
acquisti_4 = np.random.normal(loc=5, scale=2, size=n_clienti_per_gruppo)

# Combina i dati
df = pd.DataFrame({
    'Spesa Media Mensile': np.concatenate([spesa_1, spesa_2, spesa_3, spesa_4]),
    'Numero Acquisti Mensili': np.concatenate([acquisti_1, acquisti_2, acquisti_3, acquisti_4])
})

# Assicurati che non ci siano valori negativi (perché spesa/acquisti non possono esserlo)
df['Spesa Media Mensile'] = df['Spesa Media Mensile'].apply(lambda x: max(0, x))
df['Numero Acquisti Mensili'] = df['Numero Acquisti Mensili'].apply(lambda x: max(0, x))

print(f"Dataset generato con {len(df)} clienti.")
print(df.head())
print("\nStatistiche descrittive del dataset:")
print(df.describe())

# --- 2. Preprocessing dei Dati ---
# È fondamentale scalare i dati per K-Means, poiché è sensibile alla scala delle feature.
# StandardScaler rende la media 0 e la deviazione standard 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Spesa Media Mensile', 'Numero Acquisti Mensili']])

# Converti in DataFrame per facilità di visualizzazione e gestione
df_scaled = pd.DataFrame(X_scaled, columns=['Spesa Media Mensile Scalata', 'Numero Acquisti Mensili Scalato'])

# --- 3. Applicazione dell'Algoritmo K-Means ---
# Definiamo il numero di cluster (k) che ci aspettiamo
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto') # n_init='auto' per K-Means standard
kmeans.fit(X_scaled)

# Assegna le etichette dei cluster al DataFrame originale
df['Cluster'] = kmeans.labels_

# Ottieni i centroidi (in spazio scalato)
centroids_scaled = kmeans.cluster_centers_

# Trasforma i centroidi nello spazio originale per una migliore interpretabilità
centroids_original_scale = scaler.inverse_transform(centroids_scaled)
df_centroids = pd.DataFrame(centroids_original_scale, columns=['Spesa Media Mensile', 'Numero Acquisti Mensili'])
df_centroids['Cluster'] = range(k) # Assegna un ID cluster ai centroidi

# --- 4. Valutazione del Clustering ---
# Silhouette Score: misura quanto i cluster sono ben separati.
# Un valore più alto (vicino a 1) indica cluster ben distinti.
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f"\nSilhouette Score: {silhouette_avg:.3f}")

# --- 5. Visualizzazione dei Cluster e dei Centroidi ---
plt.figure(figsize=(10, 7))

# Definisci una colormap per i cluster
cmap = ListedColormap(['#FF9999', '#99FF99', '#9999FF', '#FFFF99']) # Colori personalizzati

# Grafico a dispersione dei clienti con i cluster colorati
scatter = plt.scatter(df['Spesa Media Mensile'], df['Numero Acquisti Mensili'],
                      c=df['Cluster'], cmap=cmap, s=50, alpha=0.7, edgecolors='w')

# Aggiungi i centroidi
plt.scatter(df_centroids['Spesa Media Mensile'], df_centroids['Numero Acquisti Mensili'],
            marker='X', s=200, color='black', label='Centroidi', edgecolors='w', linewidth=2)

plt.title('Segmentazione Clienti con K-Means (Spazio Originale)')
plt.xlabel('Spesa Media Mensile (€)')
plt.ylabel('Numero Acquisti Mensili')
plt.grid(True, linestyle='--', alpha=0.6)

# Crea la legenda per i cluster
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
                              markerfacecolor=cmap(i), markersize=10) for i in range(k)]
legend_elements.append(plt.Line2D([0], [0], marker='X', color='w', label='Centroidi',
                                  markerfacecolor='black', markersize=10, markeredgecolor='w'))
plt.legend(handles=legend_elements, title="Gruppi di Clienti")

plt.tight_layout()
plt.show()

# --- 6. Assegnazione di un Nuovo Cliente ---
print("\n--- Assegnazione di un Nuovo Cliente ---")

# Dati del nuovo cliente
nuovo_cliente_spesa = 320 # Esempio: un cliente con spesa media
nuovo_cliente_acquisti = 12 # Esempio: un cliente con numero medio di acquisti

nuovo_cliente_data = pd.DataFrame([[nuovo_cliente_spesa, nuovo_cliente_acquisti]],
                                   columns=['Spesa Media Mensile', 'Numero Acquisti Mensili'])

print(f"Nuovo cliente: Spesa = {nuovo_cliente_spesa}€, Acquisti = {nuovo_cliente_acquisti}")

# Scala i dati del nuovo cliente usando lo stesso scaler
nuovo_cliente_scaled = scaler.transform(nuovo_cliente_data)

# Prevedi il cluster per il nuovo cliente
predicted_cluster = kmeans.predict(nuovo_cliente_scaled)[0]
print(f"Il nuovo cliente è assegnato al Cluster: {predicted_cluster}")

# --- 7. Visualizzazione con il Nuovo Cliente ---
plt.figure(figsize=(10, 7))

# Grafico a dispersione dei clienti esistenti
scatter = plt.scatter(df['Spesa Media Mensile'], df['Numero Acquisti Mensili'],
                      c=df['Cluster'], cmap=cmap, s=50, alpha=0.7, edgecolors='w')

# Aggiungi i centroidi
plt.scatter(df_centroids['Spesa Media Mensile'], df_centroids['Numero Acquisti Mensili'],
            marker='X', s=200, color='black', label='Centroidi', edgecolors='w', linewidth=2)

# Aggiungi il nuovo cliente
plt.scatter(nuovo_cliente_spesa, nuovo_cliente_acquisti,
            marker='*', s=300, color='red', label='Nuovo Cliente', edgecolors='white', linewidth=2, zorder=5)

# Aggiungi una linea tratteggiata dal nuovo cliente al centroide del suo cluster
centroid_of_new_client = df_centroids.loc[predicted_cluster]
plt.plot([nuovo_cliente_spesa, centroid_of_new_client['Spesa Media Mensile']],
         [nuovo_cliente_acquisti, centroid_of_new_client['Numero Acquisti Mensili']],
         'r--', linewidth=1, alpha=0.7)


plt.title('Segmentazione Clienti e Assegnazione Nuovo Cliente')
plt.xlabel('Spesa Media Mensile (€)')
plt.ylabel('Numero Acquisti Mensili')
plt.grid(True, linestyle='--', alpha=0.6)

# Crea la legenda aggiornata
legend_elements_new = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
                                  markerfacecolor=cmap(i), markersize=10) for i in range(k)]
legend_elements_new.append(plt.Line2D([0], [0], marker='X', color='w', label='Centroidi',
                                      markerfacecolor='black', markersize=10, markeredgecolor='w'))
legend_elements_new.append(plt.Line2D([0], [0], marker='*', color='w', label='Nuovo Cliente',
                                      markerfacecolor='red', markersize=12, markeredgecolor='white'))
plt.legend(handles=legend_elements_new, title="Legenda")

plt.tight_layout()
plt.show()

# --- 8. Analisi dei Cluster (Profilazione) ---
print("\n--- Profilazione dei Cluster ---")
cluster_profiles = df.groupby('Cluster').agg({
    'Spesa Media Mensile': ['mean', 'median', 'std'],
    'Numero Acquisti Mensili': ['mean', 'median', 'std'],
    'Cluster': 'count' # Conta i clienti per cluster
}).rename(columns={'Cluster': 'Numero Clienti'})

print(cluster_profiles)

print("\nInterpretazione (Esempio):")
print("- Cluster 0 (Es: 'Clienti High-Value'): Alta spesa media, alto numero di acquisti.")
print("- Cluster 1 (Es: 'Clienti Budget'): Bassa spesa media, basso numero di acquisti.")
print("- Cluster 2 (Es: 'Clienti Fedeli a Media Spesa'): Spesa media, numero di acquisti molto alto.")
print("- Cluster 3 (Es: 'Clienti Occasionali Alta Spesa'): Alta spesa media, basso numero di acquisti.")
print("\nQuesti profili sono basati sui dati sintetici e dovrebbero essere interpretati in base ai valori specifici ottenuti.")
