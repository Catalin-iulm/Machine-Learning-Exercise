import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D  # Per la visualizzazione 3D

def generate_sample_data(n_samples=300, n_features=3, n_centers=4, random_state=42):
    """Genera dati di esempio per il clustering."""
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        random_state=random_state,
        cluster_std=1.0
    )
    return X, y

def preprocess_data(X):
    """Standardizza i dati."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def find_optimal_clusters(X_scaled, max_k=10):
    """Determina il numero ottimale di cluster usando il metodo Elbow."""
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    return inertia

def plot_elbow_method(inertia, max_k=10):
    """Visualizza il metodo Elbow per determinare il numero ottimale di cluster."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertia, marker='o', linestyle='--')
    plt.title('Metodo Elbow per la determinazione del numero ottimale di cluster')
    plt.xlabel('Numero di cluster (K)')
    plt.ylabel('Inerzia')
    plt.grid(True)
    plt.xticks(range(1, max_k + 1))
    plt.tight_layout()
    plt.show()

def perform_kmeans(X_scaled, n_clusters=4, random_state=42):
    """Esegue l'algoritmo K-Means e restituisce i cluster e i centroidi."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    return cluster_labels, centroids

def plot_clusters_3d(X_scaled, cluster_labels, centroids):
    """Visualizza i cluster in 3D con i centroidi."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colori diversi per ogni cluster
    scatter = ax.scatter(
        X_scaled[:, 0], 
        X_scaled[:, 1], 
        X_scaled[:, 2], 
        c=cluster_labels, 
        cmap='viridis', 
        s=50, 
        alpha=0.6,
        edgecolor='w'
    )
    
    # Plot dei centroidi
    ax.scatter(
        centroids[:, 0], 
        centroids[:, 1], 
        centroids[:, 2], 
        marker='X', 
        s=200, 
        color='red', 
        label='Centroidi',
        edgecolor='k',
        linewidth=1
    )
    
    ax.set_xlabel('Feature 1 (scalata)')
    ax.set_ylabel('Feature 2 (scalata)')
    ax.set_zlabel('Feature 3 (scalata)')
    ax.set_title(f'Risultato del clustering K-Means (K={len(centroids)})')
    ax.legend()
    plt.tight_layout()
    plt.show()

def evaluate_clustering(X_scaled, cluster_labels):
    """Valuta la qualità del clustering usando il Silhouette Score."""
    score = silhouette_score(X_scaled, cluster_labels)
    print(f"\nSilhouette Score: {score:.3f}")
    print("Interpretazione:")
    print("- Valore vicino a +1: cluster ben separati")
    print("- Valore vicino a 0: cluster che si sovrappongono")
    print("- Valore vicino a -1: punti assegnati probabilmente al cluster sbagliato")

def main():
    # 1. Generazione dei dati
    print("Generazione dei dati di esempio...")
    X, y = generate_sample_data()
    
    # 2. Preprocessing
    print("\nStandardizzazione delle features...")
    X_scaled = preprocess_data(X)
    
    # 3. Determinazione numero ottimale di cluster
    print("\nCalcolo del numero ottimale di cluster con il metodo Elbow...")
    inertia = find_optimal_clusters(X_scaled)
    plot_elbow_method(inertia)
    
    # 4. Esecuzione K-Means con K ottimale (scelto dall'utente o automatico)
    optimal_k = 4  # Da modificare in base al metodo Elbow
    print(f"\nEsecuzione di K-Means con K={optimal_k}...")
    cluster_labels, centroids = perform_kmeans(X_scaled, n_clusters=optimal_k)
    
    # 5. Visualizzazione risultati
    print("\nVisualizzazione dei cluster...")
    plot_clusters_3d(X_scaled, cluster_labels, centroids)
    
    # 6. Valutazione del clustering
    print("\nValutazione della qualità del clustering...")
    evaluate_clustering(X_scaled, cluster_labels)
    
    # 7. Informazioni aggiuntive
    print("\nInformazioni sui cluster:")
    for i in range(optimal_k):
        cluster_size = np.sum(cluster_labels == i)
        print(f"Cluster {i}: {cluster_size} punti")
    
    print("\nPosizioni dei centroidi:")
    print(centroids)

if __name__ == "__main__":
    main()
