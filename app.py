import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Configurazione Streamlit
st.set_page_config(page_title="Segmentazione Clienti Moda", layout="wide")
st.title("Segmentazione Clienti - E-commerce di Moda")

# Generazione dati più realistici
np.random.seed(42)

def generate_realistic_data():
    # Cluster 1: Clienti occasionali (pochi acquisti, bassa spesa)
    c1_acquisti = np.random.poisson(1.5, 60)
    c1_spesa = np.random.normal(50, 15, 60) * c1_acquisti
    
    # Cluster 2: Clienti fedeli (media frequenza, media spesa)
    c2_acquisti = np.random.poisson(4, 60)
    c2_spesa = np.random.normal(80, 20, 60) * c2_acquisti
    
    # Cluster 3: Clienti premium (pochi acquisti, alta spesa per acquisto)
    c3_acquisti = np.random.poisson(2, 40)
    c3_spesa = np.random.normal(200, 50, 40) * c3_acquisti
    
    # Cluster 4: Clienti compulsive (molti acquisti, spesa variabile)
    c4_acquisti = np.random.poisson(8, 40)
    c4_spesa = np.random.normal(60, 20, 40) * c4_acquisti
    
    # Combinazione dati
    data = pd.DataFrame({
        'acquisti_mensili': np.concatenate([c1_acquisti, c2_acquisti, c3_acquisti, c4_acquisti]),
        'spesa_totale': np.concatenate([c1_spesa, c2_spesa, c3_spesa, c4_spesa])
    })
    
    # Aggiungiamo spesa media per articolo
    data['spesa_media'] = data['spesa_totale'] / data['acquisti_mensili']
    data.replace([np.inf, -np.inf], 0, inplace=True)  # gestione divisione per zero
    
    return data[['acquisti_mensili', 'spesa_media']]

# Caricamento e preparazione dati
data = generate_realistic_data()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Modello K-Means ottimizzato
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(scaled_data)
data['cluster'] = kmeans.labels_

# Nuova descrizione cluster coerente
cluster_desc = {
    0: {
        'nome': "Sporadici",
        'caratteristiche': "1-2 acquisti/mese, <€50/articolo, spesa totale <€100",
        'strategie': [
            "Email di benvenuto con sconto 15%",
            "Content su come abbinare i capi",
            "Notifiche su saldi stagionali"
        ]
    },
    1: {
        'nome': "Fedeli",
        'caratteristiche': "3-5 acquisti/mese, €50-100/articolo, spesa €200-400",
        'strategie': [
            "Programma fedeltà con premi",
            "Anteprime collezioni",
            "Offerte personalizzate"
        ]
    },
    2: {
        'nome': "Premium",
        'caratteristiche': "1-3 acquisti/mese, >€150/articolo, spesa >€300",
        'strategie': [
            "Accesso VIP a lanci esclusivi",
            "Personal shopper virtuale",
            "Reso gratuito e packaging premium"
        ]
    },
    3: {
        'nome': "Compulsivi",
        'caratteristiche': "6+ acquisti/mese, <€50/articolo, spesa totale alta",
        'strategie': [
            "Bundle 'compra 3, paga 2'",
            "Abbonamento con vantaggi",
            "Notifiche su nuovi arrivi"
        ]
    }
}

# Interfaccia Streamlit migliorata
st.sidebar.header("Simula Nuovo Cliente")
col1, col2 = st.sidebar.columns(2)
with col1:
    acquisti = st.slider("Acquisti mensili", 1, 10, 2)
with col2:
    spesa_media = st.slider("Spesa media per articolo (€)", 20, 300, 80)

# Predizione
nuovo_cliente = scaler.transform([[acquisti, spesa_media]])
cluster = kmeans.predict(nuovo_cliente)[0]
info = cluster_desc[cluster]

# Visualizzazione
st.header("Risultato Segmentazione")
cols = st.columns([1, 2])
with cols[0]:
    st.metric("Cluster Assegnato", info['nome'])
    st.write(f"**Caratteristiche:** {info['caratteristiche']}")
    
    st.subheader("Strategie Consigliate:")
    for s in info['strategie']:
        st.write(f"✓ {s}")

with cols[1]:
    fig, ax = plt.subplots(figsize=(10,6))
    
    # Visualizzazione cluster
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for i in range(4):
        cluster_data = data[data['cluster'] == i]
        ax.scatter(cluster_data['acquisti_mensili'], 
                   cluster_data['spesa_media'], 
                   c=colors[i], 
                   label=cluster_desc[i]['nome'],
                   alpha=0.7)
    
    # Centroide e nuovo cliente
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(centroids[:,0], centroids[:,1], marker='X', s=200, c='black', label='Centroidi')
    ax.scatter(acquisti, spesa_media, marker='*', s=300, c='#FFD700', edgecolors='black', label='Nuovo Cliente')
    
    ax.set_xlabel('Acquisti Mensili')
    ax.set_ylabel('Spesa Media per Articolo (€)')
    ax.set_title('Segmentazione Clienti Moda')
    ax.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    st.pyplot(fig)

# Spiegazione scientifica
with st.expander("🔍 Come funziona il modello"):
    st.write("""
    **Logica di clustering:**
    - Utilizziamo l'algoritmo K-Means con k=4
    - Le variabili considerate sono:
      1. Numero di acquisti mensili
      2. Spesa media per articolo acquistato
    - I dati vengono standardizzati prima del clustering
    
    **Interpretazione:**
    - L'asse X mostra la frequenza di acquisto
    - L'asse Y mostra la qualità/prezzo medio degli acquisti
    - I cluster emergono naturalmente da queste due dimensioni
    """)
