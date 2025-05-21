import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Configurazione Streamlit
st.set_page_config(page_title="Segmentazione Clienti Moda", layout="wide")
st.title("Segmentazione Clienti - E-commerce di Moda")

# Generazione dati robusta
np.random.seed(42)

def generate_robust_data():
    # Cluster 1: Clienti occasionali (1-2 acquisti, basso valore)
    c1_acquisti = np.random.randint(1, 3, 60)
    c1_spesa = np.random.uniform(30, 70, 60) * c1_acquisti
    
    # Cluster 2: Clienti fedeli (3-5 acquisti, medio valore)
    c2_acquisti = np.random.randint(3, 6, 60)
    c2_spesa = np.random.uniform(50, 120, 60) * c2_acquisti
    
    # Cluster 3: Clienti premium (1-3 acquisti, alto valore)
    c3_acquisti = np.random.randint(1, 4, 40)
    c3_spesa = np.random.uniform(150, 300, 40) * c3_acquisti
    
    # Cluster 4: Clienti compulsive (6+ acquisti, basso valore)
    c4_acquisti = np.random.randint(6, 10, 40)
    c4_spesa = np.random.uniform(30, 70, 40) * c4_acquisti
    
    # Creazione DataFrame
    data = pd.DataFrame({
        'acquisti_mensili': np.concatenate([c1_acquisti, c2_acquisti, c3_acquisti, c4_acquisti]),
        'spesa_totale': np.concatenate([c1_spesa, c2_spesa, c3_spesa, c4_spesa])
    })
    
    # Calcolo spesa media con gestione degli zeri
    data['spesa_media'] = np.where(data['acquisti_mensili'] > 0, 
                                 data['spesa_totale'] / data['acquisti_mensili'], 
                                 0)
    
    # Filtriamo i casi con acquisti mensili = 0 (se presenti)
    data = data[data['acquisti_mensili'] > 0]
    
    return data[['acquisti_mensili', 'spesa_media']]

# Caricamento e preparazione dati
data = generate_robust_data()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Modello K-Means con gestione errori
try:
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    data['cluster'] = kmeans.labels_
except Exception as e:
    st.error(f"Errore nel clustering: {str(e)}")
    st.stop()

# Descrizione cluster ottimizzata
cluster_desc = {
    0: {
        'nome': "Sporadici",
        'caratteristiche': "1-2 acquisti/mese, €30-70/articolo",
        'strategie': [
            "Sconto benvenuto 15%",
            "Guide di stile gratuite",
            "Notifiche saldi stagionali"
        ]
    },
    1: {
        'nome': "Fedeli",
        'caratteristiche': "3-5 acquisti/mese, €50-120/articolo",
        'strategie': [
            "Programma fedeltà premium",
            "Accesso anticipato alle nuove collezioni",
            "Omaggi esclusivi"
        ]
    },
    2: {
        'nome': "Premium",
        'caratteristiche': "1-3 acquisti/mese, €150-300/articolo",
        'strategie': [
            "Servizio personal shopper",
            "Inviti a eventi esclusivi",
            "Reso gratuito illimitato"
        ]
    },
    3: {
        'nome': "Compulsivi",
        'caratteristiche': "6+ acquisti/mese, €30-70/articolo",
        'strategie': [
            "Pacchetti scontati 'compra più, risparmia'",
            "Abbonamento con vantaggi",
            "Alert nuovi arrivi"
        ]
    }
}

# Interfaccia Streamlit
st.sidebar.header("Simula Nuovo Cliente")
acquisti = st.sidebar.slider("Acquisti mensili", 1, 10, 2)
spesa_media = st.sidebar.slider("Spesa media per articolo (€)", 20, 300, 80)

# Predizione con gestione errori
try:
    nuovo_cliente = scaler.transform([[acquisti, spesa_media]])
    cluster = kmeans.predict(nuovo_cliente)[0]
    info = cluster_desc[cluster]
except Exception as e:
    st.error(f"Errore nella predizione: {str(e)}")
    st.stop()

# Visualizzazione risultati
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Risultato Segmentazione")
    st.metric("Cluster", info['nome'])
    st.write(f"**Profilo:** {info['caratteristiche']}")
    
    st.subheader("Strategie Consigliate")
    for i, strategia in enumerate(info['strategie'], 1):
        st.write(f"{i}. {strategia}")

with col2:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for i in range(4):
        cluster_data = data[data['cluster'] == i]
        ax.scatter(cluster_data['acquisti_mensili'], 
                 cluster_data['spesa_media'], 
                 c=colors[i], 
                 label=cluster_desc[i]['nome'],
                 alpha=0.6)
    
    # Centroidi
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black', label='Centroidi')
    
    # Nuovo cliente
    ax.scatter(acquisti, spesa_media, marker='*', s=300, c='gold', edgecolor='black', label='Nuovo Cliente')
    
    ax.set_xlabel('Acquisti Mensili')
    ax.set_ylabel('Spesa Media per Articolo (€)')
    ax.set_title('Segmentazione Clienti Moda')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

# Informazioni tecniche
with st.expander("ℹ️ Informazioni tecniche"):
    st.write("""
    **Parametri del modello:**
    - Algoritmo: K-Means
    - Numero cluster: 4
    - Variabili: 
      - Acquisti mensili (conteggio)
      - Spesa media per articolo (€)
    - Dati standardizzati prima del clustering
    
    **Statistiche descrittive:**
    """)
    st.write(data.describe())
