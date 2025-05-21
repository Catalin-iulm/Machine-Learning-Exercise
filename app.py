import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Configurazione Streamlit
st.set_page_config(page_title="Segmentazione Clienti - Fashion E-commerce", layout="wide")
st.title("Segmentazione Clienti - Fashion E-commerce")

# Generazione dati sintetici
np.random.seed(42)

def generate_customer_data():
    # Cluster 1: Clienti occasionali (bassa spesa, pochi acquisti)
    cluster1_spesa = np.random.normal(80, 15, 50)
    cluster1_acquisti = np.random.normal(2, 0.5, 50)
    
    # Cluster 2: Clienti fedeli (media spesa, media frequenza)
    cluster2_spesa = np.random.normal(150, 20, 50)
    cluster2_acquisti = np.random.normal(5, 1, 50)
    
    # Cluster 3: Clienti premium (alta spesa, bassa frequenza)
    cluster3_spesa = np.random.normal(300, 40, 50)
    cluster3_acquisti = np.random.normal(3, 0.7, 50)
    
    # Cluster 4: Clienti quantity-driven (bassa spesa, alta frequenza)
    cluster4_spesa = np.random.normal(100, 15, 50)
    cluster4_acquisti = np.random.normal(8, 1.5, 50)
    
    # Combiniamo i dati
    spesa = np.concatenate([cluster1_spesa, cluster2_spesa, cluster3_spesa, cluster4_spesa])
    acquisti = np.concatenate([cluster1_acquisti, cluster2_acquisti, cluster3_acquisti, cluster4_acquisti])
    
    # Creiamo un DataFrame
    data = pd.DataFrame({
        'spesa_media': spesa,
        'acquisti_mensili': acquisti
    })
    
    return data

# Carichiamo i dati
data = generate_customer_data()

# Normalizzazione dei dati
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Applicazione K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_data)
data['cluster'] = kmeans.labels_

# Descrizioni dei cluster
cluster_descriptions = {
    0: {
        'nome': "Clienti Occasionali",
        'descrizione': "Clienti che acquistano raramente e spendono poco. Potrebbero essere nuovi clienti o clienti poco coinvolti.",
        'strategie': [
            "Email di benvenuto con sconto per il primo acquisto",
            "Promozioni mirate per incentivare acquisti ripetuti",
            "Content marketing per aumentare l'engagement"
        ]
    },
    1: {
        'nome': "Clienti Fedeli",
        'descrizione': "Clienti con un buon livello di spesa e frequenza di acquisto. Sono la base solida del business.",
        'strategie': [
            "Programma fedeltà con punti e ricompense",
            "Offerte personalizzate basate sull'acquisto precedente",
            "Anteprime esclusive di nuove collezioni"
        ]
    },
    2: {
        'nome': "Clienti Premium",
        'descrizione': "Clienti che spendono molto per ogni acquisto, anche se con minore frequenza. Valorizzano qualità e esclusività.",
        'strategie': [
            "Accesso VIP a prodotti limitati",
            "Servizio personalizzato di styling",
            "Packaging premium e consegna espressa gratuita"
        ]
    },
    3: {
        'nome': "Clienti Quantity-Driven",
        'descrizione': "Clienti che acquistano frequentemente ma con spesa media bassa. Sono sensibili alle promozioni.",
        'strategie': [
            "Bundle promozionali (es. 3 capi a prezzo speciale)",
            "Programma 'acquista 10, ottieni 1 gratis'",
            "Notifiche su prodotti in saldo"
        ]
    }
}

# Interfaccia Streamlit
st.sidebar.header("Simulazione Nuovo Cliente")
nuova_spesa = st.sidebar.slider("Spesa media mensile (€)", 50, 400, 150)
nuovi_acquisti = st.sidebar.slider("Numero di acquisti mensili", 1, 10, 4)

# Previsione per il nuovo cliente
nuovo_cliente = scaler.transform([[nuova_spesa, nuovi_acquisti]])
cluster_predetto = kmeans.predict(nuovo_cliente)[0]
cluster_info = cluster_descriptions[cluster_predetto]

# Visualizzazione risultati
col1, col2 = st.columns(2)

with col1:
    st.subheader("Risultato della Segmentazione")
    st.write(f"**Cluster assegnato:** {cluster_info['nome']}")
    st.write(f"**Descrizione:** {cluster_info['descrizione']}")
    st.subheader("Strategie Consigliate:")
    for strategia in cluster_info['strategie']:
        st.write(f"- {strategia}")

with col2:
    # Visualizzazione grafica
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red', 'green', 'blue', 'purple']
    for i in range(4):
        cluster_data = data[data['cluster'] == i]
        ax.scatter(cluster_data['spesa_media'], cluster_data['acquisti_mensili'], 
                   color=colors[i], label=cluster_descriptions[i]['nome'], alpha=0.6)
    
    # Plot dei centroidi
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='black', label='Centroidi')
    
    # Plot del nuovo cliente
    ax.scatter(nuova_spesa, nuovi_acquisti, marker='*', s=300, color='gold', label='Nuovo Cliente')
    
    ax.set_xlabel('Spesa Media Mensile (€)')
    ax.set_ylabel('Numero di Acquisti Mensili')
    ax.set_title('Segmentazione Clienti - Fashion E-commerce')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

# Informazioni aggiuntive
st.expander("Informazioni sul Modello").write("""
**Tecnologie utilizzate:**
- Algoritmo K-Means per la clusterizzazione
- StandardScaler per la normalizzazione dei dati
- Streamlit per l'interfaccia web interattiva

**Parametri del modello:**
- Numero di cluster: 4
- Dati sintetici generati per simulare comportamenti realistici
- Modello addestrato su 200 osservazioni
""")
