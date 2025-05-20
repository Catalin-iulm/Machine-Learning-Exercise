import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configurazione pagina
st.set_page_config(layout="wide")
st.title("📊 Segmentazione Clienti con K-means")

# Generazione dati fittizi
@st.cache_data
def load_data():
    np.random.seed(42)
    data = {
        'Frequenza': np.random.randint(1, 100, 300),
        'ValoreCarrello': np.random.normal(50, 15, 300).astype(int),
        'Recency': np.random.randint(1, 365, 300)
    }
    return pd.DataFrame(data)

df = load_data()

# Sidebar per parametri
st.sidebar.header("Parametri")
n_clusters = st.sidebar.slider("Numero di cluster", 2, 5, 3)

# Clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Frequenza', 'ValoreCarrello', 'Recency']])
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizzazione
fig, ax = plt.subplots(figsize=(10, 6))
for cluster in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    ax.scatter(
        cluster_data['Frequenza'], 
        cluster_data['ValoreCarrello'], 
        label=f'Cluster {cluster}'
    )
ax.set_xlabel("Frequenza acquisti")
ax.set_ylabel("Valore carrello (€)")
ax.legend()
st.pyplot(fig)

# Mostra dati
st.subheader("Dati clienti")
st.dataframe(df.head())