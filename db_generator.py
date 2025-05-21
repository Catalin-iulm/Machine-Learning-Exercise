import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Clustering Marketing", layout="wide")

st.title("🎯 Segmentazione Clienti: K-Means vs DBSCAN")

@st.cache_data

def load_data():
    return pd.read_csv("data/marketing_customers.csv")

df = load_data()
st.sidebar.header("🔧 Parametri Clustering")

# Selezione algoritmo
algorithm = st.sidebar.selectbox("Scegli algoritmo", ["K-Means", "DBSCAN"])

# Parametri
if algorithm == "K-Means":
    n_clusters = st.sidebar.slider("Numero di Cluster (k)", 2, 10, 3)
    model = KMeans(n_clusters=n_clusters, random_state=42)
else:
    eps = st.sidebar.slider("eps (raggio)", 0.1, 5.0, 1.0, step=0.1)
    min_samples = st.sidebar.slider("min_samples", 2, 20, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
labels = model.fit_predict(X_scaled)
df["Cluster"] = labels

# Visualizzazione 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df["Annual_Spending"],
    df["Purchase_Frequency"],
    df["Loyalty_Score"],
    c=df["Cluster"], cmap="tab10", edgecolor="k"
)
ax.set_xlabel("Spesa Annuale")
ax.set_ylabel("Frequenza Acquisti")
ax.set_zlabel("Punteggio Fedeltà")
st.pyplot(fig)

# Output metriche
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = list(labels).count(-1) if -1 in labels else 0

st.markdown(f"""
- **Algoritmo**: {algorithm}  
- **Cluster trovati**: {n_clusters}  
- **Outlier rilevati**: {n_outliers}
""")

st.subheader("📊 Anteprima Dati Segmentati")
st.dataframe(df.head())
