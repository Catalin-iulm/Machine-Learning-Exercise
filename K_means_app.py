# kmeans_app.py
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. Titolo
st.title("🔍 Il Mio Primo Clustering!")
st.write("Clicca il pulsante per vedere la magia!")

# 2. Controlli a sinistra
with st.sidebar:
    st.header("⚙️ Controlli")
    n_punti = st.slider("Numero punti", 50, 500, 200)
    n_gruppi = st.slider("Numero gruppi", 2, 6, 3)

# 3. Genera dati
X, y = make_blobs(n_samples=n_punti, centers=n_gruppi, random_state=42)

# 4. Mostra dati originali
st.subheader("📦 Dati Originali")
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5)
st.pyplot(fig)

# 5. Pulsante magico
if st.button("🎯 Crea Gruppi!"):
    kmeans = KMeans(n_clusters=n_gruppi)
    gruppi = kmeans.fit_predict(X)
    
    # 6. Mostra risultati
    st.subheader("🌈 Risultati")
    fig2, ax2 = plt.subplots()
    ax2.scatter(X[:, 0], X[:, 1], c=gruppi, cmap='rainbow', alpha=0.6)
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=100, c='black', marker='X')
    st.pyplot(fig2)