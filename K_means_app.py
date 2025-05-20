import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from io import BytesIO

# Configurazione pagina
st.set_page_config(page_title="Clustering Retail", layout="wide", page_icon="🛒")

# Titolo semplice
st.title("🛒 Strumento di Clustering per Dati Retail")
st.markdown("""
**Facile da usare:** seleziona le opzioni a sinistra e clicca "Esegui Clustering" per vedere i risultati.
""")

# Generazione dati simulati
@st.cache_data
def generate_retail_data():
    np.random.seed(42)
    n = 500  # Meno dati per maggiore velocità
    
    data = {
        "Età": np.round(np.clip(np.random.normal(45, 15, n), 18, 80),
        "Reddito (k€)": np.round(np.clip(np.random.normal(50, 20, n), 20, 150)),
        "Visite/Mese": np.random.poisson(8, n),
        "Spesa Media (€)": np.round(np.clip(np.random.normal(80, 30, n), 10, 200)),
        "% Online": np.round(np.clip(np.random.beta(2, 5, n), 0, 1)
    }
    
    return pd.DataFrame(data)

df = generate_retail_data()

# Sidebar semplificata
with st.sidebar:
    st.header("🔧 Impostazioni")
    
    # 1. Selezione algoritmo
    algo_choice = st.radio(
        "Scegli algoritmo:",
        ["K-Means", "DBSCAN"],
        help="K-Means: per gruppi ben separati\nDBSCAN: per forme complesse"
    )
    
    # 2. Selezione colonne
    selected_features = st.multiselect(
        "Seleziona 2-3 colonne:",
        df.columns,
        default=["Età", "Reddito (k€)"]
    )
    
    if len(selected_features) < 2:
        st.warning("Seleziona almeno 2 colonne!")
        st.stop()
    
    # 3. Parametri specifici
    if algo_choice == "K-Means":
        n_clusters = st.slider("Numero di gruppi:", 2, 5, 3)
    else:
        eps = st.slider("Distanza massima (ε):", 0.1, 1.0, 0.5)
        min_samples = st.slider("Punti minimi:", 2, 10, 5)
    
    # Pulsante di esecuzione
    if st.button("▶️ Esegui Clustering", type="primary"):
        pass  # La logica verrà eseguita dopo

# Sezione 1: Anteprima dati
st.header("📋 Anteprima Dati")
st.write("Ecco le prime 10 righe del dataset:")
st.dataframe(df.head(10))

# Sezione 2: Esecuzione clustering (solo se premuto il pulsante)
if st.sidebar.button:
    st.header("🎯 Risultati Clustering")
    
    with st.spinner('Sto analizzando i dati...'):
        # Prepara i dati
        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Esegui clustering
        if algo_choice == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            model = DBSCAN(eps=eps, min_samples=min_samples)
        
        labels = model.fit_predict(X_scaled)
        
        # Riduci dimensioni per visualizzazione
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        
        # Crea dataframe per plotting
        plot_df = pd.DataFrame({
            'Asse X': X_2d[:, 0],
            'Asse Y': X_2d[:, 1],
            'Gruppo': labels.astype(str)
        })
        
        # Aggiungi tooltip con valori originali
        for feat in selected_features:
            plot_df[feat] = df[feat]
    
    # Visualizzazione risultati
    st.subheader("Mappa dei Gruppi")
    
    fig = px.scatter(
        plot_df,
        x='Asse X',
        y='Asse Y',
        color='Gruppo',
        hover_data=selected_features,
        title=f"Risultato Clustering con {algo_choice}",
        width=800,
        height=500
    )
    
    st.plotly_chart(fig)
    
    # Informazioni sui gruppi
    st.subheader("Informazioni sui Gruppi")
    
    df['Gruppo'] = labels
    group_stats = df.groupby('Gruppo')[selected_features].mean()
    
    st.write("Media delle caratteristiche per gruppo:")
    st.dataframe(group_stats.style.background_gradient())
    
    # Suggerimenti interpretazione
    st.info("""
    **Come interpretare i risultati:**
    - I punti dello stesso colore appartengono allo stesso gruppo
    - Guarda la tabella sopra per vedere le caratteristiche medie di ogni gruppo
    - I gruppi dovrebbero essere ben separati nel grafico
    """)

# Sezione 3: Guida
st.header("ℹ️ Guida Rapida")

with st.expander("Come usare questo strumento"):
    st.markdown("""
    1. **Seleziona colonne**: Scegli 2-3 caratteristiche da analizzare (es. Età e Reddito)
    2. **Scegli algoritmo**: 
       - K-Means per gruppi semplici e rotondi
       - DBSCAN per forme più complesse
    3. **Clicca 'Esegui Clustering'**
    4. **Interpreta i risultati**:
       - Ogni colore è un gruppo diverso
       - I punti vicini sono simili tra loro
    """)

with st.expander("Esempi pratici"):
    st.markdown("""
    **Segmenti tipici che potresti trovare:**
    - 👵🔵 Anziani con reddito medio-basso
    - 👨💎 Giovani professionisti con alto reddito
    - 👩🛒 Donne che fanno molti acquisti
    - 🧒📱 Giovani che comprano online
    """)

# Footer
st.markdown("---")
st.markdown("Strumento didattico per corsi di Data Science - © 2023")
