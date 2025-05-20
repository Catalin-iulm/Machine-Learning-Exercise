import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from io import BytesIO
import time
from sklearn.neighbors import NearestNeighbors

# Inizializzazione session state
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False
    st.session_state.current_algo = None
    st.session_state.X_red = None
    st.session_state.labels = None

# Configurazione pagina
st.set_page_config(page_title="Clustering Retail", layout="wide", page_icon="🛒")

# Titolo con descrizione ampliata
st.title("🛒 Laboratorio di Clustering per Retail")
st.markdown("""
**Tool interattivo** per esplorare algoritmi di clustering su dati retail simulati. 
Scopri come K-Means e DBSCAN possono identificare diversi segmenti di clientela.
""")

# Sidebar riorganizzata
with st.sidebar:
    st.header("⚙️ Configurazione Base")
    
    # Selezione algoritmo con descrizione
    algo_choice = st.radio(
        "**Selezione algoritmo**",
        ["K-Means", "DBSCAN"],
        help="K-Means: per cluster sferici di dimensioni simili\nDBSCAN: per cluster di forma arbitraria e rilevamento outlier"
    )
    
    # Selezione features con validazione
    available_features = [
        "Età", "Reddito Annuale", "Visite Mensili",
        "Spesa Media", "% Acquisti Online",
        "Frequenza Acquisti", "Valore Carrello"
    ]
    selected_features = st.multiselect(
        "**Caratteristiche per clustering** (max 5):",
        available_features,
        default=["Età", "Reddito Annuale", "Visite Mensili"],
        max_selections=5
    )
    
    # Controllo numero features
    if len(selected_features) < 2:
        st.warning("Seleziona almeno 2 caratteristiche!")
        st.stop()
    
    # Sezione avanzata con expander
    with st.expander("⚙️ Parametri Avanzati"):
        dim_reduction = st.selectbox(
            "Metodo riduzione dimensionale:",
            ["PCA", "t-SNE"],
            help="PCA: più veloce, mantiene varianza\nt-SNE: migliore per visualizzazione ma più lento"
        )
        
        if algo_choice == "K-Means":
            n_clusters = st.slider(
                "Numero cluster:",
                2, 10, 4,
                help="Numero di gruppi da identificare"
            )
            max_iter = st.slider(
                "Massime iterazioni:",
                1, 50, 10
            )
            random_seed = st.number_input(
                "Seed casuale:",
                min_value=0, value=42
            )
            
            show_evolution = st.checkbox(
                "Mostra evoluzione iterazioni",
                help="Visualizza come cambiano i cluster con più iterazioni"
            )
            
        elif algo_choice == "DBSCAN":
            eps = st.slider(
                "Raggio (ε):", 
                0.1, 2.0, 0.5, step=0.05,
                help="Distanza massima tra due punti per essere considerati vicini"
            )
            min_samples = st.slider(
                "Minimo campioni:",
                2, 20, 5,
                help="Numero minimo di punti per formare un cluster denso"
            )
            
            # Plot K-distance per aiutare a scegliere eps
            if st.checkbox("Mostra grafico k-distance"):
                with st.spinner('Calcolando k-distance...'):
                    neigh = NearestNeighbors(n_neighbors=min_samples)
                    nbrs = neigh.fit(X)
                    distances, _ = nbrs.kneighbors(X)
                    k_dist = np.sort(distances[:, min_samples-1])
                    
                    fig, ax = plt.subplots()
                    ax.plot(k_dist)
                    ax.axhline(y=eps, color='r', linestyle='--')
                    ax.set_title(f'K-Distance (k={min_samples})')
                    ax.set_xlabel('Punti')
                    ax.set_ylabel(f'{min_samples}-distanza')
                    st.pyplot(fig)
                    st.info("Scegli ε dove c'è un 'gomito' nel grafico (sopra la linea rossa).")

    # Pulsante di esecuzione con gestione stato
    if st.button("🎯 Esegui Clustering", type="primary"):
        st.session_state.clustering_done = True
        st.session_state.current_algo = algo_choice
        st.experimental_rerun()

# Generazione dati simulati con cache
@st.cache_data
def generate_retail_data():
    np.random.seed(42)
    n = 1500
    
    # Generazione dati più realistici con correlazioni
    data = {
        "Età": np.round(np.clip(np.random.normal(45, 15, n), 18, 80)),
        "Reddito Annuale": np.clip(np.random.lognormal(10.5, 0.4, n), 20000, 200000),
        "Visite Mensili": np.random.poisson(8, n),
        "Spesa Media": np.clip(np.random.weibull(1.5, n)*100, 10, 500),
        "% Acquisti Online": np.clip(np.random.beta(2, 5, n), 0, 1),
        "Frequenza Acquisti": np.random.poisson(6, n),
        "Valore Carrello": np.clip(np.random.normal(80, 30, n), 10, 300),
        "Genere": np.random.choice(["M", "F"], size=n, p=[0.45, 0.55]),
        "Cluster Reale": np.random.choice([1, 2, 3, 4], size=n, p=[0.3, 0.4, 0.2, 0.1])
    }
    
    # Aggiungi correlazioni
    data["Spesa Media"] = data["Spesa Media"] * (1 + data["% Acquisti Online"] * 0.5)
    data["Valore Carrello"] = data["Valore Carrello"] * (data["Reddito Annuale"] / 100000)
    
    return pd.DataFrame(data)

# Caricamento dati con progress bar
with st.spinner('Generazione dati simulati...'):
    df = generate_retail_data()

# Preelaborazione dati
try:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[selected_features])
except Exception as e:
    st.error(f"Errore nel preprocessing: {str(e)}")
    st.stop()

# Riduzione dimensionale con caching
@st.cache_data
def reduce_dimensionality(_X, method, random_state=42):
    if method == "PCA":
        reducer = PCA(n_components=2, random_state=random_state)
        X_red = reducer.fit_transform(_X)
        var_exp = reducer.explained_variance_ratio_.sum()
        return X_red, f"PCA (Varianza spiegata: {var_exp:.1%})"
    else:
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=30)
        X_red = reducer.fit_transform(_X)
        return X_red, "t-SNE"

# Esecuzione clustering solo quando richiesto
if st.session_state.clustering_done:
    with st.spinner('Esecuzione clustering...'):
        try:
            # Riduzione dimensionale
            X_red, reduction_method = reduce_dimensionality(X, dim_reduction)
            st.session_state.X_red = X_red
            
            # Esecuzione algoritmo selezionato
            if algo_choice == "K-Means":
                model = KMeans(
                    n_clusters=n_clusters,
                    max_iter=max_iter,
                    n_init='auto',
                    random_state=random_seed
                )
                labels = model.fit_predict(X)
                centers = scaler.inverse_transform(model.cluster_centers_)
                
                # Visualizzazione evoluzione K-Means
                if show_evolution:
                    st.subheader("Evoluzione K-Means")
                    cols = st.columns(min(3, max_iter))
                    
                    for i, col in enumerate(cols):
                        iter_num = (i+1)*(max_iter//len(cols))
                        with col:
                            temp_model = KMeans(
                                n_clusters=n_clusters,
                                max_iter=iter_num,
                                n_init=1,
                                init='random',
                                random_state=random_seed
                            )
                            temp_labels = temp_model.fit_predict(X)
                            
                            fig = px.scatter(
                                x=X_red[:, 0], y=X_red[:, 1],
                                color=temp_labels.astype(str),
                                title=f"Iterazione {iter_num}",
                                labels={'x': 'Componente 1', 'y': 'Componente 2'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            elif algo_choice == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X)
                
                # Analisi risultati DBSCAN
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                n_noise = np.sum(labels == -1)
                
                st.info(f"""
                DBSCAN ha identificato:
                - {n_clusters} cluster
                - {n_noise} punti di rumore ({(n_noise/len(labels)):.1%})
                """)
            
            st.session_state.labels = labels
            st.session_state.model = model
            
        except Exception as e:
            st.error(f"Errore durante il clustering: {str(e)}")
            st.session_state.clustering_done = False

# Visualizzazione risultati
if st.session_state.clustering_done:
    tab1, tab2, tab3 = st.tabs(["📊 Risultati", "📈 Analisi", "🎓 Guida"])
    
    with tab1:
        st.header("Visualizzazione Risultati")
        
        # Creazione dataframe per plotting
        plot_df = pd.DataFrame({
            'Componente 1': st.session_state.X_red[:, 0],
            'Componente 2': st.session_state.X_red[:, 1],
            'Cluster': st.session_state.labels.astype(str)
        })
        
        # Aggiunta features originali per tooltip
        for feat in selected_features:
            plot_df[feat] = df[feat]
        
        # Palette accessibile
        color_discrete_map = {
            '-1': 'gray', '0': '#636EFA', '1': '#EF553B', 
            '2': '#00CC96', '3': '#AB63FA', '4': '#FFA15A'
        }
        
        # Plot interattivo
        fig = px.scatter(
            plot_df,
            x='Componente 1',
            y='Componente 2',
            color='Cluster',
            title=f"Clustering con {algo_choice} ({reduction_method})",
            hover_data=selected_features,
            color_discrete_map=color_discrete_map,
            category_orders={"Cluster": sorted(plot_df['Cluster'].unique())}
        )
        
        # Aggiungi centroidi per K-Means
        if algo_choice == "K-Means":
            centers_red = reduce_dimensionality(
                scaler.transform(centers), 
                dim_reduction
            )[0]
            
            fig.add_scatter(
                x=centers_red[:, 0],
                y=centers_red[:, 1],
                mode='markers',
                marker=dict(
                    size=12,
                    color='black',
                    symbol='x',
                    line=dict(width=2)
                ),
                name='Centroidi'
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metriche di valutazione
        st.subheader("Valutazione Clustering")
        
        if len(np.unique(st.session_state.labels)) > 1:
            try:
                silhouette = silhouette_score(X, st.session_state.labels)
                st.metric(
                    "Silhouette Score",
                    f"{silhouette:.3f}",
                    help="Valore tra -1 (peggiore) e 1 (migliore)"
                )
            except:
                st.warning("Impossibile calcolare Silhouette Score")
        
        # Confronto con ground truth (se disponibile)
        if 'Cluster Reale' in df.columns:
            st.subheader("Confronto con Segmentazione Reale")
            fig_comp = px.scatter(
                x=st.session_state.X_red[:, 0],
                y=st.session_state.X_red[:, 1],
                color=df['Cluster Reale'].astype(str),
                title="Segmentazione Reale (Ground Truth)"
            )
            st.plotly_chart(fig_comp, use_container_width=True)
    
    with tab2:
        st.header("Analisi Cluster")
        
        df['Cluster'] = st.session_state.labels
        
        # Statistiche descrittive
        st.subheader("Statistiche per Cluster")
        cluster_stats = df.groupby('Cluster')[selected_features].agg(['mean', 'std'])
        st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))
        
        # Distribuzione features
        st.subheader("Distribuzione Caratteristiche")
        feat = st.selectbox("Seleziona caratteristica:", selected_features)
        
        fig_dist = px.box(
            df[df['Cluster'] != -1] if -1 in df['Cluster'].values else df,
            x='Cluster',
            y=feat,
            color='Cluster',
            color_discrete_map=color_discrete_map
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Heatmap correlazioni
        st.subheader("Correlazioni tra Caratteristiche")
        corr_matrix = df[selected_features].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale='RdBu',
            range_color=[-1, 1]
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.header("Guida Didattica")
        
        with st.expander("📚 Teoria Clustering"):
            st.markdown("""
            **Clustering** è una tecnica di apprendimento non supervisionato che raggruppa dati simili.
            
            ### K-Means
            - Divide i dati in K cluster sferici
            - Ogni punto appartiene al cluster con centroide più vicino
            - Sensibile a outlier e richiede numero cluster a priori
            
            ### DBSCAN
            - Identifica cluster come aree dense
            - Trova cluster di forma arbitraria
            - Rileva punti di rumore (outlier)
            """)
        
        with st.expander("🔍 Interpretazione Risultati"):
            st.markdown("""
            1. **Silhouette Score**: 
               - > 0.5: Struttura cluster forte
               - ~0: Cluster sovrapposti
               - < 0: Punti potrebbero essere in cluster sbagliati
            
            2. **Visualizzazione**:
               - Punti vicini nello spazio ridotto sono simili
               - Cluster ben separati indicano segmenti distinti
            
            3. **Statistiche Cluster**:
               - Confronta medie e distribuzioni per identificare pattern
            """)
        
        with st.expander("💡 Esempi Pratici"):
            st.markdown("""
            **Segmenti tipici retail**:
            - 🛍️ Clienti frequenti a basso spendimento
            - 💎 Acquirenti premium occasionali 
            - 🏠 Famiglie con spesa media costante
            - 🎯 Giovani acquirenti online
            """)

# Footer
st.markdown("---")
st.markdown("""
**Laboratorio di Clustering** - Università IULM  
Strumento didattico per il corso di Machine Learning  
[Documentazione completa](https://github.com/) | [Dataset](https://github.com/)
""")
