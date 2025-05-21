import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Configurazione della pagina
st.set_page_config(layout="wide", page_title="📊 Advanced Customer Segmentation")
st.title("📊 Advanced Electronics Customer Segmentation")

# --- 1. Generazione Dataset Sintetico Migliorato ---
@st.cache_data
def generate_enhanced_electronics_data(num_samples=300):
    """Genera dati più realistici con sovrapposizioni naturali tra cluster."""
    np.random.seed(42)
    
    # Parametri dei cluster con maggiore variabilità
    clusters = {
        "Premium Tech Enthusiasts": {
            "size": 0.25,
            "params": {
                "annual_spend": (2200, 800, 500, 7000),
                "purchase_freq": (3, 1, 1, 6),
                "product_categories": (5, 1.5, 2, 8),
                "tenure": (30, 10, 12, 60)
            }
        },
        "Value Shoppers": {
            "size": 0.35,
            "params": {
                "annual_spend": (800, 400, 200, 2500),
                "purchase_freq": (4, 1.5, 1, 8),
                "product_categories": (3, 1, 1, 6),
                "tenure": (18, 8, 3, 36)
            }
        },
        "Occasional Buyers": {
            "size": 0.25,
            "params": {
                "annual_spend": (300, 150, 50, 800),
                "purchase_freq": (1.5, 0.7, 0, 3),
                "product_categories": (2, 0.8, 1, 4),
                "tenure": (6, 4, 1, 18)
            }
        },
        "New Explorers": {
            "size": 0.15,
            "params": {
                "annual_spend": (100, 50, 20, 300),
                "purchase_freq": (0.8, 0.4, 0, 2),
                "product_categories": (1.5, 0.5, 1, 3),
                "tenure": (3, 2, 1, 6)
            }
        }
    }
    
    data = []
    for cluster_name, cluster_info in clusters.items():
        n_samples = int(num_samples * cluster_info["size"])
        params = cluster_info["params"]
        
        for _ in range(n_samples):
            # Generazione con distribuzione normale + rumore casuale
            row = {
                "Annual_Spend": np.random.normal(*params["annual_spend"][:2]),
                "Purchase_Freq": np.random.normal(*params["purchase_freq"][:2]),
                "Product_Categories": np.random.normal(*params["product_categories"][:2]),
                "Tenure_Months": np.random.normal(*params["tenure"][:2]),
                "True_Cluster": cluster_name
            }
            
            # Applicazione dei limiti (clip) e aggiunta rumore
            row = {
                "Annual_Spend": max(params["annual_spend"][2], min(params["annual_spend"][3], row["Annual_Spend"] + np.random.uniform(-200, 200))),
                "Purchase_Freq": max(params["purchase_freq"][2], min(params["purchase_freq"][3], row["Purchase_Freq"] + np.random.uniform(-0.5, 0.5))),
                "Product_Categories": max(params["product_categories"][2], min(params["product_categories"][3], row["Product_Categories"] + np.random.uniform(-1, 1))),
                "Tenure_Months": max(params["tenure"][2], min(params["tenure"][3], row["Tenure_Months"] + np.random.uniform(-3, 3))),
                "True_Cluster": cluster_name
            }
            data.append(row)
    
    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# --- 2. Funzioni di Supporto Ottimizzate ---
def get_cluster_profiles(centroids, feature_names):
    """Assegna nomi ai cluster basati sulle caratteristiche dominanti."""
    profiles = []
    for i, centroid in enumerate(centroids):
        features_rank = sorted(zip(feature_names, centroid), key=lambda x: -abs(x[1]))
        
        # Identifica le 2 feature più significative
        dominant_features = [f[0] for f in features_rank[:2]]
        profile_name = f"Group {i+1}: " + " & ".join(dominant_features)
        
        profiles.append({
            "id": i,
            "name": profile_name,
            "dominant_features": dominant_features,
            "centroid": centroid
        })
    return profiles

def plot_cluster_evolution(kmeans_model, X, feature_names):
    """Visualizza l'evoluzione dei centroidi durante le iterazioni."""
    if not hasattr(kmeans_model, 'cluster_centers_'):
        return
    
    history = kmeans_model.n_iter_
    centroids_history = kmeans_model.cluster_centers_
    
    fig = px.scatter(
        x=X[:, 0], y=X[:, 1], 
        color=kmeans_model.labels_,
        title=f"Cluster Evolution (Converged in {history} iterations)",
        labels={"x": feature_names[0], "y": feature_names[1]},
        opacity=0.6
    )
    
    # Aggiungi centroidi
    for i, centroid in enumerate(centroids_history):
        fig.add_scatter(
            x=[centroid[0]], y=[centroid[1]], 
            mode='markers',
            marker=dict(size=12, symbol='x', color='black'),
            name=f'Centroid {i+1}'
        )
    
    st.plotly_chart(fig, use_container_width=True)

# --- 3. Interfaccia Streamlit ---
# Sidebar per i parametri
st.sidebar.header("⚙️ Parameters")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 6, 4)
max_iter = st.sidebar.slider("Max Iterations", 10, 300, 100)
selected_features = st.sidebar.multiselect(
    "Features for Clustering",
    options=['Annual_Spend', 'Purchase_Freq', 'Product_Categories', 'Tenure_Months'],
    default=['Annual_Spend', 'Purchase_Freq']
)

# Caricamento dati
data = generate_enhanced_electronics_data()
st.sidebar.download_button("Download Sample Data", data.to_csv(), "customer_data.csv")

# Preprocessing
if len(selected_features) < 2:
    st.warning("Please select at least 2 features for clustering")
    st.stop()

scaler = StandardScaler()
X = scaler.fit_transform(data[selected_features])

# Esecuzione K-Means
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=max_iter, random_state=42)
kmeans.fit(X)
data['Cluster'] = kmeans.labels_

# --- 4. Visualizzazione Risultati ---
tab1, tab2, tab3 = st.tabs(["📊 Clusters", "📈 Metrics", "🧪 New Customer"])

with tab1:
    st.header("Customer Segmentation Results")
    
    # Visualizzazione principale con Plotly
    fig = px.scatter_matrix(
        data,
        dimensions=selected_features,
        color="Cluster",
        hover_name="True_Cluster",
        title="Multidimensional Cluster View"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap delle caratteristiche
    cluster_means = data.groupby('Cluster')[selected_features].mean()
    st.subheader("Cluster Characteristics")
    st.dataframe(
        cluster_means.style.background_gradient(cmap='viridis')
        .format("{:.1f}")
        .highlight_max(axis=0, color='lightgreen')
        .highlight_min(axis=0, color='#FFCCCB')
    )

with tab2:
    st.header("Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Inertia (WCSS)", f"{kmeans.inertia_:.2f}", 
                 help="Sum of squared distances to nearest cluster center")
    
    with col2:
        silhouette = silhouette_score(X, kmeans.labels_)
        st.metric("Silhouette Score", f"{silhouette:.2f}",
                 help="Higher values indicate better separation between clusters")
    
    with col3:
        db_score = davies_bouldin_score(X, kmeans.labels_)
        st.metric("Davies-Bouldin Index", f"{db_score:.2f}",
                 help="Lower values indicate better clustering")
    
    # Mostra l'evoluzione dei cluster
    st.subheader("Cluster Evolution")
    plot_cluster_evolution(kmeans, X, selected_features)

with tab3:
    st.header("Assign New Customer")
    
    with st.form("new_customer_form"):
        cols = st.columns(2)
        new_customer = {}
        
        with cols[0]:
            new_customer['Annual_Spend'] = st.number_input("Annual Spend (€)", min_value=0, value=1000)
            new_customer['Purchase_Freq'] = st.number_input("Purchase Frequency", min_value=0.0, value=2.0, step=0.5)
        
        with cols[1]:
            new_customer['Product_Categories'] = st.number_input("Product Categories", min_value=1, value=3)
            new_customer['Tenure_Months'] = st.number_input("Tenure (Months)", min_value=1, value=12)
        
        submitted = st.form_submit_button("Assign Cluster")
    
    if submitted:
        # Preprocess input
        customer_array = np.array([[new_customer[feat] for feat in selected_features]])
        scaled_input = scaler.transform(customer_array)
        
        # Prediction
        cluster = kmeans.predict(scaled_input)[0]
        distance = np.min(kmeans.transform(scaled_input))
        
        st.success(f"✅ This customer belongs to **Cluster {cluster}**")
        st.info(f"Distance to centroid: {distance:.2f}")
        
        # Show similar customers
        similar_customers = data[data['Cluster'] == cluster].sample(5)
        st.write("Similar customers in this cluster:")
        st.dataframe(similar_customers[selected_features + ['True_Cluster']])

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Advanced Customer Segmentation**  
Using K-Means clustering to identify customer groups  
in electronics e-commerce data.
""")
