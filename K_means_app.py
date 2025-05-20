import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import io
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="Advanced Clustering for Retail Analytics", page_icon="🛒")

# --- Titolo e Introduzione ---
st.title("🛒 Advanced Customer Segmentation for Retail Analytics")
st.markdown("""
This interactive application allows you to explore advanced clustering algorithms on simulated supermarket customer data. 
Discover hidden customer segments and gain actionable insights for targeted marketing strategies.
""")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Experiment Configuration")
    
    st.subheader("1. Data Generation")
    n_samples = st.slider("Number of simulated customers", 500, 10000, 3000, step=100)
    random_state = st.slider("Random seed for reproducibility", 0, 100, 42)
    noise_level = st.slider("Noise level (%)", 0, 30, 5)
    
    st.subheader("2. Algorithm Selection")
    algorithm = st.radio("Clustering Algorithm:", 
                         ["K-Means", "DBSCAN", "Hierarchical"], 
                         index=0)
    
    if algorithm == "K-Means":
        n_clusters = st.slider("Number of clusters (K)", 2, 15, 6)
        init_method = st.selectbox("Initialization method", 
                                 ["k-means++", "random"])
        max_iter = st.slider("Max iterations", 100, 500, 300)
        
    elif algorithm == "DBSCAN":
        eps = st.slider("Epsilon (neighborhood radius)", 0.1, 2.0, 0.5, step=0.05)
        min_samples = st.slider("Minimum samples", 1, 50, 10)
        metric = st.selectbox("Distance metric", 
                            ["euclidean", "cosine", "manhattan"])
        
    elif algorithm == "Hierarchical":
        n_clusters = st.slider("Number of clusters", 2, 15, 6)
        linkage_method = st.selectbox("Linkage method", 
                                    ["ward", "complete", "average", "single"])
        affinity = st.selectbox("Affinity metric", 
                               ["euclidean", "cosine", "manhattan"])
    
    st.subheader("3. Visualization")
    dim_reduction = st.selectbox("Dimensionality reduction", 
                               ["PCA", "t-SNE", "Original Features"])
    plot_engine = st.selectbox("Plotting engine", 
                             ["Matplotlib", "Plotly"])
    
    st.markdown("---")
    if st.button("🔄 Run Analysis"):
        st.experimental_rerun()

# --- Data Generation ---
@st.cache_data
def generate_retail_data(n_samples, noise_level, random_state):
    np.random.seed(random_state)
    
    # Define 6 customer archetypes with more realistic retail behaviors
    archetypes = {
        "Young Urban Professionals": {
            "age": (28, 5), "income": (55000, 12000), 
            "online_visits": (18, 5), "basket_size": (45, 10),
            "organic_pct": (0.35, 0.1), "discount_sensitivity": (0.6, 0.15),
            "store_visits": (4, 2), "brand_loyalty": (0.7, 0.1)
        },
        "Budget Families": {
            "age": (38, 6), "income": (45000, 8000),
            "online_visits": (8, 3), "basket_size": (75, 15),
            "organic_pct": (0.15, 0.08), "discount_sensitivity": (0.9, 0.05),
            "store_visits": (12, 3), "brand_loyalty": (0.4, 0.15)
        },
        "Premium Shoppers": {
            "age": (45, 8), "income": (95000, 20000),
            "online_visits": (12, 4), "basket_size": (120, 25),
            "organic_pct": (0.5, 0.15), "discount_sensitivity": (0.3, 0.1),
            "store_visits": (6, 2), "brand_loyalty": (0.85, 0.08)
        },
        "Retired Couples": {
            "age": (65, 5), "income": (40000, 10000),
            "online_visits": (4, 2), "basket_size": (55, 12),
            "organic_pct": (0.25, 0.1), "discount_sensitivity": (0.7, 0.1),
            "store_visits": (8, 2), "brand_loyalty": (0.6, 0.12)
        },
        "Health Enthusiasts": {
            "age": (35, 7), "income": (60000, 15000),
            "online_visits": (15, 4), "basket_size": (65, 15),
            "organic_pct": (0.75, 0.1), "discount_sensitivity": (0.5, 0.15),
            "store_visits": (6, 2), "brand_loyalty": (0.65, 0.12)
        },
        "Convenience Shoppers": {
            "age": (32, 8), "income": (48000, 10000),
            "online_visits": (25, 6), "basket_size": (30, 8),
            "organic_pct": (0.2, 0.1), "discount_sensitivity": (0.8, 0.1),
            "store_visits": (2, 1), "brand_loyalty": (0.3, 0.15)
        }
    }
    
    data = []
    samples_per_type = n_samples // len(archetypes)
    
    for arch_name, params in archetypes.items():
        n = samples_per_type
        
        # Generate core cluster data
        age = np.random.normal(params["age"][0], params["age"][1], n)
        income = np.random.normal(params["income"][0], params["income"][1], n)
        online_visits = np.random.poisson(params["online_visits"][0], n)
        basket_size = np.abs(np.random.normal(params["basket_size"][0], params["basket_size"][1], n))
        organic_pct = np.clip(np.random.normal(params["organic_pct"][0], params["organic_pct"][1], n), 0, 1)
        discount_sensitivity = np.clip(np.random.normal(params["discount_sensitivity"][0], params["discount_sensitivity"][1], n), 0, 1)
        store_visits = np.random.poisson(params["store_visits"][0], n)
        brand_loyalty = np.clip(np.random.normal(params["brand_loyalty"][0], params["brand_loyalty"][1], n), 0, 1)
        
        # Add some noise
        noise_mask = np.random.random(n) < (noise_level/100)
        noise_factor = 1 + np.random.normal(0, 0.5, size=(noise_mask.sum(), len(params)))
        
        # Apply noise
        if noise_mask.any():
            noisy_data = np.column_stack([age[noise_mask], income[noise_mask], 
                                        online_visits[noise_mask], basket_size[noise_mask],
                                        organic_pct[noise_mask], discount_sensitivity[noise_mask],
                                        store_visits[noise_mask], brand_loyalty[noise_mask]])
            
            noisy_data = noisy_data * noise_factor
            
            age[noise_mask] = noisy_data[:,0]
            income[noise_mask] = noisy_data[:,1]
            online_visits[noise_mask] = np.abs(noisy_data[:,2])
            basket_size[noise_mask] = np.abs(noisy_data[:,3])
            organic_pct[noise_mask] = np.clip(noisy_data[:,4], 0, 1)
            discount_sensitivity[noise_mask] = np.clip(noisy_data[:,5], 0, 1)
            store_visits[noise_mask] = np.abs(noisy_data[:,6])
            brand_loyalty[noise_mask] = np.clip(noisy_data[:,7], 0, 1)
        
        # Create records
        for i in range(n):
            gender = np.random.choice(["Male", "Female"], p=[0.45, 0.55])
            member_card = np.random.choice([True, False], p=[0.7, 0.3])
            
            data.append([
                max(18, min(80, int(age[i]))),
                gender,
                max(20000, min(200000, int(income[i]))),
                max(0, int(online_visits[i])),
                max(10, float(basket_size[i])),
                float(organic_pct[i]),
                float(discount_sensitivity[i]),
                max(0, int(store_visits[i])),
                float(brand_loyalty[i]),
                member_card,
                arch_name  # True segment for evaluation
            ])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        "Age", "Gender", "Annual Income ($)", 
        "Monthly Online Visits", "Avg Basket Size ($)",
        "Organic Purchase %", "Discount Sensitivity",
        "Monthly Store Visits", "Brand Loyalty Score",
        "Loyalty Card Member", "True Segment"
    ])
    
    # Shuffle data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Prepare features for clustering
    features = [
        "Age", "Annual Income ($)", "Monthly Online Visits", 
        "Avg Basket Size ($)", "Organic Purchase %", 
        "Discount Sensitivity", "Monthly Store Visits", 
        "Brand Loyalty Score"
    ]
    
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled, features

# Generate data
df, X_scaled, features = generate_retail_data(n_samples, noise_level, random_state)

# --- Dimensionality Reduction ---
@st.cache_data
def reduce_dimensions(X, method, random_state):
    if method == "PCA":
        reducer = PCA(n_components=2, random_state=random_state)
        reduced = reducer.fit_transform(X)
        explained_var = reducer.explained_variance_ratio_.sum() * 100
        return reduced, f"PCA (Variance Explained: {explained_var:.1f}%)"
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=30)
        reduced = reducer.fit_transform(X)
        return reduced, "t-SNE"
    else:
        # Use first two original features
        return X[:, :2], "Original Features (First 2)"

X_reduced, reduction_method = reduce_dimensions(X_scaled, dim_reduction, random_state)

# --- Clustering ---
@st.cache_data
def perform_clustering(X, algorithm, params, random_state):
    if algorithm == "K-Means":
        model = KMeans(
            n_clusters=params['n_clusters'],
            init=params['init_method'],
            max_iter=params['max_iter'],
            random_state=random_state,
            n_init='auto'
        )
        labels = model.fit_predict(X)
        centers = model.cluster_centers_ if hasattr(model, 'cluster_centers_') else None
        
    elif algorithm == "DBSCAN":
        model = DBSCAN(
            eps=params['eps'],
            min_samples=params['min_samples'],
            metric=params['metric']
        )
        labels = model.fit_predict(X)
        centers = None
        
    elif algorithm == "Hierarchical":
        model = AgglomerativeClustering(
            n_clusters=params['n_clusters'],
            affinity=params['affinity'],
            linkage=params['linkage_method']
        )
        labels = model.fit_predict(X)
        centers = None
    
    return labels, model, centers

# Prepare algorithm parameters
algo_params = {
    "K-Means": {
        'n_clusters': n_clusters,
        'init_method': init_method,
        'max_iter': max_iter
    },
    "DBSCAN": {
        'eps': eps,
        'min_samples': min_samples,
        'metric': metric
    },
    "Hierarchical": {
        'n_clusters': n_clusters,
        'affinity': affinity,
        'linkage_method': linkage_method
    }
}

labels, model, centers = perform_clustering(
    X_scaled, 
    algorithm, 
    algo_params[algorithm], 
    random_state
)

# --- Evaluation Metrics ---
def calculate_metrics(X, labels):
    metrics = {}
    
    if len(set(labels)) > 1:
        metrics['Silhouette Score'] = silhouette_score(X, labels)
        metrics['Davies-Bouldin Index'] = davies_bouldin_score(X, labels)
        metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(X, labels)
    else:
        metrics['Silhouette Score'] = None
        metrics['Davies-Bouldin Index'] = None
        metrics['Calinski-Harabasz Index'] = None
    
    # Cluster counts
    unique, counts = np.unique(labels, return_counts=True)
    metrics['Cluster Distribution'] = dict(zip(unique, counts))
    metrics['Number of Clusters'] = len(unique) - (1 if -1 in unique else 0)
    metrics['Noise Points'] = counts[unique == -1][0] if -1 in unique else 0
    
    return metrics

metrics = calculate_metrics(X_scaled, labels)

# --- Visualization ---
def create_cluster_plot(X, labels, centers, method, engine):
    plot_df = pd.DataFrame({
        "x": X[:, 0],
        "y": X[:, 1],
        "cluster": labels,
        "size": df["Avg Basket Size ($)"] / 10  # Scale for visualization
    })
    
    if engine == "Plotly":
        fig = px.scatter(
            plot_df, x="x", y="y", color="cluster",
            size="size", hover_data={
                "x": False, "y": False,
                "Age": df["Age"],
                "Income": df["Annual Income ($)"],
                "Online Visits": df["Monthly Online Visits"],
                "Basket Size": df["Avg Basket Size ($)"]
            },
            title=f"Customer Segments ({method})",
            labels={"x": "Component 1", "y": "Component 2"},
            color_continuous_scale=px.colors.qualitative.Plotly
        )
        
        if centers is not None:
            centers_df = pd.DataFrame({
                "x": centers[:, 0],
                "y": centers[:, 1],
                "cluster": ["Center"] * len(centers)
            })
            fig.add_scatter(
                x=centers_df["x"], y=centers_df["y"],
                mode="markers", marker=dict(size=12, color="black", symbol="x"),
                name="Cluster Centers"
            )
        
        fig.update_layout(
            hovermode="closest",
            legend_title_text='Cluster'
        )
        
        return fig
    
    else:  # Matplotlib
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            X[:, 0], X[:, 1], 
            c=labels, s=df["Avg Basket Size ($)"]/5,
            cmap="viridis", alpha=0.7
        )
        
        if centers is not None:
            plt.scatter(
                centers[:, 0], centers[:, 1],
                marker="X", s=200, c="red", 
                edgecolors="black", linewidths=1.5,
                label="Cluster Centers"
            )
        
        plt.title(f"Customer Segments ({method})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.colorbar(scatter, label="Cluster")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        return plt.gcf()

# Create the plot
cluster_plot = create_cluster_plot(
    X_reduced, labels, 
    centers if centers is not None and dim_reduction != "t-SNE" else None,
    reduction_method,
    plot_engine
)

# --- Cluster Profiling ---
def profile_clusters(df, labels, features):
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels
    
    # Numeric summary
    cluster_profiles = df_clustered.groupby("Cluster")[features].agg(
        ["mean", "median", "std", "count"]
    )
    
    # Categorical summary
    if "Gender" in df.columns:
        gender_dist = pd.crosstab(df_clustered["Cluster"], df_clustered["Gender"])
        gender_dist_pct = gender_dist.div(gender_dist.sum(1), axis=0)
        cluster_profiles = pd.concat([cluster_profiles, gender_dist_pct], axis=1)
    
    if "Loyalty Card Member" in df.columns:
        loyalty_dist = pd.crosstab(df_clustered["Cluster"], df_clustered["Loyalty Card Member"])
        loyalty_dist_pct = loyalty_dist.div(loyalty_dist.sum(1), axis=0)
        cluster_profiles = pd.concat([cluster_profiles, loyalty_dist_pct], axis=1)
    
    return cluster_profiles, df_clustered

cluster_profiles, df_clustered = profile_clusters(df, labels, features)

# --- True Segment Comparison ---
def compare_true_segments(df_clustered):
    if "True Segment" not in df_clustered.columns:
        return None
    
    comparison = pd.crosstab(
        df_clustered["Cluster"], 
        df_clustered["True Segment"],
        normalize="index"
    )
    
    purity = comparison.max(axis=1).mean()
    
    return comparison, purity

true_segment_comparison, purity_score = compare_true_segments(df_clustered)

# --- Main Display ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visualization", 
    "📈 Cluster Analysis",
    "📋 Data Explorer",
    "📚 Documentation"
])

with tab1:
    st.header("Customer Segmentation Visualization")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if plot_engine == "Plotly":
            st.plotly_chart(cluster_plot, use_container_width=True)
        else:
            st.pyplot(cluster_plot)
            plt.close()
    
    with col2:
        st.subheader("Algorithm Metrics")
        
        st.metric("Number of Clusters", metrics["Number of Clusters"])
        st.metric("Noise Points", metrics["Noise Points"])
        
        if metrics['Silhouette Score'] is not None:
            st.metric("Silhouette Score", f"{metrics['Silhouette Score']:.3f}",
                     help="Higher values indicate better defined clusters (-1 to 1)")
        
        if metrics['Davies-Bouldin Index'] is not None:
            st.metric("Davies-Bouldin Index", f"{metrics['Davies-Bouldin Index']:.3f}",
                     help="Lower values indicate better clustering (0 to ∞)")
        
        if metrics['Calinski-Harabasz Index'] is not None:
            st.metric("Calinski-Harabasz", f"{metrics['Calinski-Harabasz Index']:.3f}",
                     help="Higher values indicate better clustering (0 to ∞)")
        
        if true_segment_comparison is not None:
            st.metric("Segment Purity", f"{purity_score:.1%}",
                     help="How well clusters match true segments")

with tab2:
    st.header("Cluster Analysis")
    
    st.subheader("Cluster Characteristics")
    st.dataframe(cluster_profiles.style.background_gradient(cmap="Blues"))
    
    st.subheader("Feature Distributions by Cluster")
    selected_feature = st.selectbox("Select feature to visualize", features)
    
    if plot_engine == "Plotly":
        fig = px.box(df_clustered, x="Cluster", y=selected_feature, color="Cluster")
        st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(10, 5))
        df_clustered.boxplot(column=selected_feature, by="Cluster", grid=False)
        plt.title(f"Distribution of {selected_feature} by Cluster")
        plt.suptitle("")
        st.pyplot(plt.gcf())
        plt.close()
    
    if true_segment_comparison is not None:
        st.subheader("True Segment Comparison")
        st.dataframe(true_segment_comparison.style.background_gradient(cmap="Greens", axis=1))
        
        fig = px.imshow(true_segment_comparison, text_auto=".1%")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Data Explorer")
    
    st.subheader("Raw Data with Cluster Assignments")
    st.dataframe(df_clustered)
    
    # Download button
    csv = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download clustered data as CSV",
        data=csv,
        file_name="retail_customers_clustered.csv",
        mime="text/csv"
    )
    
    st.subheader("Feature Correlations")
    corr_matrix = df_clustered[features].corr()
    
    if plot_engine == "Plotly":
        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu")
        st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(features)), features, rotation=45)
        plt.yticks(range(len(features)), features)
        for i in range(len(features)):
            for j in range(len(features)):
                plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                        ha="center", va="center", color="black")
        st.pyplot(plt.gcf())
        plt.close()

with tab4:
    st.header("Documentation")
    
    st.subheader("About This Tool")
    st.markdown("""
    This interactive tool allows retail analysts to:
    - Explore customer segmentation using different algorithms
    - Compare algorithm performance using multiple metrics
    - Profile customer segments based on their characteristics
    - Discover hidden patterns in customer behavior
    
    **Key Features:**
    - Three clustering algorithms (K-Means, DBSCAN, Hierarchical)
    - Multiple dimensionality reduction techniques
    - Interactive visualizations with Plotly
    - Comprehensive cluster profiling
    - Evaluation against ground truth (when available)
    """)
    
    st.subheader("Feature Descriptions")
    feature_descriptions = {
        "Age": "Customer age in years",
        "Annual Income ($)": "Estimated annual household income",
        "Monthly Online Visits": "Number of visits to online store/app per month",
        "Avg Basket Size ($)": "Average spending per shopping trip",
        "Organic Purchase %": "Percentage of purchases that are organic products",
        "Discount Sensitivity": "Likelihood to respond to discounts (0-1 scale)",
        "Monthly Store Visits": "Number of physical store visits per month",
        "Brand Loyalty Score": "Preference for branded vs private label (0-1 scale)",
        "Loyalty Card Member": "Whether customer has loyalty card",
        "True Segment": "Ground truth segment (only in simulated data)"
    }
    
    for feat, desc in feature_descriptions.items():
        st.markdown(f"**{feat}**: {desc}")
    
    st.subheader("Algorithm Guidance")
    st.markdown("""
    **K-Means**:
    - Best for spherical clusters of similar size
    - Requires specifying number of clusters
    - Sensitive to outliers
    
    **DBSCAN**:
    - Finds clusters of arbitrary shape
    - Identifies noise points
    - Requires tuning epsilon and min_samples
    
    **Hierarchical**:
    - Creates nested cluster hierarchy
    - Can use different linkage methods
    - Useful for understanding data structure
    """)

# --- Footer ---
st.markdown("---")
st.markdown("""
    *Advanced Retail Analytics Tool* | Created with Streamlit | 
    [GitHub Repository](https://github.com/your-repo)
""")
