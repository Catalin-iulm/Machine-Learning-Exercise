import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors # Not strictly needed if using named cmap

# --- 1. Dataset Sintetico ---
def generate_gym_data(num_samples_per_group=60):
    """Generates a synthetic dataset for gym client segmentation."""
    np.random.seed(42) # For reproducible synthetic data
    all_data = []
    
    # Group 1: Potential "Neofiti Entusiasti"
    n1 = num_samples_per_group
    g1 = pd.DataFrame({
        'Frequenza_Settimanale': np.random.normal(loc=3.5, scale=1.0, size=n1).clip(1, 7),
        'Durata_Allenamento_Min': np.random.normal(loc=50, scale=10, size=n1).clip(30, 90),
        'Obiettivo_Principale_encoded': np.random.choice([0, 2], size=n1, p=[0.6, 0.4]), # 0: Dimagrimento, 2: Benessere
        'Anni_Esperienza_Fitness': np.random.normal(loc=0.5, scale=0.5, size=n1).clip(0, 2),
        'Eta': np.random.normal(loc=28, scale=7, size=n1).clip(18, 50),
        'Uso_Servizi_Extra_encoded': np.random.choice([0, 1], size=n1, p=[0.6, 0.4])
    })
    all_data.append(g1)

    # Group 2: Potential "Veterani Dedicati"
    n2 = num_samples_per_group
    g2 = pd.DataFrame({
        'Frequenza_Settimanale': np.random.normal(loc=5.5, scale=1.0, size=n2).clip(3, 7),
        'Durata_Allenamento_Min': np.random.normal(loc=100, scale=20, size=n2).clip(60, 180),
        'Obiettivo_Principale_encoded': np.random.choice([1, 2], size=n2, p=[0.7, 0.3]), # 1: Massa, 2: Benessere
        'Anni_Esperienza_Fitness': np.random.normal(loc=8, scale=3, size=n2).clip(4, 20),
        'Eta': np.random.normal(loc=35, scale=8, size=n2).clip(22, 60),
        'Uso_Servizi_Extra_encoded': np.random.choice([0, 1], size=n2, p=[0.8, 0.2])
    })
    all_data.append(g2)

    # Group 3: Potential "Clienti Occasionali"
    n3 = num_samples_per_group
    g3 = pd.DataFrame({
        'Frequenza_Settimanale': np.random.normal(loc=1.5, scale=0.7, size=n3).clip(0, 3),
        'Durata_Allenamento_Min': np.random.normal(loc=45, scale=15, size=n3).clip(20, 75),
        'Obiettivo_Principale_encoded': np.random.choice([0, 1, 2], size=n3, p=[0.3, 0.1, 0.6]),
        'Anni_Esperienza_Fitness': np.random.normal(loc=2, scale=2, size=n3).clip(0, 10),
        'Eta': np.random.normal(loc=45, scale=12, size=n3).clip(16, 70),
        'Uso_Servizi_Extra_encoded': np.random.choice([0, 1], size=n3, p=[0.9, 0.1])
    })
    all_data.append(g3)
    
    # Group 4: Potential "Amanti dei Corsi"
    n4 = num_samples_per_group
    g4 = pd.DataFrame({
        'Frequenza_Settimanale': np.random.normal(loc=3, scale=1.0, size=n4).clip(2, 5),
        'Durata_Allenamento_Min': np.random.normal(loc=60, scale=10, size=n4).clip(45, 75),
        'Obiettivo_Principale_encoded': np.random.choice([0, 2], size=n4, p=[0.4, 0.6]),
        'Anni_Esperienza_Fitness': np.random.normal(loc=2.5, scale=1.5, size=n4).clip(0, 8),
        'Eta': np.random.normal(loc=38, scale=10, size=n4).clip(20, 65),
        'Uso_Servizi_Extra_encoded': np.random.choice([0, 1], size=n4, p=[0.1, 0.9]) # High for services (courses)
    })
    all_data.append(g4)

    gym_df = pd.concat(all_data, ignore_index=True)
    gym_df = gym_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
    return gym_df

# --- 4. Etichette Significative per i Cluster ---
def get_cluster_profile_names(centroids_df, k_clusters, feature_cols_list):
    """Assigns meaningful names to clusters based on centroid characteristics."""
    # feature_cols_list = ['Frequenza_Settimanale', 'Durata_Allenamento_Min', 'Obiettivo_Principale_encoded', 
    #                      'Anni_Esperienza_Fitness', 'Eta', 'Uso_Servizi_Extra_encoded']
    
    assigned_names_map = {}
    # Define profiles based on expected characteristics
    profile_definitions = {
        "Neofiti Entusiasti": {'Anni_Esperienza_Fitness': ('<', 2), 'Frequenza_Settimanale': ('>=', 2.5)},
        "Veterani Dedicati": {'Anni_Esperienza_Fitness': ('>', 4), 'Frequenza_Settimanale': ('>=', 3), 'Durata_Allenamento_Min': ('>', 75)},
        "Clienti Occasionali": {'Frequenza_Settimanale': ('<', 2.0)},
        "Amanti dei Corsi": {'Uso_Servizi_Extra_encoded': ('>', 0.5), 'Frequenza_Settimanale': ('>=', 2.5)}
    }
    
    available_names = list(profile_definitions.keys())
    used_names = []

    for i in range(k_clusters):
        centroid = centroids_df.iloc[i]
        best_match_name = f"Gruppo {i+1}" # Default
        
        # Try to match predefined profiles
        found_match = False
        for name, rules in profile_definitions.items():
            if name in used_names:
                continue # Skip if this profile name is already assigned
            
            match = True
            for feature, (op, val) in rules.items():
                feat_idx = feature_cols_list.index(feature)
                if op == '<' and not (centroid.iloc[feat_idx] < val): match = False; break
                if op == '>' and not (centroid.iloc[feat_idx] > val): match = False; break
                if op == '<=' and not (centroid.iloc[feat_idx] <= val): match = False; break
                if op == '>=' and not (centroid.iloc[feat_idx] >= val): match = False; break
            
            if match:
                best_match_name = name
                used_names.append(name)
                found_match = True
                break # Found a good match for this centroid

        if not found_match: # If no predefined profile matched, use a generic one from remaining available_names
            for name in available_names:
                if name not in used_names:
                    best_match_name = name
                    used_names.append(name)
                    break
        
        assigned_names_map[i] = best_match_name
        
    # Ensure all clusters get a name, even if it's generic and repeated if K > len(profile_definitions)
    # This simplified version might assign the same generic name if K is large and specific rules don't cover all.
    # For K=4 and 4 profiles, it should work well if centroids are distinct enough.
    current_names = list(assigned_names_map.values())
    for i in range(k_clusters):
        if i not in assigned_names_map or assigned_names_map[i].startswith("Gruppo"):
            # Try to pick a name not yet used if possible, from a pool
            fallback_pool = ["Profilo Alfa", "Profilo Beta", "Profilo Gamma", "Profilo Delta", "Profilo Epsilon"]
            chosen_fallback = f"Gruppo Generico {i+1}"
            for fb_name in fallback_pool:
                if fb_name not in current_names:
                    chosen_fallback = fb_name
                    current_names.append(fb_name) # Mark as used for next fallback
                    break
            assigned_names_map[i] = chosen_fallback


    return assigned_names_map


# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("🏋️‍♂️ Segmentazione Clienti Palestra con K-Means")

# --- Global Settings & Data Generation ---
K_CLUSTERS = 4 # Fixed number of clusters to align with naming strategy
FEATURE_COLS_FOR_CLUSTERING = ['Frequenza_Settimanale', 'Durata_Allenamento_Min', 
                               'Obiettivo_Principale_encoded', 'Anni_Esperienza_Fitness', 
                               'Eta', 'Uso_Servizi_Extra_encoded']

data_df = generate_gym_data(num_samples_per_group=75) # Generate 4*75 = 300 samples
data_original_df = data_df.copy()

X = data_df[FEATURE_COLS_FOR_CLUSTERING].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --- Sidebar Controls ---
st.sidebar.header("⚙️ Parametri K-Means")
chosen_max_iterations = st.sidebar.slider("Numero Totale di Iterazioni K-Means", 1, 20, 10,
                                          help="Seleziona il numero massimo di iterazioni per l'algoritmo K-Means.")

st.sidebar.header("📊 Visualizzazione Grafici")
feature_options_display = ['Frequenza_Settimanale', 'Durata_Allenamento_Min', 'Anni_Esperienza_Fitness', 'Eta']
x_axis_feat = st.sidebar.selectbox("Feature per Asse X:", feature_options_display, index=0)
y_axis_feat = st.sidebar.selectbox("Feature per Asse Y:", feature_options_display, index=1)
if x_axis_feat == y_axis_feat: # Ensure different features are chosen for X and Y
    st.sidebar.warning("Seleziona feature diverse per l'asse X e Y per una visualizzazione significativa.")
    st.stop()


# --- 2. Clusterizzazione Dinamica con K-Means (Evoluzione) ---
st.header("🔄 Evoluzione dei Cluster")
st.markdown(f"Visualizzazione dei cluster dopo 1, {chosen_max_iterations // 2 if chosen_max_iterations > 1 else 1}, e {chosen_max_iterations} iterazioni.")

iterations_to_show = sorted(list(set([1] + ([chosen_max_iterations // 2] if chosen_max_iterations > 2 else []) + [chosen_max_iterations])))
if not iterations_to_show or iterations_to_show[-1] == 0 : iterations_to_show = [1]
if len(iterations_to_show) > 1 and iterations_to_show[0] == 0 and len(iterations_to_show) > 1: iterations_to_show = iterations_to_show[1:]
if not iterations_to_show : iterations_to_show = [1] # Failsafe

evolution_cols = st.columns(len(iterations_to_show))

for i, num_iter in enumerate(iterations_to_show):
    with evolution_cols[i]:
        st.subheader(f"Iter: {num_iter}")
        # Use n_init=1 and random_state to ensure k-means++ init is same, then observe effect of max_iter
        kmeans_evol = KMeans(n_clusters=K_CLUSTERS, init='k-means++', n_init=1, max_iter=num_iter, random_state=42)
        kmeans_evol.fit(X_scaled)
        labels_evol = kmeans_evol.labels_
        centroids_evol_scaled = kmeans_evol.cluster_centers_
        centroids_evol_original = scaler.inverse_transform(centroids_evol_scaled)

        plot_df_evol = data_original_df.copy()
        plot_df_evol['Cluster_Temp'] = labels_evol
        
        fig_evol, ax_evol = plt.subplots(figsize=(6, 5)) # Smaller plots for columns
        cmap_evol = plt.cm.get_cmap('viridis', K_CLUSTERS)
        
        scatter_evol = ax_evol.scatter(plot_df_evol[x_axis_feat], plot_df_evol[y_axis_feat], 
                                       c=plot_df_evol['Cluster_Temp'], cmap=cmap_evol, alpha=0.6, s=30, edgecolors='grey', linewidth=0.5)
        
        x_feat_idx_plot = FEATURE_COLS_FOR_CLUSTERING.index(x_axis_feat)
        y_feat_idx_plot = FEATURE_COLS_FOR_CLUSTERING.index(y_axis_feat)
        ax_evol.scatter(centroids_evol_original[:, x_feat_idx_plot], centroids_evol_original[:, y_feat_idx_plot], 
                        marker='X', s=150, color='red', edgecolors='black', label='Centroidi')
        
        ax_evol.set_xlabel(x_axis_feat, fontsize=9)
        ax_evol.set_ylabel(y_axis_feat, fontsize=9)
        ax_evol.set_title(f"Cluster (max {num_iter} iter.)", fontsize=10)
        ax_evol.tick_params(axis='both', which='major', labelsize=8)
        ax_evol.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_evol)

st.markdown("---")

# --- Final Model & Cluster Naming (using chosen_max_iterations) ---
st.header("🏆 Risultato Finale della Clusterizzazione")
# For final model, use more n_init for robustness
kmeans_final = KMeans(n_clusters=K_CLUSTERS, init='k-means++', n_init=10, max_iter=chosen_max_iterations, random_state=42) # [cite: 10]
final_labels = kmeans_final.fit_predict(X_scaled)
final_centroids_scaled = kmeans_final.cluster_centers_
final_centroids_original = scaler.inverse_transform(final_centroids_scaled)
final_centroids_df = pd.DataFrame(final_centroids_original, columns=FEATURE_COLS_FOR_CLUSTERING)

data_with_final_labels = data_original_df.copy()
data_with_final_labels['Cluster_ID'] = final_labels

# Assign meaningful names
cluster_names_map = get_cluster_profile_names(final_centroids_df, K_CLUSTERS, FEATURE_COLS_FOR_CLUSTERING)
data_with_final_labels['Profilo_Cliente'] = data_with_final_labels['Cluster_ID'].map(cluster_names_map)

st.write(f"**Profili dei {K_CLUSTERS} Cluster Identificati (basati su {chosen_max_iterations} iterazioni massime):**")

# Display summary of clusters (means of original values)
# Ensure 'Profilo_Cliente' is correctly mapped for groupby
if not data_with_final_labels['Profilo_Cliente'].isnull().all():
    cluster_summary_display = data_with_final_labels.groupby('Profilo_Cliente')[FEATURE_COLS_FOR_CLUSTERING].mean()
    st.dataframe(cluster_summary_display.style.format("{:.1f}").highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='pink'))
else:
    st.warning("Mappatura dei profili clienti non riuscita. Controllare la logica di `get_cluster_profile_names`.")

st.markdown("---")

# --- 3. Assegnazione Nuovo Cliente ---
st.header("👤 Assegna Nuovo Cliente a un Cluster")
with st.form("new_client_form"):
    st.write("Inserisci le caratteristiche del nuovo cliente:")
    
    cols_form_1 = st.columns(3)
    with cols_form_1[0]:
        nc_freq = st.slider("Frequenza Settimanale", 0, 7, 3)
    with cols_form_1[1]:
        nc_dur = st.slider("Durata Media Allenamento (min)", 15, 180, 60)
    with cols_form_1[2]:
        nc_exp = st.slider("Anni di Esperienza Fitness", 0, 30, 1)

    cols_form_2 = st.columns(3)
    with cols_form_2[0]:
        obiettivo_map = {"Dimagrimento": 0, "Massa Muscolare": 1, "Benessere Generale": 2}
        nc_obj_str = st.selectbox("Obiettivo Principale", list(obiettivo_map.keys()))
        nc_obj = obiettivo_map[nc_obj_str]
    with cols_form_2[1]:
        nc_eta = st.slider("Età", 16, 80, 30)
    with cols_form_2[2]:
        servizi_map = {"No (non usa servizi extra)": 0, "Sì (usa servizi extra)": 1}
        nc_serv_str = st.selectbox("Usa Servizi Extra?", list(servizi_map.keys()))
        nc_serv = servizi_map[nc_serv_str]
    
    submit_button = st.form_submit_button(label="🎯 Trova Cluster per Nuovo Cliente")

new_customer_data_for_plot = None # Will store unscaled features for plotting
assigned_profile_nc_name = "N/A"

if submit_button:
    new_customer_features_unscaled = np.array([[nc_freq, nc_dur, nc_obj, nc_exp, nc_eta, nc_serv]])
    new_customer_scaled = scaler.transform(new_customer_features_unscaled) # Use the same scaler
    
    assigned_cluster_idx_nc = kmeans_final.predict(new_customer_scaled)[0] # [cite: 11]
    assigned_profile_nc_name = cluster_names_map.get(assigned_cluster_idx_nc, f"Gruppo Sconosciuto {assigned_cluster_idx_nc}")
    
    st.success(f"🎉 Il nuovo cliente è stato assegnato al Cluster ID: **{assigned_cluster_idx_nc}**")
    st.info(f"👤 Profilo Utente Corrispondente: **{assigned_profile_nc_name}**")

    new_customer_data_for_plot = new_customer_features_unscaled[0] # Save for plotting

st.markdown("---")

# --- Visualizzazione Finale con Nuovo Cliente (se presente) ---
st.header("🗺️ Grafico Finale dei Cluster con Nuovo Cliente")
fig_final, ax_final = plt.subplots(figsize=(10, 7))

# Define a color palette for profiles - use 'Accent' like in PDF example [cite: 11]
unique_profiles_final = sorted(list(data_with_final_labels['Profilo_Cliente'].unique()))
cmap_final_name = 'Accent' 
try:
    cmap_final = plt.cm.get_cmap(cmap_final_name, len(unique_profiles_final))
except ValueError: # Fallback if too many profiles for Accent
    cmap_final = plt.cm.get_cmap('viridis', len(unique_profiles_final))

profile_color_map_final = {profile: cmap_final(i) for i, profile in enumerate(unique_profiles_final)}

# Plot existing data points, colored by their final assigned cluster profile
for profile_name_iter, color_iter in profile_color_map_final.items():
    subset = data_with_final_labels[data_with_final_labels['Profilo_Cliente'] == profile_name_iter]
    if not subset.empty:
        ax_final.scatter(subset[x_axis_feat], subset[y_axis_feat], label=profile_name_iter, 
                         color=color_iter, alpha=0.7, edgecolors='k', s=50, linewidth=0.5)

# Plot final centroids
# Ensure using original scale and correct feature indices for plotting
x_feat_idx_final_plot = FEATURE_COLS_FOR_CLUSTERING.index(x_axis_feat)
y_feat_idx_final_plot = FEATURE_COLS_FOR_CLUSTERING.index(y_axis_feat)

# Plot centroids with labels if possible (requires iterating through them and linking to profile names)
for i, profile_name_iter in enumerate(unique_profiles_final): # Assuming centroid i corresponds to profile i in sorted unique_profiles
    # This link (centroid index to profile name) needs to be robust.
    # Let's find the cluster ID that maps to this profile name
    original_cluster_id_for_profile = [cid for cid, name in cluster_names_map.items() if name == profile_name_iter]
    if original_cluster_id_for_profile:
        centroid_coords = final_centroids_original[original_cluster_id_for_profile[0]]
        ax_final.scatter(centroid_coords[x_feat_idx_final_plot], centroid_coords[y_feat_idx_final_plot],
                         marker='X', s=250, color=profile_color_map_final[profile_name_iter], 
                         edgecolors='black', linewidths=1.5, label=f'Centroide: {profile_name_iter}' if i==0 else None) # Only one centroid label to avoid clutter, or specific labels

ax_final.scatter([], [], marker='X', s=250, color='gray', edgecolors='black', label='Centroidi') # General Centroid Label for legend

# Highlight new client if data is available
if new_customer_data_for_plot is not None:
    nc_plot_x = new_customer_data_for_plot[x_feat_idx_final_plot]
    nc_plot_y = new_customer_data_for_plot[y_feat_idx_final_plot]
    
    # Marker style similar to PDF example for 'Nuovo Cliente' [cite: 11]
    ax_final.scatter(nc_plot_x, nc_plot_y,
                     marker='*', s=350, facecolors='white', edgecolors='blue', linewidths=2, 
                     label=f'📍 Nuovo Cliente ({assigned_profile_nc_name})')

ax_final.set_xlabel(x_axis_feat, fontsize=12)
ax_final.set_ylabel(y_axis_feat, fontsize=12)
ax_final.set_title("Segmentazione Clienti Palestra K-Means (Finale)", fontsize=14)
ax_final.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9) # Place legend outside
ax_final.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend outside
st.pyplot(fig_final)

st.markdown("---")
st.caption("Applicazione dimostrativa per la segmentazione clienti con K-Means.")
