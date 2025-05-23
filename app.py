import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 1. Generazione Dataset Sintetico per Negozio ---
def generate_electronics_data(num_samples_per_group=75):
    """Generates a synthetic dataset for electronics store customer segmentation."""
    np.random.seed(42) # Per riproducibilità
    all_data = []
    
    # Gruppo 1: Clienti Top Fedeli
    n1 = num_samples_per_group
    g1 = pd.DataFrame({
        'Spesa_Annua_Media_Euro': np.random.normal(loc=2500, scale=800, size=n1).clip(800, 6000),
        'Frequenza_Acquisti_Trimestrale': np.random.normal(loc=5, scale=1.5, size=n1).clip(2, 10),
        'Numero_Categorie_Prodotto_Acquistate': np.random.normal(loc=6, scale=2, size=n1).clip(3, 10),
        'Anzianita_Cliente_Mesi': np.random.normal(loc=36, scale=12, size=n1).clip(12, 72)
    })
    all_data.append(g1)

    # Gruppo 2: Acquirenti Occasionali di Valore
    n2 = num_samples_per_group
    g2 = pd.DataFrame({
        'Spesa_Annua_Media_Euro': np.random.normal(loc=1200, scale=500, size=n2).clip(500, 3000),
        'Frequenza_Acquisti_Trimestrale': np.random.normal(loc=1.5, scale=0.5, size=n2).clip(0, 3),
        'Numero_Categorie_Prodotto_Acquistate': np.random.normal(loc=2.5, scale=1, size=n2).clip(1, 5),
        'Anzianita_Cliente_Mesi': np.random.normal(loc=18, scale=10, size=n2).clip(3, 48)
    })
    all_data.append(g2)

    # Gruppo 3: Clienti Regolari Contenuti
    n3 = num_samples_per_group
    g3 = pd.DataFrame({
        'Spesa_Annua_Media_Euro': np.random.normal(loc=400, scale=150, size=n3).clip(100, 800),
        'Frequenza_Acquisti_Trimestrale': np.random.normal(loc=3, scale=1, size=n3).clip(1, 6),
        'Numero_Categorie_Prodotto_Acquistate': np.random.normal(loc=4, scale=1.5, size=n3).clip(2, 7),
        'Anzianita_Cliente_Mesi': np.random.normal(loc=24, scale=8, size=n3).clip(6, 48)
    })
    all_data.append(g3)
    
    # Gruppo 4: Nuovi Esploratori / Acquirenti Infrequenti
    n4 = num_samples_per_group
    g4 = pd.DataFrame({
        'Spesa_Annua_Media_Euro': np.random.normal(loc=150, scale=70, size=n4).clip(30, 400),
        'Frequenza_Acquisti_Trimestrale': np.random.normal(loc=0.8, scale=0.4, size=n4).clip(0, 2),
        'Numero_Categorie_Prodotto_Acquistate': np.random.normal(loc=1.5, scale=0.5, size=n4).clip(1, 3),
        'Anzianita_Cliente_Mesi': np.random.normal(loc=6, scale=4, size=n4).clip(1, 18)
    })
    all_data.append(g4)

    electronics_df = pd.concat(all_data, ignore_index=True)
    electronics_df = electronics_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
    return electronics_df

# --- 4. Etichette Significative per i Cluster  ---
def get_electronics_cluster_names(centroids_df, k_clusters, feature_cols_list):
    """Assegna nomi significativi ai cluster ."""
    assigned_names_map = {}
    
    profile_definitions = {
        "Cliente Top Fedele": {'Spesa_Annua_Media_Euro': ('>=', 1500), 'Anzianita_Cliente_Mesi': ('>=', 24), 'Frequenza_Acquisti_Trimestrale': ('>=', 3)},
        "Acquirente Occasionale di Valore": {'Spesa_Annua_Media_Euro': ('>=', 800), 'Frequenza_Acquisti_Trimestrale': ('<', 2.5)},
        "Cliente Regolare Standard": {'Spesa_Annua_Media_Euro': ('<', 1000), 'Frequenza_Acquisti_Trimestrale': ('>=', 2), 'Anzianita_Cliente_Mesi': ('>=', 12)},
        "Nuovo Cliente o Esploratore": {'Anzianita_Cliente_Mesi': ('<', 12), 'Spesa_Annua_Media_Euro': ('<', 500)}
    }
    
    used_names_indices = [] 

    for i_centroid in range(k_clusters): 
        centroid_series = centroids_df.iloc[i_centroid]
        best_match_name = f"Gruppo Elettronica {i_centroid+1}" 
        found_specific_match = False

        for name_key, rules in profile_definitions.items():
            is_name_already_taken_by_dominant_centroid = False
            for assigned_idx, assigned_name in assigned_names_map.items():
                if assigned_name == name_key and assigned_idx != i_centroid : 
                    is_name_already_taken_by_dominant_centroid = True
                    break
            if is_name_already_taken_by_dominant_centroid:
                continue

            match = True
            for feature, (op, val) in rules.items():
                if not isinstance(feature, str) or feature not in centroids_df.columns:
                    match = False
                    break
                
                centroid_val = centroid_series[feature]

                if op == '<' and not (centroid_val < val): match = False; break
                if op == '>' and not (centroid_val > val): match = False; break
                if op == '<=' and not (centroid_val <= val): match = False; break
                if op == '>=' and not (centroid_val >= val): match = False; break
            
            if match:
                if i_centroid not in used_names_indices and name_key not in [assigned_names_map.get(j) for j in used_names_indices if j != i_centroid]:
                    best_match_name = name_key
                    used_names_indices.append(i_centroid)
                    found_specific_match = True
                    break
        
        assigned_names_map[i_centroid] = best_match_name

    final_names = {}
    temp_used_profile_names = []
    for i_centroid in range(k_clusters): 
        proposed_name = assigned_names_map.get(i_centroid, f"Gruppo Elettronica {i_centroid+1}")
        original_proposed_name = proposed_name
        suffix_counter = 1
        while proposed_name in temp_used_profile_names:
            proposed_name = f"{original_proposed_name} ({suffix_counter})"
            suffix_counter += 1
        final_names[i_centroid] = proposed_name
        temp_used_profile_names.append(proposed_name)
            
    return final_names

# --- Configurazione App Streamlit ---
st.set_page_config(layout="wide")
st.title("💻 Segmentazione Clienti E-commerce con K-Means")

# --- Impostazioni Globali e Generazione Dati ---
K_CLUSTERS = 4 
FEATURE_COLS_ELETTRONICA = ['Spesa_Annua_Media_Euro', 'Frequenza_Acquisti_Trimestrale', 
                            'Numero_Categorie_Prodotto_Acquistate', 'Anzianita_Cliente_Mesi']

data_df = generate_electronics_data(num_samples_per_group=75)
data_original_df = data_df.copy()

X = data_df[FEATURE_COLS_ELETTRONICA].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Controlli Sidebar ---
st.sidebar.header("⚙️ Parametri K-Means")
chosen_max_iterations = st.sidebar.slider("Numero Massimo di Iterazioni K-Means", 1, 4, 4,
                                          help="Seleziona il numero massimo di iterazioni per l'algoritmo K-Means.")

st.sidebar.header("📊 Visualizzazione Grafici")
x_axis_feat = st.sidebar.selectbox("Feature per Asse X:", FEATURE_COLS_ELETTRONICA, index=0)
y_axis_feat = st.sidebar.selectbox("Feature per Asse Y:", FEATURE_COLS_ELETTRONICA, index=1)

if x_axis_feat == y_axis_feat:
    st.sidebar.warning("Seleziona feature diverse per l'asse X e Y per una visualizzazione significativa.")
    st.stop()

# --- Evoluzione dei Cluster ---
st.header("🔄 Evoluzione dei Cluster")

iterations_to_show = sorted(list(set([1] + ([chosen_max_iterations // 2] if chosen_max_iterations > 2 else []) + [chosen_max_iterations])))
if not iterations_to_show or iterations_to_show[-1] == 0 : iterations_to_show = [1]
if len(iterations_to_show) > 1 and iterations_to_show[0] == 0 : iterations_to_show = iterations_to_show[1:] 
if not iterations_to_show : iterations_to_show = [1] 
if len(iterations_to_show) > 1 and iterations_to_show[0] == iterations_to_show[1] and len(iterations_to_show) > 1 : 
    iterations_to_show = sorted(list(set(iterations_to_show)))

if len(iterations_to_show) == 1:
    iterations_text = f"{iterations_to_show[0]} iterazione."
elif len(iterations_to_show) == 2:
    iterations_text = f"{iterations_to_show[0]} e {iterations_to_show[1]} iterazioni."
else:
    iterations_text_parts = [str(it) for it in iterations_to_show[:-1]]
    iterations_text = ", ".join(iterations_text_parts) + f", e {iterations_to_show[-1]} iterazioni."

st.markdown(f"Visualizzazione dei cluster e inerzia dopo {iterations_text}")

evolution_cols = st.columns(len(iterations_to_show))
middle_plot_index = len(iterations_to_show) // 2 

for i, num_iter in enumerate(iterations_to_show):
    with evolution_cols[i]:
        st.subheader(f"Iter: {num_iter}")
        kmeans_evol = KMeans(n_clusters=K_CLUSTERS, init='k-means++', n_init=1, max_iter=num_iter, random_state=42)
        kmeans_evol.fit(X_scaled) 
        labels_evol = kmeans_evol.labels_
        centroids_evol_scaled = kmeans_evol.cluster_centers_
        centroids_evol_original = scaler.inverse_transform(centroids_evol_scaled) 
        inertia_evol = kmeans_evol.inertia_

        plot_df_evol = data_original_df.copy()
        plot_df_evol['Cluster_Temp'] = labels_evol
        
        fig_evol, ax_evol = plt.subplots(figsize=(6, 5))
        cmap_evol = plt.cm.get_cmap('viridis', K_CLUSTERS)
        
        scatter_evol = ax_evol.scatter(plot_df_evol[x_axis_feat], plot_df_evol[y_axis_feat], 
                                       c=plot_df_evol['Cluster_Temp'], cmap=cmap_evol, alpha=0.6, s=30, edgecolors='grey', linewidth=0.5)
        
        x_feat_idx_plot = FEATURE_COLS_ELETTRONICA.index(x_axis_feat)
        y_feat_idx_plot = FEATURE_COLS_ELETTRONICA.index(y_axis_feat)
        ax_evol.scatter(centroids_evol_original[:, x_feat_idx_plot], centroids_evol_original[:, y_feat_idx_plot], 
                        marker='X', s=150, color='red', edgecolors='black', label='Centroidi')
        
        ax_evol.set_xlabel(x_axis_feat.replace("_", " "), fontsize=9)
        ax_evol.set_ylabel(y_axis_feat.replace("_", " "), fontsize=9)
        ax_evol.set_title(f"Cluster (max {num_iter} iter.)", fontsize=10)
        ax_evol.tick_params(axis='both', which='major', labelsize=8)
        ax_evol.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_evol)
        
        st.markdown(f"**Inerzia (WCSS):** `{inertia_evol:.2f}`")
        if i == middle_plot_index:
            st.caption("L'inerzia misura la somma delle distanze quadratiche dei campioni dal loro centroide più vicino. Valori più bassi indicano cluster più compatti.")
        st.markdown("---") 

st.markdown("---") 

# --- Modello Finale & Naming Cluster ---
st.header("🏆 Risultato Finale della Clusterizzazione")
kmeans_final = KMeans(n_clusters=K_CLUSTERS, init='k-means++', n_init=10, max_iter=chosen_max_iterations, random_state=42)
final_labels = kmeans_final.fit_predict(X_scaled)
final_centroids_scaled = kmeans_final.cluster_centers_
final_centroids_original = scaler.inverse_transform(final_centroids_scaled)
final_centroids_df = pd.DataFrame(final_centroids_original, columns=FEATURE_COLS_ELETTRONICA)

final_inertia = kmeans_final.inertia_

data_with_final_labels = data_original_df.copy()
data_with_final_labels['Cluster_ID'] = final_labels

cluster_names_map = get_electronics_cluster_names(final_centroids_df, K_CLUSTERS, FEATURE_COLS_ELETTRONICA)
data_with_final_labels['Profilo_Cliente'] = data_with_final_labels['Cluster_ID'].map(cluster_names_map)

st.write(f"**Profili dei {K_CLUSTERS} Cluster Identificati (basati su {chosen_max_iterations} iterazioni massime):**")
st.write(f"**Inerzia Finale del Modello:** `{final_inertia:.2f}`")


if not data_with_final_labels['Profilo_Cliente'].isnull().all():
    cluster_summary_display = data_with_final_labels.groupby('Profilo_Cliente')[FEATURE_COLS_ELETTRONICA].mean()
    st.dataframe(
        cluster_summary_display
        .style.format("{:.1f}")
        .highlight_max(axis=0, color='lightgreen', props='font-weight:bold;')
        .highlight_min(axis=0, color='#FFCCCB', props='font-weight:bold;')
    )
else:
    st.warning("Mappatura dei profili clienti non riuscita. Controllare la logica di `get_electronics_cluster_names`.")

st.markdown("---")

# --- Simulazione Assegnazione Profilo Cliente --- MODIFICATO QUI ---
st.header("👤 Simula Assegnazione Profilo Cliente") 
with st.form("profile_simulation_form"): # Nome del form modificato per chiarezza
    st.write("Inserisci le caratteristiche di un profilo cliente per simularne l'assegnazione a un cluster:") # Testo modificato
    
    cols_form_1 = st.columns(2)
    with cols_form_1[0]:
        # I campi di input rimangono, l'utente può simulare qualsiasi profilo
        nc_spesa = st.number_input("Spesa Annua Media (€)", min_value=0.0, value=500.0, step=50.0, key="sim_spesa")
        nc_freq = st.number_input("Frequenza Acquisti Trimestrale", min_value=0, value=2, step=1, key="sim_freq")
    with cols_form_1[1]:
        nc_cat = st.number_input("Numero Categorie Prodotto Acquistate", min_value=1, value=3, step=1, key="sim_cat")
        nc_anz = st.number_input("Anzianità Cliente (Mesi)", min_value=0, value=12, step=1, key="sim_anz")
    
    submit_button = st.form_submit_button(label="🎯 Simula Assegnazione Cluster") # Etichetta pulsante modificata

simulated_profile_data_for_plot = None # Rinominata variabile per chiarezza
assigned_profile_sim_name = "N/A" # Rinominata variabile per chiarezza

if submit_button:
    simulated_profile_features_unscaled = np.array([[nc_spesa, nc_freq, nc_cat, nc_anz]])
    simulated_profile_scaled = scaler.transform(simulated_profile_features_unscaled)
    
    assigned_cluster_idx_sim = kmeans_final.predict(simulated_profile_scaled)[0] # Rinominata variabile
    assigned_profile_sim_name = cluster_names_map.get(assigned_cluster_idx_sim, f"Gruppo Elettronica {assigned_cluster_idx_sim}")
    
    # Messaggi di output modificati
    st.success(f"🎉 Il profilo cliente simulato è stato assegnato al Cluster ID: **{assigned_cluster_idx_sim}**")
    st.info(f"👤 Profilo Utente Corrispondente: **{assigned_profile_sim_name}**")
    simulated_profile_data_for_plot = simulated_profile_features_unscaled[0]

st.markdown("---")

# --- Visualizzazione Finale --- MODIFICATO QUI ---
st.header("🗺️ Grafico Finale dei Cluster con Profilo Cliente Simulato") # Titolo modificato
fig_final, ax_final = plt.subplots(figsize=(10, 7))

unique_profiles_final = sorted(list(data_with_final_labels['Profilo_Cliente'].unique()))
cmap_final_name = 'Accent' 
try:
    cmap_final = plt.cm.get_cmap(cmap_final_name, len(unique_profiles_final))
except ValueError: 
    cmap_final = plt.cm.get_cmap('viridis', len(unique_profiles_final))

profile_color_map_final = {profile: cmap_final(i) for i, profile in enumerate(unique_profiles_final)}

for profile_name_iter, color_iter in profile_color_map_final.items():
    subset = data_with_final_labels[data_with_final_labels['Profilo_Cliente'] == profile_name_iter]
    if not subset.empty:
        ax_final.scatter(subset[x_axis_feat], subset[y_axis_feat], label=profile_name_iter, 
                         color=color_iter, alpha=0.7, edgecolors='k', s=50, linewidth=0.5)

x_feat_idx_final_plot = FEATURE_COLS_ELETTRONICA.index(x_axis_feat)
y_feat_idx_final_plot = FEATURE_COLS_ELETTRONICA.index(y_axis_feat)

ax_final.scatter([], [], marker='X', s=250, color='red', edgecolors='black', label='Centroidi') 

for i_centroid in range(K_CLUSTERS): 
    ax_final.scatter(final_centroids_original[i_centroid, x_feat_idx_final_plot], 
                     final_centroids_original[i_centroid, y_feat_idx_final_plot],
                     marker='X', s=250, color='red', edgecolors='black', linewidths=1.5) 

if simulated_profile_data_for_plot is not None: # Variabile rinominata
    sim_plot_x = simulated_profile_data_for_plot[x_feat_idx_final_plot] # Variabile rinominata
    sim_plot_y = simulated_profile_data_for_plot[y_feat_idx_final_plot] # Variabile rinominata
    
    ax_final.scatter(sim_plot_x, sim_plot_y, # Variabili rinominate
                     marker='*', s=350, facecolors='white', edgecolors='blue', linewidths=2, 
                     label=f'📍 Profilo Simulato ({assigned_profile_sim_name})') # Etichetta modificata

ax_final.set_xlabel(x_axis_feat.replace("_", " "), fontsize=12)
ax_final.set_ylabel(y_axis_feat.replace("_", " "), fontsize=12)
ax_final.set_title("Segmentazione Clienti E-commerce (Finale)", fontsize=14) # Titolo grafico finale può rimanere generico
ax_final.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9) 
ax_final.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0, 0.85, 1]) 
st.pyplot(fig_final)

st.markdown("---")
st.caption("Applicazione dimostrativa per la segmentazione clienti con K-Means in un e-commerce.")
