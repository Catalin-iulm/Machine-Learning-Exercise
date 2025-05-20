import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from datetime import datetime, timedelta
import random

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="MarketPro: Individuazione Bodybuilder")

# --- Funzione per Generare Dati Simulati (Cashed per performance) ---
@st.cache_data
def generate_customer_data_for_bodybuilders(n_samples, random_state_data):
    np.random.seed(random_state_data)
    random.seed(random_state_data)

    data_points = []
    current_id = 1000  # Starting customer ID
    
    # Date range for purchase timestamps (last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    # 1. Bodybuilder/Fitness Enthusiasts (Niche, ~10% of customers)
    # 2. Standard Families (Largest segment, ~35%)
    # 3. Budget Shoppers (Cost-conscious, ~20%)
    # 4. Health-Conscious (General, not extreme fitness, ~20%)
    # 5. Convenience/Junk Food Consumers (~15%)

    for i in range(n_samples):
        customer_id = current_id + i
        segment_choice = np.random.choice([
            "Bodybuilder", "Standard Family", "Budget Shopper", 
            "Health-Conscious", "Junk Food Consumer"
        ], p=[0.10, 0.35, 0.20, 0.20, 0.15])

        # Base features for all segments
        if segment_choice == "Bodybuilder":
            # Core protein metrics
            spesa_prot = np.random.normal(80, 15)  # Very high
            spesa_carbo = np.random.normal(60, 10) # High
            spesa_junk = np.random.normal(5, 3)    # Very low
            freq_sport = np.random.normal(4, 1)    # High
            eta = np.random.normal(28, 5)          # Young adults
            varieta_prot = np.random.normal(18, 3) # High variety
            
            # New specific metrics
            perc_proteine_polvere = np.random.normal(25, 5)  # >15%
            kg_carne_magra = np.random.normal(4.5, 1)        # >3kg/week
            integratori_specifici = random.choices([0, 1, 2, 3], weights=[0.1, 0.2, 0.3, 0.4])[0]  # 0-3 products
            orario_acquisto = random.choice(["7:00-9:00", "18:00-20:00", "12:00-14:00", "18:00-20:00", "7:00-9:00"])
            uova_settimanali = np.random.normal(24, 6)       # ~2 dozen eggs/week
            rapporto_carbo_prot = np.random.normal(1.2, 0.3) # <1.5
            accessori_fitness = random.choices([0, 1, 2], weights=[0.1, 0.3, 0.6])[0]  # 0-2 items/month
            brand_loyalty = np.random.normal(85, 10)         # >70%
            no_go_products = random.choices([0, 1], weights=[0.9, 0.1])[0]  # Rarely buys junk
            
            # Generate realistic purchase timestamps (post-workout times)
            purchase_dates = [start_date + timedelta(days=random.randint(0, 180), 
                              hours=random.choice([7,8,18,19])) for _ in range(random.randint(15, 30))]
            
        elif segment_choice == "Standard Family":
            spesa_prot = np.random.normal(40, 10)
            spesa_carbo = np.random.normal(35, 8)
            spesa_junk = np.random.normal(25, 7)
            freq_sport = np.random.normal(1, 0.5)
            eta = np.random.normal(42, 7)
            varieta_prot = np.random.normal(8, 2)
            
            # New metrics
            perc_proteine_polvere = np.random.normal(5, 3)
            kg_carne_magra = np.random.normal(2, 0.8)
            integratori_specifici = random.choices([0, 1], weights=[0.8, 0.2])[0]
            orario_acquisto = random.choice(["9:00-12:00", "15:00-18:00", "12:00-14:00"])
            uova_settimanali = np.random.normal(12, 4)
            rapporto_carbo_prot = np.random.normal(2.5, 0.5)
            accessori_fitness = 0
            brand_loyalty = np.random.normal(40, 15)
            no_go_products = random.choices([0, 1], weights=[0.6, 0.4])[0]
            
            # Random purchase times
            purchase_dates = [start_date + timedelta(days=random.randint(0, 180), 
                              hours=random.randint(9,20)) for _ in range(random.randint(8, 15))]
            
        elif segment_choice == "Budget Shopper":
            spesa_prot = np.random.normal(25, 5)
            spesa_carbo = np.random.normal(20, 5)
            spesa_junk = np.random.normal(15, 5)
            freq_sport = np.random.normal(0.5, 0.3)
            eta = np.random.normal(35, 10)
            varieta_prot = np.random.normal(5, 1)
            
            # New metrics
            perc_proteine_polvere = np.random.normal(2, 1)
            kg_carne_magra = np.random.normal(1, 0.5)
            integratori_specifici = 0
            orario_acquisto = random.choice(["9:00-12:00", "15:00-18:00"])
            uova_settimanali = np.random.normal(6, 3)
            rapporto_carbo_prot = np.random.normal(3.0, 0.6)
            accessori_fitness = 0
            brand_loyalty = np.random.normal(30, 10)
            no_go_products = random.choices([0, 1], weights=[0.4, 0.6])[0]
            
            purchase_dates = [start_date + timedelta(days=random.randint(0, 180), 
                              hours=random.randint(10,19)) for _ in range(random.randint(5, 10))]
            
        elif segment_choice == "Health-Conscious":
            spesa_prot = np.random.normal(55, 10)
            spesa_carbo = np.random.normal(45, 10)
            spesa_junk = np.random.normal(10, 4)
            freq_sport = np.random.normal(2, 0.8)
            eta = np.random.normal(38, 8)
            varieta_prot = np.random.normal(12, 3)
            
            # New metrics
            perc_proteine_polvere = np.random.normal(12, 4)
            kg_carne_magra = np.random.normal(2.5, 0.7)
            integratori_specifici = random.choices([0, 1], weights=[0.7, 0.3])[0]
            orario_acquisto = random.choice(["7:00-9:00", "12:00-14:00", "15:00-18:00"])
            uova_settimanali = np.random.normal(18, 5)
            rapporto_carbo_prot = np.random.normal(1.8, 0.4)
            accessori_fitness = random.choices([0, 1], weights=[0.8, 0.2])[0]
            brand_loyalty = np.random.normal(60, 15)
            no_go_products = random.choices([0, 1], weights=[0.8, 0.2])[0]
            
            purchase_dates = [start_date + timedelta(days=random.randint(0, 180), 
                              hours=random.randint(7,20)) for _ in range(random.randint(10, 20))]
            
        elif segment_choice == "Junk Food Consumer":
            spesa_prot = np.random.normal(15, 5)
            spesa_carbo = np.random.normal(12, 5)
            spesa_junk = np.random.normal(45, 10) # Very high
            freq_sport = np.random.normal(0.2, 0.1)
            eta = np.random.normal(25, 5)
            varieta_prot = np.random.normal(3, 1)
            
            # New metrics
            perc_proteine_polvere = np.random.normal(1, 0.5)
            kg_carne_magra = np.random.normal(0.5, 0.3)
            integratori_specifici = 0
            orario_acquisto = random.choice(["12:00-14:00", "18:00-20:00", "20:00-22:00"])
            uova_settimanali = np.random.normal(3, 2)
            rapporto_carbo_prot = np.random.normal(4.0, 0.8)
            accessori_fitness = 0
            brand_loyalty = np.random.normal(20, 10)
            no_go_products = 1  # Always buys junk food
            
            purchase_dates = [start_date + timedelta(days=random.randint(0, 180), 
                              hours=random.randint(12,22)) for _ in range(random.randint(5, 15))]
        
        # Calculate purchase frequency (times/month)
        purchase_frequency = len(purchase_dates) / 6  # 6 months period
        
        # Calculate seasonality factor (higher in Jan and May-Jun)
        month_counts = [d.month for d in purchase_dates]
        seasonality_factor = sum(1 for m in month_counts if m in [1, 5, 6]) / len(month_counts) if month_counts else 0
        
        data_points.append([
            customer_id,
            max(0, spesa_prot), 
            max(0, spesa_carbo), 
            max(0, spesa_junk), 
            max(0, freq_sport), 
            max(18, eta), 
            max(1, varieta_prot),
            max(0, min(100, perc_proteine_polvere)),  # % protein powder
            max(0, kg_carne_magra),  # kg lean meat
            integratori_specifici,  # specific supplements count
            orario_acquisto,  # purchase time category
            max(0, uova_settimanali),  # eggs per week
            max(0.1, rapporto_carbo_prot),  # carbs/protein ratio
            accessori_fitness,  # fitness accessories count
            max(0, min(100, brand_loyalty)),  # brand loyalty %
            no_go_products,  # buys forbidden products
            purchase_frequency,  # purchases/month
            seasonality_factor,  # seasonal purchase pattern
            segment_choice  # actual segment for validation
        ])
    
    df = pd.DataFrame(data_points, columns=[
        'ID_Cliente',
        'Spesa_Proteine_Settimanale (€)',
        'Spesa_Carbo_Complessi_Settimanale (€)',
        'Spesa_JunkFood_Settimanale (€)',
        'Frequenza_Reparto_SportSalute (volte/mese)',
        'Età (anni)',
        'Varietà_Prodotti_Proteici (count)',
        '%_Acquisti_Proteine_Polvere',
        'Kg_Carne_Magra_Settimanali',
        'Integratori_Specifici (count)',
        'Orario_Acquisto_Preferito',
        'Uova_Acquistate_Settimanali (count)',
        'Rapporto_Carboidrati_Proteine',
        'Accessori_Fitness_Acquistati (count)',
        'Brand_Loyalty (%)',
        'Acquista_Prodotti_NoGo',
        'Frequenza_Acquisti (volte/mese)',
        'Stagionalità_Acquisti',
        'Segmento_Reale'  # For validation only
    ])
    
    return df

# ... (rest of the code remains the same until the data visualization part)

# --- Generazione e Visualizzazione Dati (Prima del Clustering) ---
customer_df = generate_customer_data_for_bodybuilders(n_samples, random_state_data)

st.subheader("📊 Panoramica dei Dati Clienti Simulati")
st.write("Ecco un estratto del dataset simulato con le nuove metriche specifiche:")
st.dataframe(customer_df.head())

st.write("Per visualizzare i cluster e individuare i bodybuilder, possiamo usare diverse combinazioni di feature:")
st.markdown("- **'%_Acquisti_Proteine_Polvere' vs 'Kg_Carne_Magra_Settimanali'**: Combinazione molto specifica per bodybuilder")
st.markdown("- **'Spesa_Proteine_Settimanale (€)' vs 'Rapporto_Carboidrati_Proteine'**: Mostra sia l'assunzione proteica che il bilanciamento con i carboidrati")
st.markdown("- **'Integratori_Specifici (count)' vs 'Accessori_Fitness_Acquistati (count)'**: Comportamenti di acquisto correlati al fitness")

# Select which features to use for clustering
feature_options = [
    'Spesa_Proteine_Settimanale (€)',
    'Spesa_Carbo_Complessi_Settimanale (€)',
    'Spesa_JunkFood_Settimanale (€)',
    'Frequenza_Reparto_SportSalute (volte/mese)',
    'Varietà_Prodotti_Proteici (count)',
    '%_Acquisti_Proteine_Polvere',
    'Kg_Carne_Magra_Settimanali',
    'Integratori_Specifici (count)',
    'Uova_Acquistate_Settimanali (count)',
    'Rapporto_Carboidrati_Proteine',
    'Accessori_Fitness_Acquistati (count)',
    'Brand_Loyalty (%)'
]

# Let user select which features to use for clustering
selected_features = st.multiselect(
    "Seleziona le feature da usare per il clustering:",
    feature_options,
    default=['%_Acquisti_Proteine_Polvere', 'Kg_Carne_Magra_Settimanali', 'Integratori_Specifici (count)']
)

# ... (rest of the code remains the same, just use selected_features instead of hardcoded features)
