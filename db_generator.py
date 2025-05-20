import numpy as np
import pandas as pd

def generate_advanced_customer_db(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    data = []
    regions = ['Nord', 'Centro', 'Sud', 'Isole']
    genders = ['Male', 'Female']
    marital_statuses = ['Single', 'Partnered']
    supplement_types = ['Whey', 'Caseina', 'Vegetale', 'Altro']
    fitness_products = ['Tapis roulant', 'Pesi', 'Cyclette', 'Nessuno']

    for i in range(n_samples):
        segment = np.random.choice(
            ["Bodybuilder", "Standard Family", "Budget Shopper", "Health-Conscious", "Junk Food Consumer"],
            p=[0.10, 0.35, 0.20, 0.20, 0.15]
        )
        gender = np.random.choice(genders)
        region = np.random.choice(regions)
        marital = np.random.choice(marital_statuses)
        education = int(np.clip(np.random.normal(14, 2), 8, 20))
        income = int(np.clip(np.random.normal(35000, 12000), 15000, 100000))
        fitness_level = int(np.clip(np.random.normal(3, 1), 1, 5))
        fitness_product = np.random.choice(fitness_products, p=[0.3, 0.3, 0.2, 0.2])
        supplement_type = np.random.choice(supplement_types, p=[0.6, 0.15, 0.2, 0.05])
        ore_allenamento = np.clip(np.random.normal(5, 2), 0, 14)
        freq_integratori = np.clip(np.random.normal(4, 2), 0, 12)

        # Caratteristiche specifiche per segmento
        if segment == "Bodybuilder":
            spesa_prot = np.random.normal(80, 15)
            spesa_carbo = np.random.normal(55, 10)
            spesa_junk = np.random.normal(5, 3)
            freq_sport = np.random.normal(5, 1)
            eta = np.random.normal(29, 4)
            varieta_prot = np.random.normal(18, 2)
            ore_allenamento += np.random.normal(5, 2)
            fitness_level = int(np.clip(np.random.normal(5, 0.5), 4, 5))
        elif segment == "Standard Family":
            spesa_prot = np.random.normal(40, 10)
            spesa_carbo = np.random.normal(30, 8)
            spesa_junk = np.random.normal(20, 7)
            freq_sport = np.random.normal(1, 0.5)
            eta = np.random.normal(45, 7)
            varieta_prot = np.random.normal(8, 2)
        elif segment == "Budget Shopper":
            spesa_prot = np.random.normal(20, 5)
            spesa_carbo = np.random.normal(15, 5)
            spesa_junk = np.random.normal(10, 5)
            freq_sport = np.random.normal(0.5, 0.3)
            eta = np.random.normal(35, 10)
            varieta_prot = np.random.normal(5, 1)
        elif segment == "Health-Conscious":
            spesa_prot = np.random.normal(50, 10)
            spesa_carbo = np.random.normal(40, 10)
            spesa_junk = np.random.normal(8, 4)
            freq_sport = np.random.normal(2, 0.8)
            eta = np.random.normal(38, 8)
            varieta_prot = np.random.normal(12, 3)
        elif segment == "Junk Food Consumer":
            spesa_prot = np.random.normal(15, 5)
            spesa_carbo = np.random.normal(10, 5)
            spesa_junk = np.random.normal(40, 10)
            freq_sport = np.random.normal(0.2, 0.1)
            eta = np.random.normal(28, 5)
            varieta_prot = np.random.normal(3, 1)

        data.append([
            f"CUST{i+1:05d}", segment, gender, region, marital, education, income,
            max(0, spesa_prot), max(0, spesa_carbo), max(0, spesa_junk),
            max(0, freq_sport), max(18, eta), max(1, varieta_prot),
            round(ore_allenamento, 1), int(freq_integratori), supplement_type,
            fitness_product, fitness_level
        ])

    columns = [
        "Customer_ID", "Segment", "Gender", "Region", "Marital_Status", "Education_Years", "Annual_Income",
        "Spesa_Proteine_Settimanale", "Spesa_Carbo_Complessi_Settimanale", "Spesa_JunkFood_Settimanale",
        "Frequenza_Reparto_SportSalute", "Eta", "Varieta_Prodotti_Proteici",
        "Ore_Allenamento_Settimanali", "Frequenza_Acquisto_Integratori", "Tipo_Integratore_Preferito",
        "Prodotto_Fitness_Posseduto", "Livello_Fitness"
    ]
    return pd.DataFrame(data, columns=columns)
