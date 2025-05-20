import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px

# 1. Configurazione della Pagina Streamlit
st.set_page_config(layout="wide", page_title="Analisi Avanzata della Segmentazione Clienti", page_icon="🛒")

# 2. Titolo e Introduzione
st.title("🛒 Analisi Avanzata della Segmentazione Clienti per il Retail")
st.markdown("""
Questa applicazione interattiva ti permette di esplorare algoritmi di clustering avanzati su dati simulati di clienti di supermercati.
Scopri segmenti di clienti nascosti e ottieni insight utili per strategie di marketing mirate.
""")

# 3. Funzione per Generare Dati Simulati di Clienti Retail
@st.cache_data
def genera_dati_retail(n_campioni: int, livello_rumore: int, stato_casuale: int):
    np.random.seed(stato_casuale)
    archetipi = {
        "Giovani Professionisti Urbani": {
            "eta": (28, 5), "reddito": (55000, 12000),
            "visite_online_mensili": (18, 5), "dimensione_cestino_media": (45, 10),
            "pct_acquisti_bio": (0.35, 0.1), "sensibilita_sconto": (0.6, 0.15),
            "visite_negozio_mensili": (4, 2), "lealta_brand": (0.7, 0.1)
        },
        "Famiglie con Budget Limitato": {
            "eta": (38, 6), "reddito": (45000, 8000),
            "visite_online_mensili": (8, 3), "dimensione_cestino_media": (75, 15),
            "pct_acquisti_bio": (0.15, 0.08), "sensibilita_sconto": (0.9, 0.05),
            "visite_negozio_mensili": (12, 3), "lealta_brand": (0.4, 0.15)
        },
        "Acquirenti Premium": {
            "eta": (45, 8), "reddito": (95000, 20000),
            "visite_online_mensili": (12, 4), "dimensione_cestino_media": (120, 25),
            "pct_acquisti_bio": (0.5, 0.15), "sensibilita_sconto": (0.3, 0.1),
            "visite_negozio_mensili": (6, 2), "lealta_brand": (0.85, 0.08)
        },
        "Coppie in Pensione": {
            "eta": (65, 5), "reddito": (40000, 10000),
            "visite_online_mensili": (4, 2), "dimensione_cestino_media": (55, 12),
            "pct_acquisti_bio": (0.25, 0.1), "sensibilita_sconto": (0.7, 0.1),
            "visite_negozio_mensili": (8, 2), "lealta_brand": (0.6, 0.12)
        },
        "Appassionati di Salute": {
            "eta": (35, 7), "reddito": (60000, 15000),
            "visite_online_mensili": (15, 4), "dimensione_cestino_media": (65, 15),
            "pct_acquisti_bio": (0.75, 0.1), "sensibilita_sconto": (0.5, 0.15),
            "visite_negozio_mensili": (6, 2), "lealta_brand": (0.65, 0.12)
        },
        "Acquirenti di Convenienza": {
            "eta": (32, 8), "reddito": (48000, 10000),
            "visite_online_mensili": (25, 6), "dimensione_cestino_media": (30, 8),
            "pct_acquisti_bio": (0.2, 0.1), "sensibilita_sconto": (0.8, 0.1),
            "visite_negozio_mensili": (2, 1), "lealta_brand": (0.3, 0.15)
        }
    }

    data = []
    campioni_per_tipo = n_campioni // len(archetipi)
    campioni_rimanenti = n_campioni % len(archetipi)

    for i, (nome_archetipo, params) in enumerate(archetipi.items()):
        n = campioni_per_tipo + (1 if i < campioni_rimanenti else 0)
        eta = np.random.normal(params["eta"][0], params["eta"][1], n)
        reddito = np.random.normal(params["reddito"][0], params["reddito"][1], n)
        visite_online_mensili = np.random.poisson(params["visite_online_mensili"][0], n)
        dimensione_cestino_media = np.abs(np.random.normal(params["dimensione_cestino_media"][0], params["dimensione_cestino_media"][1], n))
        pct_acquisti_bio = np.clip(np.random.normal(params["pct_acquisti_bio"][0], params["pct_acquisti_bio"][1], n), 0, 1)
        sensibilita_sconto = np.clip(np.random.normal(params["sensibilita_sconto"][0], params["sensibilita_sconto"][1], n), 0, 1)
        visite_negozio_mensili = np.random.poisson(params["visite_negozio_mensili"][0], n)
        lealta_brand = np.clip(np.random.normal(params["lealta_brand"][0], params["lealta_brand"][1], n), 0, 1)

        maschera_rumore = np.random.random(n) < (livello_rumore/100)
        if maschera_rumore.any():
            fattore_rumore_singolo = 1 + np.random.normal(0, 0.5, size=np.sum(maschera_rumore))
            eta[maschera_rumore] *= fattore_rumore_singolo
            reddito[maschera_rumore] *= fattore_rumore_singolo
            visite_online_mensili[maschera_rumore] = np.abs(visite_online_mensili[maschera_rumore] * fattore_rumore_singolo)
            dimensione_cestino_media[maschera_rumore] = np.abs(dimensione_cestino_media[maschera_rumore] * fattore_rumore_singolo)
            pct_acquisti_bio[maschera_rumore] = np.clip(pct_acquisti_bio[maschera_rumore] * fattore_rumore_singolo, 0, 1)
            sensibilita_sconto[maschera_rumore] = np.clip(sensibilita_sconto[maschera_rumore] * fattore_rumore_singolo, 0, 1)
            visite_negozio_mensili[maschera_rumore] = np.abs(visite_negozio_mensili[maschera_rumore] * fattore_rumore_singolo)
            lealta_brand[maschera_rumore] = np.clip(lealta_brand[maschera_rumore] * fattore_rumore_singolo, 0, 1)

        for j in range(n):
            genere = np.random.choice(["Maschio", "Femmina"], p=[0.45, 0.55])
            carta_fedelta = np.random.choice([True, False], p=[0.7, 0.3])
            data.append([
                max(18, min(80, int(eta[j]))),
                genere,
                max(20000, min(200000, int(reddito[j]))),
                max(0, int(visite_online_mensili[j])),
                max(10, float(dimensione_cestino_media[j])),
                float(pct_acquisti_bio[j]),
                float(sensibilita_sconto[j]),
                max(0, int(visite_negozio_mensili[j])),
                float(lealta_brand[j]),
                carta_fedelta,
                nome_archetipo
            ])

    df = pd.DataFrame(data, columns=[
        "Età", "Genere", "Reddito Annuo ($)",
        "Visite Online Mensili", "Dimensione Cestino Media ($)",
        "Percentuale Acquisti Bio", "Sensibilità allo Sconto",
        "Visite Mensili al Negozio", "Punteggio Lealtà Brand",
        "Membro Carta Fedeltà", "Segmento Vero"
    ])
    df = df.sample(frac=1, random_state=stato_casuale).reset_index(drop=True)
    caratteristiche_numeriche = [
        "Età", "Reddito Annuo ($)", "Visite Online Mensili",
        "Dimensione Cestino Media ($)", "Percentuale Acquisti Bio",
        "Sensibilità allo Sconto", "Visite Mensili al Negozio",
        "Punteggio Lealtà Brand"
    ]
    X = df[caratteristiche_numeriche].copy()
    scaler = StandardScaler()
    X_scalato = scaler.fit_transform(X)
    return df, X_scalato, caratteristiche_numeriche, scaler

# ... (continua con le altre funzioni, correggendo la sintassi e la logica come sopra)

