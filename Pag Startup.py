import os
import pandas as pd
import numpy as np
import seaborn as sns
from supabase import create_client, Client

SUPABASE_URL = "**"
SUPABASE_KEY = "**"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


#GRAFICO 1 PERCENTILI

# Chiama KAGGLE dataset  
df22 = pd.read_csv(r"C:\Users\giaco\Desktop\AI\VC_Data_Science\VC_code\dataset\Start-up\kaggle 'global startup success dataset'\global_startup_success_dataset.csv")

# Mappatura dei vecchi nomi verso i nuovi
rename_map = {
    "Startup Name": "name",
    "Founded Year": "year_founded",
    "Country": "country",
    "Industry": "industry",
    "Funding Stage": "funding_stage",
    "Total Funding ($M)": "total_funding",
    "Number of Employees": "team_size",
    "Annual Revenue ($M)": "revenue",
    "Valuation ($B)": "valuation"
    # Le altre colonne restano invariate
}
# Rinominare solo quelle presenti
df22 = df22.rename(columns=lambda x: rename_map.get(x, x))



def update_dataset_incremental(df):
    """
    Aggiorna il dataframe df aggiungendo solo le nuove righe presenti in Supabase.
    - Usa 'id' come chiave univoca per evitare duplicati
    - Riempie con NaN i campi mancanti
    """
    # Leggi dati da Supabase
    res = supabase.table("startups").select("*").execute()
    data = res.data
    if not data:
        # Nessun dato nuovo
        return df

    supabase_df = pd.DataFrame(data)

    # Assicurati che tutte le colonne di supabase_df siano presenti in df
    for col in supabase_df.columns:
        if col not in df.columns:
            df[col] = pd.NA  # crea colonna vuota nel df corrente

    # Assicurati che tutte le colonne di df siano presenti in supabase_df
    for col in df.columns:
        if col not in supabase_df.columns:
            supabase_df[col] = pd.NA  # crea colonna vuota nel supabase_df

    # Trova nuove righe non presenti nel df usando 'id'
    new_rows = supabase_df[~supabase_df['id'].isin(df.get('id', []))]

    # Aggiungi le nuove righe
    if not new_rows.empty:
        df = pd.concat([df, new_rows], ignore_index=True)

    return df

df22 = update_dataset_incremental(df22)



# PEERS Selezione peer group (industry + funding stage)
def get_peer_group(df, industry, funding_stage, exclude_id=None):
    peers = df22[
        (df["industry"] == industry) &
        (df["funding_stage"] == funding_stage)
    ]
    
    if exclude_id is not None:
        peers = peers[peers["id"] != exclude_id]
    
    return peers



# Calcolo dei percentili
def compute_percentiles(peer_df, user_data, metrics, min_peers=5):
    """
    Calcola percentili robusti usando ECDF (empirical cumulative distribution function)
    peer_df: DataFrame con dati dei peer
    user_data: dict o Series con valori utente
    metrics: lista delle metriche da calcolare
    min_peers: minimo numero di peer per calcolare percentile
    """
    results = {}

    for metric in metrics:
        val = user_data.get(metric)
        if metric not in peer_df.columns or pd.isna(val):
            continue

        peer_values = peer_df[metric].dropna().astype(float)

        # Guardrail: pochi peer = niente percentile
        if len(peer_values) < min_peers:
            results[metric] = {
                "value": float(val),
                "percentile": None,
                "note": f"Insufficient peer data ({len(peer_values)})",
                "peer_count": len(peer_values)
            }
            continue

        # Ordina i peer
        sorted_peers = np.sort(peer_values)
        # ECDF: percentuale di peer con valori <= user
        percentile = 100 * np.searchsorted(sorted_peers, val, side="right") / len(sorted_peers)

        results[metric] = {
            "value": float(val),
            "percentile": round(percentile, 1),
            "peer_median": round(np.median(sorted_peers), 2),
            "peer_mean": round(np.mean(sorted_peers), 2),
            "peer_count": len(peer_values)
        }

    return results

metrics = [
    "total_funding",
    "team_size",
    "revenue",
    "valuation"
]




# costruzione JSON
def build_percentile_json(results):
    metrics = []
    percentiles = []

    for metric, data in results.items():
        if data.get("percentile") is not None:
            metrics.append(metric.replace("_", " ").title())
            percentiles.append(float(round(data["percentile"], 2)))

    if not metrics:
        return None

    return {
        "metrics": metrics,
        "percentiles": percentiles,
        "reference_lines": {
            "median": 50,
            "top_quartile": 75
        }
    }



res = (
    supabase
    .table("startups")
    .select("id, created_at")
    .order("created_at", desc=True)
    .limit(1)
    .execute()
)
latest_supabase_id = res.data[0]["id"] 
user_startup = df22[df22["id"] == latest_supabase_id].iloc[0]

peer_df = get_peer_group(
    df22,
    user_startup["industry"],
    user_startup["funding_stage"],
    exclude_id=user_startup["id"]
)
percentile_results = compute_percentiles(
    peer_df,
    user_startup,
    metrics
)
percentile_json = build_percentile_json(percentile_results)



# Caricamneto risultati su Supabase
startup_id = user_startup["id"]
import json
# Converte NumPy -> Python float per JSON
percentile_json_clean = json.loads(json.dumps(percentile_json, default=float))

supabase.table("startups").update({
    "percentile_json": percentile_json_clean
}).eq("id", startup_id).execute()






## GRAFICO 2 CES AND GRWOTH INDEX

from sklearn.preprocessing import StandardScaler
from scipy.stats import percentileofscore

scaler = StandardScaler()

df1 = pd.read_csv(r"C:\Users\giaco\Desktop\AI\VC_Data_Science\VC_code\dataset\Start-up\Global Startup Accelerator Dataset\2023-07-13-yc-companies.csv")
df2 = pd.read_csv(r"C:\Users\giaco\Desktop\AI\VC_Data_Science\VC_code\dataset\Start-up\kaggle 'global startup success dataset'\global_startup_success_dataset.csv")
df3 = pd.read_csv(r"C:\Users\giaco\Desktop\AI\VC_Data_Science\VC_code\dataset\Start-up\export.csv")    

rename_map = {
    "Startup Name": "name",
    "Founded Year": "year_founded",
    "Country": "country",
    "Industry": "industry",
    "Funding Stage": "funding_stage",
    "Total Funding ($M)": "total_funding",
    "Number of Employees": "team_size",
    "Annual Revenue ($M)": "revenue",
    "Valuation ($B)": "valuation"
    # Le altre colonne restano invariate
}
# Rinominare solo quelle presenti
df2 = df2.rename(columns=lambda x: rename_map.get(x, x))

rename_df3 = {
    "company_name": "name",
    "Industry": "industry",
    "founded": "year_founded",
    "current_employees": "team_size",
    "last_employees": "last_employees"
}

df3 = df3.rename(columns=rename_df3)

# -----------------------------
# INDUSTRY GROUP MAPPING (dataset -> website form)
# -----------------------------

industry_group_map = {

    # AI
    "ai": "AI",
    "analytics": "AI",
    "3d": "AI",

    # SaaS
    "saas": "Saas",
    "tech services": "Saas",

    # FinTech
    "fintech": "FinTech",
    "finance": "FinTech",
    "banking": "FinTech",
    "insurance": "FinTech",
    "investment banks": "FinTech",
    "accounting": "FinTech",
    "investments": "FinTech",
    "venture capital": "FinTech",

    # Healthcare
    "healthcare": "Healthcare",
    "hospital/healthcare": "Healthcare",
    "health": "Healthcare",
    "digital health": "Healthcare",
    "medical": "Healthcare",
    "medical offices": "Healthcare",
    "medical equip": "Healthcare",
    "home health care services": "Healthcare",
    "fitness": "Healthcare",
    "veterinary": "Healthcare",

    # Biotech
    "biotech": "Biotech",
    "pharma": "Biotech",

    # E-commerce
    "ecommerce": "E-commerce",
    "ecommercetech": "E-commerce",
    "retail": "E-commerce",
    "consumer": "E-commerce",
    "wholesale": "E-commerce",
    "trade": "E-commerce",
    "food and beverage retail": "E-commerce",

    # FoodTech
    "foodtech": "FoodTech",
    "food": "FoodTech",
    "restaurants": "FoodTech",

    # EdTech
    "edtech": "EdTech",
    "education": "EdTech",
    "e-learning providers": "EdTech",
    "training": "EdTech",

    # Proptech
    "real estate": "Proptech",
    "real estate tech": "Proptech",
    "facilities": "Proptech",
    "architecture": "Proptech",

    # Mobility
    "transportation": "Mobility",
    "automotive": "Mobility",
    "aviation": "Mobility",
    "maritime": "Mobility",

    # Gaming / Entertainment
    "gaming": "Gaming",
    "entertainment": "Gaming",
    "media": "Gaming",
    "sports": "Gaming",
    "casinos": "Gaming",
    "arts": "Gaming",

    # Hardware / IoT
    "hardware": "Hardware",
    "iot": "Hardware",
    "electronics": "Hardware",
    "semiconductors": "Hardware",
    "machinery": "Hardware",

    # Energy / CleanTech
    "energy": "Energy",
    "energy/oil": "Energy",
    "utilities": "Energy",
    "environmental": "Cleantech",

    # Logistics
    "logistics": "Logistics",

    # Cybersecurity
    "it security": "Cybersecurity",
    "security": "Cybersecurity",

    # Tech generic
    "technology": "Tech",
    "internet": "Tech",
    "networking": "Tech",
    "devops": "Tech",
    "martech": "Tech",
    "salestech": "Tech",
    "marketing": "Tech",
    "marketing services": "Tech",
    "consulting": "Tech",

    # Industrial & Manufacturing
    "manufacturing": "Industrial",
    "industrial": "Industrial",
    "engineering": "Industrial",
    "materials": "Industrial",
    "plastics": "Industrial",
    "chemicals": "Industrial",
    "textiles": "Industrial",
    "construction": "Industrial",
    "civil": "Industrial",
    "landscaping": "Industrial",
    "mining": "Industrial",
    "personal care product manufacturing": "Industrial",

    # Deep Tech
    "research": "DeepTech",
    "defense": "DeepTech",
    "telecom": "DeepTech",

    # Legal & Professional
    "legal": "Professional",
    "legaltech": "Professional",
    "legal tech": "Professional",
    "hr": "Professional",
    "recruiting": "Professional",

    # Consumer & Lifestyle
    "apparel": "Consumer",
    "cosmetics": "Consumer",
    "consumer services": "Consumer",
    "hospitality": "Consumer",
    "content": "Consumer",
    "events": "Consumer",
    "event tech": "Consumer",

    # Public / Nonprofit
    "nonprofit": "Public",
    "nonprofit organizations": "Public",
    "non-profit organizations": "Public",
    "religious": "Public",
    "public policy offices": "Public",}

# -----------------------------
# Apply mapping to df3
# -----------------------------

# Normalizza testo
df3["industry"] = df3["industry"].astype(str).str.strip().str.lower()

# Applica mapping
df3["industry"] = df3["industry"].map(industry_group_map)

# Se non mappata → Other
df3["industry"] = df3["industry"].fillna("Other")
#print(df3["industry"].unique())




# Trasforma Yes/No in 1/0
for col in ["IPO?", "Acquired?"]:
    if col in df2.columns:
        df2[col] = df2[col].map({"Yes": 1, "No": 0})
# -----------------------------
# 1️⃣ Capital Efficiency Score (CES)
# -----------------------------
def compute_capital_efficiency(df3, user_input=None):
    """
    Calcola Capital Efficiency Score (CES) per il dataset df2 e opzionalmente per l'utente.

    Args:
        df2 (pd.DataFrame): Dataset peer con colonne 'revenue', 'valuation', 'team_size', 'total_funding'.
        user_input (dict, optional): Dizionario con gli stessi valori dello user. Default None.

    Returns:
        df2_ces (pd.DataFrame): DataFrame con CES calcolato per i peer.
        user_ces (float|None): CES calcolato per lo user, None se non calcolabile.
    """
    df = df3.copy()

    # Calcolo efficienze
    df["revenue_eff"] = df["revenue"] / df["total_funding"]
    df["valuation_eff"] = (df["valuation"] * 1000) / df["total_funding"]
    df["employee_eff"] = df["team_size"] / df["total_funding"]

    # Pulizia dati invalidi
    eff_cols = ["revenue_eff", "valuation_eff", "employee_eff"]
    df[eff_cols] = df[eff_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=eff_cols)

    # Standardizzazione
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(df[eff_cols])

    # CES peer
       # CES peer (raw score senza z-score)
    df["CES_raw"] = (
        0.4 * df["revenue_eff"] +
        0.4 * df["valuation_eff"] +
        0.2 * df["employee_eff"]
    )

    user_ces = None

    if user_input is not None:
        try:
            user_revenue_eff = user_input.get("revenue", 0) / max(user_input.get("total_funding", 1), 1)
            user_valuation_eff = (user_input.get("valuation", 0) * 1000) / max(user_input.get("total_funding", 1), 1)
            user_employee_eff = user_input.get("team_size", 1) / max(user_input.get("total_funding", 1), 1)

            user_raw = (
                0.4 * user_revenue_eff +
                0.4 * user_valuation_eff +
                0.2 * user_employee_eff
            )

            # Percentile invece di z-score
            user_ces = percentileofscore(df["CES_raw"], user_raw)

        except Exception:
            user_ces = None

    return df, float(user_ces) if user_ces is not None else None




# -----------------------------
# 2️⃣ Growth Expectation Index (GEI)
# -----------------------------
def compute_growth_expectation_index(df3, user_input):

    industry = user_input["industry"]
    funding_stage = user_input.get("funding_stage", None)
    user_team_size = user_input["team_size"]

    # -------------------------
    # 1️⃣ Filtra peer
    # -------------------------
    peers = df3[df3["industry"] == industry].copy()

    if funding_stage and "funding_stage" in df3.columns:
        peers = peers[peers["funding_stage"] == funding_stage]

    peers = peers.dropna(subset=["team_size", "employee_growth"])

    if len(peers) < 10:
        return {
            "median_growth": None,
            "gei": None,
            "confidence_band": None
        }
    
    # Rimuovi outlier estremi
    peers = peers[
        peers["employee_growth"].between(
            peers["employee_growth"].quantile(0.01),
            peers["employee_growth"].quantile(0.99)
        )
    ]



    # -------------------------
    # 3️⃣ Calcola crescita attesa per bucket simile di team size
    # -------------------------
    # Trova peer con team size simile (+/- 30%)
    similar_peers = peers[
        peers["team_size"].between(
            user_team_size * 0.7,
            user_team_size * 1.3
        )
    ]

    if len(similar_peers) < 5:
        similar_peers = peers  # fallback globale industry

    median_growth = similar_peers["employee_growth"].median()




    # -------------------------
    # 4️⃣ GEI = posizione percentile dello user_team_size
    # rispetto alla distribuzione crescita attesa
    # -------------------------
    gei = percentileofscore(
        peers["employee_growth"],
        median_growth
    )

    return {
    "median_growth": float(median_growth),
    "gei": float(gei),
    "confidence_band": {
        "p25": float(peers["employee_growth"].quantile(0.25)),
        "p75": float(peers["employee_growth"].quantile(0.75))
    }
}



# calcola compute_outcome_likelihood
# -----------------------------
def compute_outcome_likelihood(df2, industry, funding_stage):
    subset = df2[
        (df2["industry"] == industry) &
        (df2["funding_stage"] == funding_stage)
    ]

    if len(subset) == 0:
        return {"IPO": None, "Acquired": None, "Stagnant": None}

    ipo_rate = subset["IPO?"].mean()
    acq_rate = subset["Acquired?"].mean()

    stagnant_rate = (
        (subset["IPO?"] == 0) &
        (subset["Acquired?"] == 0)
    ).mean()

    return {
        "IPO": float(ipo_rate),
        "Acquired": float(acq_rate),
        "Stagnant": float(stagnant_rate)
    }




# -----------------------------
# Build JSON finale per startup user
# -----------------------------
def build_startup_analysis_json(user_input, df2, df3, current_year=2026):
    peer_subset = df2[
    (df2["industry"] == user_input["industry"]) &
    (df2["funding_stage"] == user_input["funding_stage"])
]
    df2_ces, ces_value = compute_capital_efficiency(peer_subset, user_input)
    user_years = current_year - user_input["year_founded"]
    gei_data = compute_growth_expectation_index(df3, user_input)
    outcome_probs = compute_outcome_likelihood(df2, user_input["industry"], user_input["funding_stage"])
    
    return {
        "startup_profile": {
            "company_name": user_input["name"],
            "industry": user_input["industry"],
            "country": user_input["country"],
            "funding_stage": user_input["funding_stage"],
            "years_since_founded": user_years
        },
        "metrics": {
            "capital_efficiency_score": ces_value,
            "growth_expectation_index": gei_data["gei"]
        },
        "historical_outcomes": outcome_probs,
        "positioning": {
            "x_axis": "Capital Efficiency Score",
            "y_axis": "Growth Expectation Index"
        },
    }




# -----------------------------
# Leggi la startup inserita dall'utente da Supabase
# -----------------------------
res = supabase.table("startups").select("*").order("created_at", desc=True).limit(1).execute()
data = res.data

if not data:
    print("Nessuna startup trovata su Supabase")
else:
    # Prendiamo il primo record più recente
    user_startup = data[0]  # dizionario con chiavi come companyName, yearFounded, ecc.

    # Converti eventuali valori numerici se necessario e gestisci None
    year_founded = user_startup.get("year_founded")
    team_size = user_startup.get("team_size")
    
    user_input = {
        "name": user_startup.get("name") or "Unknown",
        "year_founded": year_founded if year_founded is not None else 2026,  # fallback all'anno corrente
        "country": user_startup.get("country") or "Unknown",
        "industry": user_startup.get("industry") or "Unknown",
        "funding_stage": user_startup.get("funding_stage") or "Unknown",
        "total_funding": user_startup.get("total_funding") or 0,
        "revenue": user_startup.get("revenue") or 0,
        "valuation": user_startup.get("valuation") or 0,
        "team_size": team_size if team_size is not None else 1,  # fallback a 1
        "short_description": user_startup.get("short_description") or ""
    }

    # Calcola gli anni dalla fondazione in sicurezza
    current_year = 2026
    user_input["years_since_founded"] = current_year - user_input["year_founded"] if user_input["year_founded"] else 0

    # -----------------------------
    # Costruisci il JSON con le metriche
    # -----------------------------
    startup_json = build_startup_analysis_json(user_input, df2, df3)

    supabase.table("startups").update({
    "ces_gei": startup_json
    }).eq("id", startup_id).execute()







#GRAFICO 3 MAPPA CALCULATE COUNTRY METRICS

df2 = pd.read_parquet(r"C:\Users\giaco\Desktop\AI\VenturekaRepo\VentuReka_repo\dataset\df2.parquet")
df3 = pd.read_parquet(r"C:\Users\giaco\Desktop\AI\VenturekaRepo\VentuReka_repo\dataset\df3.parquet")
df4 = pd.read_parquet(r"C:\Users\giaco\Desktop\AI\VenturekaRepo\VentuReka_repo\dataset\df4.parquet")



user_industry = user_startup["industry"]
user_stage = user_startup["funding_stage"]
user_year = user_startup["year_founded"]
year_window = 3


# ---- Industry normalization ----
df4["industry"] = df4["industry"].astype(str)
df2["industry"] = df2["industry"].astype(str)
df3["industry"] = df3["industry"].astype(str)

df4["year_founded"] = (
    pd.to_datetime(df4["year_founded"], errors="coerce")
    .dt.year
    .astype("Int64")
)

#qui sotto ci sarebbe un problema, per cui non capisco perchè con il rename_map non funziona, quindi lo faccio a mano
df2["year_founded"] = df2["year_founded"].astype(int)

# ---- Filter peer group ----
df4_peer = df4[
    df4["industry"].str.contains(user_industry, case=False, na=False) #&
    #df4["year_founded"].between(user_year - year_window, user_year + year_window)
].copy()

df2_peer = df2[
    (df2["industry"].str.contains(user_industry, case=False, na=False)) #&
    #(df2["funding_stage"].str.contains(user_stage, case=False, na=False)) &
    #(df2["year_founded"].between(user_year - year_window, user_year + year_window))
].copy()

df3_peer = df3[
    df3["industry"].str.contains(user_industry, case=False, na=False) #&
    #df3["year_founded"].between(user_year - year_window, user_year + year_window)
].copy()



# -------------------------
# DOMANDA 1 — Dove ho maggiore probabilità di successo?
# -------------------------
# Converti success/fail in 1/0
df4_peer["startup_success_binary"] = (
    df4_peer["startup_success"]
    .str.lower()  # sicurezza contro maiuscole
    .map({"success": 1, "fail": 0})
).copy()

# Calcolo success rate per paese
success_country = (
    df4_peer
        .groupby("country")
        .agg(
            success_rate=("startup_success_binary", "mean"),
            number_of_startups=("country", "count")
        )
        .reset_index()
)

# Converti IPO/acquired in 1/0
for col in ["ipo", "acquired"]:
    if col in df2_peer.columns:
        # Mappa yes/no → 1/0
        df2_peer[col] = df2_peer[col].map({"Yes": 1, "No": 0}).copy()
        # Assicurati che sia numerica
        df2_peer[col] = pd.to_numeric(df2_peer[col], errors="coerce").copy()

exit_country = (
    df2_peer.groupby("country")
        .agg(ipo_rate=("ipo", "mean"),
             acquisition_rate=("acquired", "mean"))
        .reset_index()
)



# -------------------------
# DOMANDA 2 — Dove il funding medio è più alto?
# -------------------------
def p75_safe(x):
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    return np.percentile(x, 75)
# Converti in numerico (già fatto)
df2_funding = df2_peer[['country', 'total_funding']].copy()
df2_funding['total_funding'] = pd.to_numeric(df2_funding['total_funding'], errors='coerce') * 1000000

df3_funding = df3_peer[['country', 'total_funding']].copy()
df3_funding['total_funding'] = pd.to_numeric(df3_funding['total_funding'], errors='coerce') * 1000000

df4_funding = df4_peer[['country', 'total_funding']].copy()
df4_funding['total_funding'] = pd.to_numeric(df4_funding['total_funding'], errors='coerce')




# Merge verticale (concat)
funding_all = pd.concat([df2_funding, df3_funding, df4_funding], ignore_index=True)

funding_all = funding_all.dropna(subset=['total_funding'])

funding_country_4 = (
    funding_all.groupby('country')
        .agg(
            median_funding=('total_funding', 'median'),
            p75_funding=('total_funding', lambda x: np.percentile(x, 75)),
            p90_funding=('total_funding', lambda x: np.percentile(x, 90)),
            mean_funding=('total_funding', 'mean'),
            count_startups=('total_funding', 'count')
        )
        .reset_index()
)




# -------------------------
# DOMANDA 3 — Dove la crescita è più elevata?
# -------------------------

df3_peer["employee_growth"] = pd.to_numeric(df3_peer["employee_growth"], errors="coerce")
df3_peer["total_funding"] = pd.to_numeric(df3_peer["total_funding"], errors="coerce")

df3_peer["growth_per_funding"] = df3_peer["employee_growth"] / df3_peer["total_funding"]

growth_country = (
    df3_peer.groupby("country")
        .agg(median_employee_growth=("employee_growth", "median"),
             median_growth_per_funding=("growth_per_funding", "median"))
        .reset_index()
)



# -------------------------
# DOMANDA 4 — Dove il capitale per startup è migliore?
# -------------------------

capital_density = (
    df4_peer.groupby("country")
        .agg(total_funding_country=("total_funding", "sum"),
             number_of_startups=("country", "count"),
             successful_startups=("startup_success_binary", "sum"))
        .reset_index()
)

capital_density["capital_density"] = (
    capital_density["total_funding_country"] /
    capital_density["number_of_startups"]
)

capital_density["funding_per_success"] = (
    capital_density["total_funding_country"] /
    capital_density["successful_startups"].replace(0, np.nan)
)



# ============================================================
# STEP 5 — MERGE ALL METRICS
# ============================================================

country_metrics = success_country \
    .merge(exit_country, on="country", how="left") \
    .merge(funding_country_4, on="country", how="left") \
    .merge(growth_country, on="country", how="left") \
    .merge(capital_density[["country", "capital_density"]], on="country", how="left")

# Remove small samples
country_metrics = country_metrics[country_metrics["number_of_startups"] >= 30]



# ============================================================
# STEP 6 — NORMALIZE TO PERCENTILE 0–100
# ============================================================

def percentile(series):
    return series.rank(pct=True) * 100

country_metrics["success_score"] = percentile(country_metrics["success_rate"])
country_metrics["funding_score"] = percentile(country_metrics["median_funding"])
country_metrics["growth_score"] = percentile(country_metrics["median_employee_growth"])
country_metrics["capital_score"] = percentile(country_metrics["capital_density"])




# ============================================================
# STEP 7 — GEOGRAPHIC ATTRACTIVENESS COMPOSITE INDEX
# ============================================================

country_metrics["geographic_attractiveness_composite_index"] = (
    0.35 * country_metrics["success_score"] +
    0.35 * country_metrics["funding_score"] +
    #0.20 * country_metrics["growth_score"] + non ci sono abbastanza dati per growth, quindi lo escludo
    0.30 * country_metrics["capital_score"]
)

# Ranking
country_metrics = country_metrics.sort_values(
    "geographic_attractiveness_composite_index",
    ascending=False
).reset_index(drop=True)

country_metrics["ranking"] = country_metrics.index + 1

# ============================================================
# FINAL OUTPUT
# ============================================================

final_output = country_metrics[[
    "country",
    "geographic_attractiveness_composite_index",
    "ranking",
    "success_rate",
    "median_funding",
    "median_employee_growth",
    "capital_density"
]]




# ============================================================
# SAVE JSON
# ============================================================

final_output = final_output.replace({np.nan: None})

final_output.to_json(
    "geographic_attractiveness_index.json",
    orient="records",
    indent=2
)
map_data = final_output.replace({np.nan: None}).to_dict(orient="records")
country_metrics_df = pd.read_json("geographic_attractiveness_index.json")



# Add to Supabase table
supabase.table("startups").update({
    "map": {
        "industry": user_industry,
        "year": user_year,
        "results": map_data
    }
}).eq("id", startup_id).execute()
