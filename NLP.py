import pandas as pd
import numpy as np

df1 = pd.read_parquet(r"C:\Users\giaco\Desktop\AI\VenturekaRepo\VentuReka_repo\dataset\df1.parquet")
df2 = pd.read_parquet(r"C:\Users\giaco\Desktop\AI\VenturekaRepo\VentuReka_repo\dataset\df2.parquet")
df3 = pd.read_parquet(r"C:\Users\giaco\Desktop\AI\VenturekaRepo\VentuReka_repo\dataset\df3.parquet")
df4 = pd.read_parquet(r"C:\Users\giaco\Desktop\AI\VenturekaRepo\VentuReka_repo\dataset\df4.parquet")

# Leggi dati da Supabase
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = "https://nwiroxpurtitxczepabb.supabase.co"
SUPABASE_KEY = os.getenv("Supabase_API_kEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

restt = supabase.table("startups").select("*").execute()
dat = restt.data

supabase_df = pd.DataFrame(dat)



# Call OpenAPI
from openai import OpenAI
import numpy as np
import os

api_key = os.getenv("OpenAi_API_key")
client = OpenAI(api_key=api_key)

# Standardizzazione Dataset
# Dataset 1
df1_nlp = df1[[
    "name",
    "short_description",
    "long_description",
    "country",
    "year_founded",
    "industry",
    "team_size",
    "location"
]].copy()

df1_nlp["startup_success"] = None
df1_nlp["valuation"] = None
df1_nlp["employee_growth"] = None

# Dataset 3 (aggiungiamo metriche)
df3_nlp = df3[[
    "name",
    "industry",
    "valuation",
    "employee_growth",
    "country"
]].copy()

df3_nlp["short_description"] = None
df3_nlp["long_description"] = None
df3_nlp["startup_success"] = None
df3_nlp["year_founded"] = None

# Dataset 4 (success info)
df4_nlp = df4[[
    "name",
    "year_founded",
    "industry",
    "startup_success",
    "country",
    "location",
    "short_description",
    "total_funding"
]].copy()


df4_nlp["long_description"] = None
df4_nlp["valuation"] = None
df4_nlp["employee_growth"] = None
df4_nlp["team_size"] = None
# print(df4_nlp["short_description"].head(10))



# Metodo 2: proviamo a fare 60 k df e 8k df separati e poi unire i best similar
# qua crei una colonna di testo da embeddare "Startup operating in..." per df4
def build_text_from_tags(row):
    tags = row.get("short_description", "")
    if not isinstance(tags, str) or not tags.strip():
        return ""
    parts = [t.strip() for t in tags.split("|") if t.strip()]
    if not parts:
        return ""
    return "Startup operating in " + ", ".join(parts)

df60k = df4_nlp.copy()
df60k["text_for_embedding"] = df60k.apply(build_text_from_tags, axis=1)
#print(df60k["text_for_embedding"])
# invece di usare solo texts_60k "sciolto", tieni il dataframe filtrato
mask60 = df60k["text_for_embedding"].notna() & (df60k["text_for_embedding"].str.strip() != "")
df60k_valid = df60k[mask60]  # drop=False conserva indice originale
texts_60k = df60k_valid["text_for_embedding"].tolist()
 # 3) Salva mappa index originale (posizione FAISS -> index df originale)
orig_idx_60k = df60k_valid.index.to_numpy()
np.save("artifacts/orig_idx_60k.npy", orig_idx_60k)




# qua crei una colonna di testo da embeddare per df1
def build_text1(row):
    parts = [
        str(row.get("company_name", "")),
        str(row.get("industry", "")),
        str(row.get("short_description", ""))#,
        #str(row.get("long_description", "")),
    ]
    return " | ".join([p for p in parts if p and p != "None"])
df8k = df1_nlp.copy()
df8k["text_for_embedding"] = df8k.apply(build_text1, axis=1)

# df8k già contiene la colonna text_for_embedding
mask8 = df8k["text_for_embedding"].notna() & (df8k["text_for_embedding"].str.strip() != "")
df8k_valid = df8k[mask8]  # indice originale preservato
texts_8k = df8k_valid["text_for_embedding"].tolist()

# mappa posizione FAISS -> index originale df8k
orig_idx_8k = df8k_valid.index.to_numpy()
np.save("artifacts/orig_idx_8k.npy", orig_idx_8k)


import faiss
# Crea e salva embedding + index
os.makedirs("artifacts", exist_ok=True)
#np.save("artifacts/emb_8k.npy", emb_8k)
#np.save("artifacts/emb_60k.npy", emb_60k)

emb_8k = np.load("artifacts/emb_8k.npy")
emb_60k = np.load("artifacts/emb_60k.npy")
# Normalizza per cosine similarity (cosine = inner product su vettori normalizzati)
faiss.normalize_L2(emb_8k)
faiss.normalize_L2(emb_60k)

# Crea index FAISS (IP = inner product)
index_8k = faiss.IndexFlatIP(emb_8k.shape[1])
index_8k.add(emb_8k)

index_60k = faiss.IndexFlatIP(emb_60k.shape[1])
index_60k.add(emb_60k)

# Salva index
faiss.write_index(index_8k, "artifacts/index_8k.faiss")
faiss.write_index(index_60k, "artifacts/index_60k.faiss")





index_8k = faiss.read_index("artifacts/index_8k.faiss")
index_60k = faiss.read_index("artifacts/index_60k.faiss")

orig_idx_60k = np.load("artifacts/orig_idx_60k.npy")
orig_idx_8k = np.load("artifacts/orig_idx_8k.npy")

def get_name_col(df):
    for c in ["startup_name", "company_name", "name"]:
        if c in df.columns:
            return c
    raise KeyError(f"Nessuna colonna nome trovata in df. Colonne: {df.columns.tolist()}")

name_col_60k = get_name_col(df60k)
name_col_8k = get_name_col(df8k)

def embed_query(text, model="text-embedding-3-large"):
    r = client.embeddings.create(input=[text], model=model)
    q = np.array([r.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(q)
    return q

def search_60k(user_text, top_k=5):
    q = embed_query(user_text)
    scores, ids = index_60k.search(q, top_k)

    original_rows = orig_idx_60k[ids[0]]  # indici reali df60k
    out = df60k.loc[original_rows].copy() # tutte le colonne originali
    out["similarity"] = scores[0]
    out["original_index"] = original_rows
    out["faiss_rank"] = range(1, len(out) + 1)

    # opzionale: riordina per similarity decrescente
    out = out.sort_values("similarity", ascending=False)
    return out

def search_8k(user_text, top_k=5, start_k=20, max_k=500):
    q = embed_query(user_text)
    k = min(start_k, index_8k.ntotal)

    while True:
        scores, ids = index_8k.search(q, k)

        original_rows = orig_idx_8k[ids[0]]
        out = df8k.loc[original_rows].copy()   # tutte le colonne originali
        out["similarity"] = scores[0]
        out["original_index"] = original_rows
        out["faiss_rank"] = range(1, len(out) + 1)

        # deduplica per nome mantenendo il più simile (primo)
        out_unique = out.drop_duplicates(subset=[name_col_8k], keep="first")

        if len(out_unique) >= top_k or k >= min(max_k, index_8k.ntotal):
            return out_unique.head(top_k).sort_values("similarity", ascending=False)

        k = min(k * 2, index_8k.ntotal)

user_text = supabase_df["short_description"].dropna().astype(str).str.strip().iloc[-1]

res = search_60k(user_text, top_k=5)
#display(res)
res8k = search_8k(user_text)
#display(res8k)




# layer 2
res["source"] = "df4"
res8k["source"] = "df1"

#mappa allo schema canonico unico
def to_numeric_funding(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("$", "").replace(",", "").strip()
    return pd.to_numeric(s, errors="coerce")

def map_success_df1(status):
    # puoi tarare questa mappa
    if pd.isna(status):
        return np.nan
    s = str(status).lower().strip()
    if s in ["acquired", "ipo", "public"]:
        return 1.0
    if s in ["active", "operating"]:
        return 0.6
    if s in ["closed", "dead", "inactive", "failed"]:
        return 0.0
    return 0.4

current_year = pd.Timestamp.today().year

def build_canonical_df4(df):
    df["year_founded"] = pd.to_datetime(df["year_founded"], errors="coerce", dayfirst=True)
    year_founded = df["year_founded"].dt.year   
    out = pd.DataFrame({
        "peer_id": df.index.astype(str),
        "source": "df4",
        "name": df["name"] if "name" in df.columns else df.get("startup_name"),
        "similarity": pd.to_numeric(df["similarity"], errors="coerce"),
        "industry": df["industry"] if "industry" in df.columns else np.nan,
        "total_funding": df["total_funding"].apply(to_numeric_funding) if "total_funding" in df.columns else np.nan,
        "success_label": df["startup_success"] if "startup_success" in df.columns else np.nan,
        "age_years": current_year - year_founded,
        "geo_country": df["country"] if "country" in df.columns else np.nan,
        "geo_region": df["location"] if "location" in df.columns else np.nan,
        "team_size": np.nan,
        "growth_proxy": np.nan,  # non presente in df4
    })
    return out

def build_canonical_df1(df):
    year_founded = pd.to_numeric(df["year_founded"], errors="coerce") if "year_founded" in df.columns else np.nan
    team_size = pd.to_numeric(df["team_size"], errors="coerce") if "team_size" in df.columns else np.nan

    out = pd.DataFrame({
        "peer_id": df.index.astype(str),
        "source": "df1",
        "name": df["name"] if "name" in df.columns else df.get("startup_name"),
        "similarity": pd.to_numeric(df["similarity"], errors="coerce"),
        "industry": df["industry"] if "industry" in df.columns else np.nan,
        "total_funding": np.nan,  # non disponibile in df1
        "success_label": df["status"].apply(map_success_df1) if "status" in df.columns else np.nan,
        "age_years": current_year - year_founded,
        "geo_country": df["country"] if "country" in df.columns else np.nan,
        "geo_region": df["location"] if "location" in df.columns else np.nan,
        "team_size": team_size,
        "growth_proxy": team_size / (current_year - year_founded),
    })
    return out

canon4 = build_canonical_df4(res)
canon1 = build_canonical_df1(res8k)

#display(canon4.head(3))
#display(canon1.head(3))



# concat+normalizzazione+score
peers = pd.concat([canon4, canon1], ignore_index=True)

def pct_rank(s):
    return s.rank(pct=True)

peers["similarity_norm"] = pct_rank(peers["similarity"])
peers["funding_norm"] = pct_rank(peers["total_funding"])
peers["success_norm"] = pct_rank(peers["success_label"])
peers["growth_norm"] = pct_rank(peers["growth_proxy"])
#peers["eff_norm"] = pct_rank(peers["efficiency_proxy"])

# fillna neutro
for c in ["funding_norm", "success_norm", "growth_norm", "similarity_norm"]:
    peers[c] = peers[c].fillna(0.5)

# geo match opzionale (se hai paese utente)
import pycountry

def to_iso2(country_name: str):
    if not country_name or str(country_name).strip() == "":
        return None
    name = str(country_name).strip()

    # Prova match diretto
    c = pycountry.countries.get(name=name)
    if c:
        return c.alpha_2

    # Prova ricerca fuzzy (gestisce varianti tipo "United States", "USA", ecc.)
    try:
        return pycountry.countries.search_fuzzy(name)[0].alpha_2
    except:
        return None

user_country_name = supabase_df["country"].dropna().astype(str).str.strip().iloc[-1]
user_country = to_iso2(user_country_name)

peers["geo_match"] = (peers["geo_country"].astype(str).str.upper() == str(user_country).upper()).astype(float)

peers["peer_score"] = (
    0.35 * peers["similarity_norm"] +
    0.25 * peers["success_norm"] +
    0.25 * peers["growth_norm"] +
    #0.10 * peers["eff_norm"] +
    0.15 * peers["geo_match"]
)

top_peers = peers.sort_values("peer_score", ascending=False).head(10)
#display(top_peers[["name", "source", "similarity", "peer_score"]])



#aggregati + gap vs utente

def agg_stats(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"median": None, "q1": None, "q3": None, "n": 0}
    return {
        "median": float(s.median()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
        "n": int(len(s))
    }

aggregates = {
    "funding_usd": agg_stats(top_peers["funding_norm"]),
    "success_norm": agg_stats(top_peers["success_norm"]),
    "growth_norm": agg_stats(top_peers["growth_norm"]),
    #"eff_norm": agg_stats(top_peers["eff_norm"]),
    "similarity": agg_stats(top_peers["similarity_norm"])
}

# metriche utente (metti quelle reali)
funding_usd = (pd.to_numeric( supabase_df["total_funding"].dropna().astype(str).str.strip(), errors="coerce").dropna().astype(int).iloc[-1])
team_size = (pd.to_numeric(supabase_df["team_size"].dropna().astype(str).str.strip(), errors="coerce").dropna().astype(int).iloc[-1])

user_metrics = {
    "funding_usd": funding_usd * 1000000,
    "team_size": team_size,
    "country": user_country
}

peer_team_proxy = pd.to_numeric(top_peers["growth_proxy"], errors="coerce")
team_median = peer_team_proxy.dropna().median() if peer_team_proxy.notna().any() else np.nan

gaps_vs_user = {
    "funding_gap_pct": None if aggregates["funding_usd"]["median"] in [None, 0] else (user_metrics["funding_usd"] - aggregates["funding_usd"]["median"]) / aggregates["funding_usd"]["median"],
    "team_size_gap_pct": None if pd.isna(team_median) or team_median == 0 else (user_metrics["team_size"] - team_median) / team_median,
    "geo_match_rate": float(top_peers["geo_match"].mean())
}



import json
# JSON finale + invio al LLM
report_payload = {
    "query_startup": {
        "text": user_text,
        "country": user_metrics["country"],
        "funding_usd": user_metrics["funding_usd"],
        "team_size": user_metrics["team_size"]
    },
    "top_peers": top_peers[[
        "name", "source", "similarity_norm", "peer_score",
        "funding_norm", "success_norm", "growth_norm", 
        "geo_country", "geo_region"
    ]].fillna("").to_dict(orient="records"),
    "aggregates": aggregates,
    "gaps_vs_user": gaps_vs_user,
    "risks_opportunities_input": {
        "missing_data_rate": float(top_peers.isna().mean().mean()),
        "peer_count": int(len(top_peers)),
        "cross_source_mix": top_peers["source"].value_counts(normalize=True).to_dict()
    }
}

prompt = f"""
You are a senior VC strategy analyst.
Write an "AI Strategic Report" in Italian, concise but decision-oriented.

Rules:
- Use ONLY the data provided in JSON.
- If a metric is missing/null, explicitly state "dato non disponibile".
- Do not invent numbers.
- Keep numeric references clear and comparable (user vs peer).
- Prioritize actionable recommendations.

Output structure (strict):
1) Executive Summary (max 8 righe)
2) Peer Landscape (top comparable startups and why)
3) Benchmark Quantitativo (funding, team, revenue, valuation, efficiency, growth, country factors)
4) Gap Analysis (user vs peer median)
5) Strategic Actions:
   - 30 giorni (3 azioni)
   - 90 giorni (3 azioni)
   - 180 giorni (3 azioni)
6) Key Risks & Mitigations (max 5)
7) KPI da monitorare (max 8, with target direction)

Tone:
- Professional, concrete, investor-ready.
- No generic motivational language.

JSON:
{json.dumps(report_payload, ensure_ascii=False, default=str)}
"""

resp = client.responses.create(
    model="gpt-4.1",
    input=prompt
)

print(resp.output_text)


# aggiungere report a Supabase
restt = (
    supabase
    .table("startups")
    .select("id, created_at")
    .order("created_at", desc=True)
    .limit(1)
    .execute()
)

latest_supabase_id = restt.data[0]["id"] 

user_startup = supabase_df[supabase_df["id"] == latest_supabase_id].iloc[0]
startup_id = user_startup["id"]

supabase.table("startups").update({
    "report_text": resp.output_text
}).eq("id", startup_id).execute()
