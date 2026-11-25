"""
Data loading, taxonomic standardization, and literature database ingestion.

Reads:
  1. Field survey workbook (`data/raw/Base_MB_2023_ang.xlsx`, sheet `Feuil1`).
  2. Curated peer-reviewed literature toxicology database
     (`data/literature/literature_toxicology_database.csv`).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TOXIDROME_MAP = {
    "Ballooning": "Digestive",
    "Diarrhea": "Digestive",
    "Sore mouth": "Digestive",
    "Constipation": "Digestive",
    "Hypersalivation": "Digestive",
    "Bloody diarrhea": "Digestive",
    "Loss of appetite": "Digestive",
    "Undefined": "Unspecified",
    "Blind": "Neurological",
    "Madness": "Neurological",
    "Diziness": "Neurological",
    "Tremor": "Neurological",
    "Fainting": "Neurological",
    "Paralysis": "Neurological",
    "Weight loss": "General",
    "Rheumatism": "General",
    "Fever": "General",
    "Weakness": "General",
    "Sterility": "Reproductive",
    "Abortion": "Reproductive",
    "Yellowing": "Hepato-renal",
    "Urinary retention": "Hepato-renal",
    "Yellowing eyes": "Hepato-renal",
    "Yellow of the spleen": "Hepato-renal",
    "Swelling of the liver": "Hepato-renal",
    "Swelling of the spleen": "Hepato-renal",
}

HOST_MAP = {
    "Caprin": "Goat",
    "Ovin": "Sheep",
    "Bovin": "Cattle",
    "Camelin": "Camel",
    "Donkey": "Donkey",
    "Ane": "Donkey",
    "Dog": "Dog",
    "Hyena": "Hyena",
}

ORGAN_MAP = {
    "Whole plant": "Whole plant",
    "Flower": "Flower",
    "flower": "Flower",
    "Sheet": "Leaf",
    "Fruit": "Fruit",
    "Stem": "Stem",
}

FAMILY_FIX_TRADITIONAL = {
    "Zygophyllceae": "Zygophyllaceae",
    "Boragynaceae": "Boraginaceae",
    "Capparidaceae": "Capparaceae",
}

FAMILY_FIX_APGIV = {
    **FAMILY_FIX_TRADITIONAL,
    "Asclepiadaceae": "Apocynaceae",
}


def clean_full_scientific_name(name: str) -> str:
    """Standardize known raw typographical and encoding artifacts."""
    cleaned = name.replace("Ricinus comminus", "Ricinus communis")
    cleaned = cleaned.replace("N.E.Br.Â", "N.E.Br.")
    cleaned = cleaned.replace("GonÃ§.", "Gonç.")
    cleaned = cleaned.replace("Â", "")
    return cleaned.strip()


def clean_binomial(name: str) -> str:
    """Extract standardized genus + species binomial."""
    return " ".join(clean_full_scientific_name(name).split()[:2])


def load_survey_data(excel_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and curate the Mauritanian ethno-toxicological field survey dataset."""
    df_raw = pd.read_excel(excel_path, sheet_name="Feuil1")
    df = df_raw.copy()
    df["informant_id"] = df["Number"].ffill().astype(int)
    df["citation_id"] = df["Scientific name"].notna().cumsum()
    df["sci_raw"] = df["Scientific name"].ffill().astype(str).str.strip()
    df["fam_raw"] = df["Family"].ffill().astype(str).str.strip()
    df["vern_raw"] = df["Vernacular name"].ffill().astype(str).str.strip()

    df["sci_full"] = df["sci_raw"].apply(clean_full_scientific_name)
    df["species"] = df["sci_raw"].apply(clean_binomial)
    df["family_22"] = df["fam_raw"].replace(FAMILY_FIX_TRADITIONAL)
    df["family_21"] = df["fam_raw"].replace(FAMILY_FIX_APGIV)

    df["host"] = df["Exposed animal"].astype(str).str.strip().map(HOST_MAP)
    df["host_class"] = df["host"].apply(
        lambda h: "ruminant"
        if h in {"Goat", "Sheep", "Cattle", "Camel"}
        else ("non-ruminant" if pd.notna(h) else np.nan)
    )
    df["organ"] = df["Toxic parts"].astype(str).str.strip().map(ORGAN_MAP)
    df["symptom"] = (
        df["Symptoms of poisoning"]
        .astype(str)
        .str.strip()
        .where(df["Symptoms of poisoning"].notna())
    )
    df["toxidrome"] = df["symptom"].map(TOXIDROME_MAP)
    df["obs_clean"] = (
        df["Observations"]
        .astype(str)
        .str.strip()
        .where(df["Observations"].notna())
    )

    informants = df_raw[df_raw["Number"].notna()].copy()
    informants["informant_id"] = informants["Number"].astype(int)
    citations = df[df["Scientific name"].notna()].copy()
    return df, informants, citations


def load_literature_database(csv_path: str) -> pd.DataFrame:
    """Load the curated 41-species peer-reviewed toxicology literature dataset."""
    lit_df = pd.read_csv(csv_path, encoding="utf-8")
    required_cols = {
        "Species",
        "Chem_Confirm_K",
        "Evidence_Tier",
        "Documented_Phytotoxins",
        "Primary_Reference",
    }
    missing = required_cols - set(lit_df.columns)
    if missing:
        raise ValueError(f"Literature database missing required columns: {sorted(missing)}")
    lit_df["Species"] = lit_df["Species"].astype(str).str.strip()
    lit_df["Chem_Confirm_K"] = lit_df["Chem_Confirm_K"].astype(float)
    return lit_df
