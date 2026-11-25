# Data Dictionary — Field Survey & Literature Toxicology Database

## 1. Raw Field Survey (`data/raw/Base_MB_2023_ang.xlsx`, Sheet `Feuil1`)

- **Structure:** 1,123 rows representing hierarchical ethnoveterinary records from **106 surveyed herders** across Tagant (`n=57`), Brakna (`n=36`), and Trarza (`n=13`), Mauritania (August 2021 – January 2022).
- **Primary Entities:** 364 unique informant–plant citation events across 41 curated plant species (21 APG IV families; 22 traditional families), 1,118 host-exposure links, and 380 clinical-sign records.
- **Automated Curation Applied in `src/data_loader.py`:**
  - `Ricinus comminus L.` (7 citations) merged into *Ricinus communis* L. (total 30 citations).
  - Family spelling corrections: `Zygophyllceae` → `Zygophyllaceae`, `Boragynaceae` → `Boraginaceae`, `Capparidaceae` → `Capparaceae`.
  - APG IV harmonization: `Asclepiadaceae` merged into `Apocynaceae` (subfamily Asclepiadoideae).
  - Bilingual harmonization: `Sheet` → `Leaf`, `Ane` → `Donkey`, `Caprin/Ovin/Bovin/Camelin` → `Goat/Sheep/Cattle/Camel`.

## 2. Literature Toxicology Database (`data/literature/literature_toxicology_database.csv` & `.xlsx`)

Curated peer-reviewed phytochemical and veterinary toxicology dataset covering **all 41 reported species** under a strict 3-tier rubric (14 species-level confirmed at `K = 1.0`; 10 genus/subfamily or mechanical-hazard supportive at `K = 0.5`; 17 unconfirmed at `K = 0.0`), dynamically loaded by `src/vrpi.py`.

| Column | Type | Description |
|---|---|---|
| `Species_ID` | string | Unique species identifier (`SP_01`–`SP_41`, ordered by citation count) |
| `Species` | string | Accepted Latin binomial matching `src/data_loader.py` |
| `Full_Scientific_Name` | string | Scientific binomial with botanical authority |
| `Family_APGIV` | string | Harmonized APG IV botanical family (21 families) |
| `Family_Traditional` | string | Traditional family keeping *Asclepiadaceae* separate (22 families) |
| `Vernacular_Names` | string | Local Hassaniya/Mauritanian vernacular name(s) from field survey |
| `Origin` | string | Spontaneous vs. Cultivated/Introduced |
| `Survey_Citations_n` | int | Exact citation count across 106 surveyed herders (sum = 364) |
| `Chem_Confirm_K` | float | 3-tier VRPI chemical confirmation score (`1.0` = species-level confirmed; `0.5` = genus/subfamily or mechanical hazard; `0.0` = unconfirmed) |
| `Evidence_Tier` | string | Categorical evidence tier (`Confirmed (Species-Level)`, `Supportive (Genus/Subfamily-Level)`, `Supportive (Mechanical Hazard)`, or `Unconfirmed (No Published Evidence)`) |
| `Documented_Phytotoxins` | string | Documented toxic compounds, alkaloids, glycosides, or suspected mechanical hazards |
| `Chemical_Class` | string | Secondary metabolite chemical family or hazard class |
| `Target_System_Mechanism` | string | Documented physiological target system and mechanism of toxicity |
| `Documented_Animal_Evidence` | string | Livestock or experimental animal species documented in peer-reviewed studies |
| `Primary_Reference` | string | Peer-reviewed bibliographic citation(s) matching manuscript bibliography |
| `DOI_or_PMID` | string | Digital Object Identifier (DOI) or PubMed ID (PMID) |
| `Databases_Queried` | string | Bibliographic databases screened during systematic search |
| `Screening_Notes` | string | Curatorial notes, taxonomic remarks, and screening priorities |
