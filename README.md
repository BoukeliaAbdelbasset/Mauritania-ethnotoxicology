# Mauritanian Livestock Ethno-Toxicology — Reproducible Code & Literature Database (v1.1.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI: Reproducibility](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF.svg)](.github/workflows/ci.yml)
[![Target Journal: Toxicon](https://img.shields.io/badge/Journal-Toxicon%20(Elsevier)-1F4E5F.svg)](https://www.sciencedirect.com/journal/toxicon)

Official, audit-proof reproducible Python codebase (`v1.1.0`), field survey dataset, and peer-reviewed **3-tier literature toxicology database** accompanying the revised manuscript:

> **Survey of plants reported as toxic to livestock in Mauritania: An ethno-toxicological study**  
> Mariem Cheikh Ahmed, Latifa Khattabi, Abdelbasset Boukelia\*, Ndongo Diouf, Mohamed Lemine Babe, Abdellahi El Ghassem, Aliou Ba, Ahmedou Soule, and Ahmedou M. Vadel Salihi.

## Key Verified Outputs (`--seed 42`)

- **Survey Inventory (`data/raw/Base_MB_2023_ang.xlsx`)**: 106 herders (Tagant 57, Brakna 36, Trarza 13; age 15–75 years), 364 plant-citation events across 41 curated species and 21 APG IV families (Fabaceae informant-based FIV = 100.00%, 106/106), and 1,118 host-exposure records (99.11% ruminants).
- **3-Tier Literature Toxicology Database (`data/literature/literature_toxicology_database.csv` & `.xlsx`)**:
  - **Tier 1 ($K_j = 1.0$, 14 species; 120/364 citations = 32.97%)**: Species-level peer-reviewed phytotoxin or veterinary poisoning confirmation.
  - **Tier 2 ($K_j = 0.5$, 10 species; 206/364 citations = 56.59%)**: Genus/subfamily-level toxicity (9 species) or documented mechanical hazard (*Aristida funiculata* awns, 1 species). Combined $K_j \ge 0.5$: **326/364 citations (89.56%)**.
  - **Tier 3 ($K_j = 0.0$, 17 species; 38/364 citations = 10.44%)**: No published toxicological evidence identified (prioritized for phytochemical screening).
- **Modular Pipeline (`src/`)**:
  1. `src/data_loader.py` — Automated taxonomic spelling curation, APG IV harmonization, and dynamic loading of survey + literature datasets.
  2. `src/cct_model.py` — Bayesian CCT-inspired continuous-salience hierarchical agreement model with identical `TruncatedBeta(2, 2)` priors on `[0.5, 1.0]` across PyMC v5 NUTS, exact NumPy/SciPy Gibbs ($\max \hat{R} = 1.0064$), and fast EM-MAP (`fit_cct_em_map`, $r = 0.9986$ vs. Gibbs) for bootstrap resampling.
  3. `src/network_analysis.py` — Informant-based FIV, dual symptom entropy metrics (`Within_Sign_Concentration_Csign` and `Flora_Wide_Exclusivity_Sflora` in `Table3_Symptom_Concentration_Exclusivity.csv`), toxidrome ICF & NMI ($0.33$), tripartite motif permutations (primary $n=364$ + expanded $n=393$ unique tuples sensitivity), non-tied ($83.47, p=0.001$) & strict Almeida-Neto ($61.24, p=0.038$) NODF under Bascompte et al. (2003) Null Model 2, and risk-collapse robustness ($R = 0.394$ vs. $0.500$).
  4. `src/vrpi.py` — Five-criterion VRPI **excluding `Unspecified`/`Undefined` signs from clinical fidelity ($F_j$)**, 1,000 informant-level EM-CCT bootstrap resamples (aligned to the 22-species multi-citation pool), and formal weight-sensitivity audit (Spearman $\rho = 0.895\text{–}0.972$; 100% Top-3 invariance).
  5. `src/plotting.py` — Renders all 12 publication-ready 300 DPI PNG figures (`outputs/figures/`, including `Fig5_Symptom_Reporting_Concentration.png`).
  6. `tests/test_pipeline.py` & `.github/workflows/ci.yml` — Automated zero-contradiction verification suite.

## One-Command Reproduction

```bash
# Option A: Virtualenv / Pip
bash run_reproduction.sh

# Option B: Conda (Ubuntu/Linux)
conda env create -f environment.yml
conda activate mauritania-ethnotox
python src/run_full_pipeline.py --seed 42
python tests/test_pipeline.py
```

## License & Citation

Released under the [MIT License](LICENSE). See [`CITATION.cff`](CITATION.cff) (`v1.1.0`) for software citation metadata.
