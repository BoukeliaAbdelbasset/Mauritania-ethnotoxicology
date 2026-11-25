"""Automated Zero-Contradiction Reproducibility & Integrity Test Suite (v1.1.0)."""

import json
import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TAB_DIR = os.path.join(BASE_DIR, "outputs", "tables")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")


def test_version_synchronization():
    with open(os.path.join(BASE_DIR, "pyproject.toml"), encoding="utf-8") as f:
        v_py = re.search(r'version\s*=\s*"([^"]+)"', f.read()).group(1)
    with open(os.path.join(BASE_DIR, "CITATION.cff"), encoding="utf-8") as f:
        v_cff = re.search(r'version:\s*"([^"]+)"', f.read()).group(1)
    with open(os.path.join(BASE_DIR, "src", "__init__.py"), encoding="utf-8") as f:
        v_init = re.search(r'__version__\s*=\s*"([^"]+)"', f.read()).group(1)
    assert v_py == v_cff == v_init == "1.1.0", f"Version mismatch: {v_py}, {v_cff}, {v_init}"


def test_literature_database_3tier_integrity():
    lit_csv = os.path.join(DATA_DIR, "literature", "literature_toxicology_database.csv")
    lit_xlsx = os.path.join(DATA_DIR, "literature", "literature_toxicology_database.xlsx")
    assert os.path.exists(lit_csv) and os.path.exists(lit_xlsx)
    lit = pd.read_csv(lit_csv)
    assert len(lit) == 41 and lit["Species"].nunique() == 41
    assert (lit["Chem_Confirm_K"] == 1.0).sum() == 14
    assert (lit["Chem_Confirm_K"] == 0.5).sum() == 10
    assert (lit["Chem_Confirm_K"] == 0.0).sum() == 17
    assert lit["Survey_Citations_n"].sum() == 364
    assert lit[lit["Chem_Confirm_K"] >= 0.5]["Survey_Citations_n"].sum() == 326
    assert not lit.astype(str).apply(lambda c: c.str.contains("Ayinke|Dynes", case=False)).any().any()


def test_unspecified_fidelity_bug_eliminated():
    s1 = pd.read_csv(os.path.join(TAB_DIR, "Supplementary_Table_S1_All_41_Species.csv"))
    unspec = s1[s1["Dominant_Specified_Toxidrome"].str.startswith("Unspecified")]
    assert len(unspec) > 0
    assert (unspec["Fidelity_F"] == 0.0).all()
    assert s1[s1["Species"] == "Adenium obesum"].iloc[0]["Fidelity_F"] == 0.0


def test_summary_metrics_and_sensitivity():
    with open(os.path.join(TAB_DIR, "summary_metrics.json"), "r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["n_informants"] == 106
    assert m["n_species"] == 41
    assert m["n_families_apgiv"] == 21
    assert m["n_citations"] == 364
    assert m["fabaceae_fiv"] == 100.0
    assert m["literature_species_level_k10"] == 14
    assert m["literature_genus_mech_k05"] == 10
    assert m["literature_unconfirmed_k00"] == 17
    assert m["literature_citations_k_ge_05_n"] == 326
    assert abs(m["literature_citations_k_ge_05_pct"] - 89.56) < 0.01
    assert m["cct_max_rhat"] < 1.05
    assert abs(m["nodf_nontied"] - 83.47) < 0.05 and m["nodf_nontied_p"] <= 0.01
    assert abs(m["nodf_strict"] - 61.24) < 0.05 and m["nodf_strict_p"] <= 0.05
    assert abs(m["risk_collapse_auc_targeted"] - 0.3942) < 0.005
    for scheme, res in m["vrpi_weight_sensitivity"].items():
        assert res["spearman_rho"] >= 0.89
        assert res["top5"][:3] == ["Indigofera senegalensis", "Ricinus communis", "Prosopis juliflora"]

    s1 = pd.read_csv(os.path.join(TAB_DIR, "Supplementary_Table_S1_All_41_Species.csv"))
    mc = s1[s1["Citations_n"] >= 2]
    assert len(mc) == 22
    assert (mc["Boot_Rank_MultiCit_Hi"] <= 22).all() and (mc["Boot_Rank_MultiCit_Lo"] >= 1).all()


def test_tables_and_figures_complete():
    sym = pd.read_csv(os.path.join(TAB_DIR, "Table3_Symptom_Concentration_Exclusivity.csv"))
    assert "Within_Sign_Concentration_Csign" in sym.columns
    assert "Flora_Wide_Exclusivity_Sflora" in sym.columns
    assert (sym["Entropy_bits"] >= 0.0).all()
    assert os.path.exists(os.path.join(TAB_DIR, "Table_Tripartite_Motifs_ExpandedSensitivity.csv"))
    figs = [f for f in os.listdir(FIG_DIR) if f.endswith(".png")]
    assert len(figs) == 12


if __name__ == "__main__":
    test_version_synchronization()
    test_literature_database_3tier_integrity()
    test_unspecified_fidelity_bug_eliminated()
    test_summary_metrics_and_sensitivity()
    test_tables_and_figures_complete()
    print("ALL V4 ZERO-CONTRADICTION TESTS PASSED.")
