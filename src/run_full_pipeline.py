"""
Main CLI Entry Point — Full Reproducible Pipeline for Toxicon Submission.

Reproduces all statistical analyses, CSV/JSON/Excel tables, and all 12 300 DPI
figures from:
  - Field survey workbook: `data/raw/Base_MB_2023_ang.xlsx`
  - Literature toxicology database: `data/literature/literature_toxicology_database.csv`

Usage:
    python src/run_full_pipeline.py
    python src/run_full_pipeline.py --seed 42 --out-dir outputs
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_survey_data, load_literature_database
from cct_model import fit_bayesian_cct_pymc
from network_analysis import (
    compute_fiv_table,
    compute_symptom_concentration_and_exclusivity,
    compute_toxidrome_icf_and_nmi,
    compute_tripartite_motifs,
    compute_tripartite_motifs_expanded,
    compute_nodf,
    bascompte_null_nodf,
    compute_risk_collapse_curve,
)
from vrpi import compute_vrpi_and_bootstrap, compute_vrpi_weight_sensitivity
from plotting import generate_all_figures


def export_supplementary_excel(path, master_df, fiv_21, fiv_22, sym_table, icf_table, motif_table, lit_df):
    wb = openpyxl.Workbook()
    header_fill = PatternFill("solid", fgColor="1F4E5F")
    zebra_fill = PatternFill("solid", fgColor="F4F7F8")
    header_font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    body_font = Font(name="Calibri", size=10, color="1F2933")
    italic_font = Font(name="Calibri", size=10, italic=True, color="1F2933")
    thin_border = Border(
        left=Side(style="thin", color="D9E2EC"),
        right=Side(style="thin", color="D9E2EC"),
        top=Side(style="thin", color="D9E2EC"),
        bottom=Side(style="thin", color="D9E2EC"),
    )
    sheets = [
        ("TableS1_All_41_Species", master_df),
        ("Literature_Database", lit_df),
        ("Table2_FIV_APGIV_21", fiv_21),
        ("Table2b_FIV_Trad_22", fiv_22),
        ("Table3_Symptom_Concentration", sym_table),
        ("Table4_Toxidrome_ICF", icf_table),
        ("Table5_Tripartite_Motifs", motif_table),
    ]
    for idx, (name, frame) in enumerate(sheets):
        ws = wb.active if idx == 0 else wb.create_sheet(title=name)
        ws.title = name
        ws.freeze_panes = "A2"
        for c_idx, col_name in enumerate(frame.columns, 1):
            cell = ws.cell(row=1, column=c_idx, value=col_name.replace("_", " "))
            cell.fill, cell.font, cell.border = header_fill, header_font, thin_border
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 26
        for r_idx, row in enumerate(frame.itertuples(index=False), 2):
            for c_idx, val in enumerate(row, 1):
                cell = ws.cell(row=r_idx, column=c_idx, value=val)
                col_n = frame.columns[c_idx - 1]
                cell.font = italic_font if col_n in ("Species", "Full_Scientific_Name") else body_font
                if r_idx % 2 == 1:
                    cell.fill = zebra_fill
                cell.border = thin_border
                if isinstance(val, (int, float, np.integer, np.floating)):
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                    if isinstance(val, (float, np.floating)):
                        cell.number_format = "0.00" if ("pct" in col_n or "Z_" in col_n) else "0.000"
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")
        for c_idx, col_name in enumerate(frame.columns, 1):
            max_len = max([len(col_name)] + [len(str(v)) for v in frame[col_name].astype(str)])
            ws.column_dimensions[get_column_letter(c_idx)].width = min(max_len + 4, 42)
        ws.auto_filter.ref = ws.dimensions
    wb.save(path)


def parse_args():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Reproduce Mauritanian ethno-toxicology analyses & figures.")
    parser.add_argument("--data", default=os.path.join(base_dir, "data", "raw", "Base_MB_2023_ang.xlsx"))
    parser.add_argument("--literature", default=os.path.join(base_dir, "data", "literature", "literature_toxicology_database.csv"))
    parser.add_argument("--out-dir", default=os.path.join(base_dir, "outputs"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    tab_dir = os.path.join(args.out_dir, "tables")
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"[1/6] Loading survey data ({args.data}) & literature database ({args.literature})...")
    df, informants, citations = load_survey_data(args.data)
    lit_df = load_literature_database(args.literature)

    n_inf = len(informants)
    n_cit = len(citations)
    species_order = citations["species"].value_counts().index.tolist()
    n_sp = len(species_order)

    # Sociodemographics
    sp_per_inf = citations.groupby("informant_id")["species"].nunique()
    informants["n_species"] = informants["informant_id"].map(sp_per_inf).fillna(0).astype(int)
    r_age, p_age = stats.pearsonr(informants["Age"], informants["n_species"])

    # FIV Tables
    fiv_21 = compute_fiv_table(citations, "family_21", n_inf)
    fiv_22 = compute_fiv_table(citations, "family_22", n_inf)
    fiv_21.to_csv(os.path.join(tab_dir, "Table2_FIV_21_APGIV_Families.csv"), index=False)
    fiv_22.to_csv(os.path.join(tab_dir, "Table2b_FIV_22_Traditional_Families.csv"), index=False)

    print("[2/6] Fitting Bayesian Cultural Consensus Theory (CCT) model...")
    Y_df = (pd.crosstab(citations["informant_id"], citations["species"]) > 0).astype(int)
    Y_df = Y_df.reindex(index=range(1, n_inf + 1), columns=species_order, fill_value=0)
    cct_res = fit_bayesian_cct_pymc(Y_df.values, n_chains=4, n_tune=1000, n_draws=2000, seed=args.seed)
    print(f"      backend={cct_res['backend']} | max R-hat={cct_res['max_rhat']:.4f} | mean competence={cct_res['c_mean'].mean():.3f}")

    print("[3/6] Computing symptom reporting exclusivity, ICF, NMI, and tripartite motifs...")
    sym_table = compute_symptom_concentration_and_exclusivity(df, n_sp, min_reports=4)
    sym_table.to_csv(os.path.join(tab_dir, "Table3_Symptom_Concentration_Exclusivity.csv"), index=False)
    icf_table, nmi_geom = compute_toxidrome_icf_and_nmi(df)
    icf_table.to_csv(os.path.join(tab_dir, "Table_ICF_Toxidromes.csv"), index=False)
    motif_table = compute_tripartite_motifs(df, n_perm=2000, min_obs=5, seed=args.seed)
    motif_table.to_csv(os.path.join(tab_dir, "Table_Tripartite_Motifs.csv"), index=False)
    motif_exp = compute_tripartite_motifs_expanded(df, n_perm=2000, min_obs=5, seed=args.seed)
    motif_exp.to_csv(os.path.join(tab_dir, "Table_Tripartite_Motifs_ExpandedSensitivity.csv"), index=False)

    print("[4/6] Evaluating bipartite nestedness (NODF) and risk-collapse robustness...")
    ph_df = df[df["host"].notna()].copy()
    mat_bin = (pd.crosstab(ph_df["species"], ph_df["host"]) > 0).astype(int).reindex(index=species_order, fill_value=0)
    nodf_obs = compute_nodf(mat_bin.values, strict_almeida_neto=False)
    nodf_strict = compute_nodf(mat_bin.values, strict_almeida_neto=True)
    nodf_null = bascompte_null_nodf(mat_bin.values, n_perm=1000, strict_almeida_neto=False, seed=args.seed)
    nodf_null_mean = float(nodf_null.mean())
    nodf_z = float((nodf_obs - nodf_null_mean) / nodf_null.std(ddof=1))
    nodf_p = float((np.sum(nodf_null >= nodf_obs) + 1) / (len(nodf_null) + 1))
    nodf_strict_null = bascompte_null_nodf(mat_bin.values, n_perm=1000, strict_almeida_neto=True, seed=args.seed)
    nodf_strict_null_mean = float(nodf_strict_null.mean())
    nodf_strict_z = float((nodf_strict - nodf_strict_null_mean) / nodf_strict_null.std(ddof=1))
    nodf_strict_p = float((np.sum(nodf_strict_null >= nodf_strict) + 1) / (len(nodf_strict_null) + 1))
    x_frac, rem_frac, auc_targeted = compute_risk_collapse_curve(mat_bin)

    print("[5/6] Computing VRPI composite ranking with literature DB & 1,000 bootstrap resamples...")
    theta_map = dict(zip(species_order, cct_res["theta_mean"]))
    theta_lo_map = dict(zip(species_order, cct_res["theta_ci_lo"]))
    theta_hi_map = dict(zip(species_order, cct_res["theta_ci_hi"]))
    cit_cnt_map = citations["species"].value_counts().to_dict()
    vrpi_df = compute_vrpi_and_bootstrap(
        df=df,
        Y_matrix=Y_df.values,
        species_order=species_order,
        citations_count_map=cit_cnt_map,
        theta_map=theta_map,
        lit_df=lit_df,
        max_hosts=int(mat_bin.sum(axis=1).max()),
        n_informants=n_inf,
        n_boot=1000,
        seed=args.seed,
    )

    meta_sp = (
        citations.groupby("species")
        .agg(
            Full_Scientific_Name=("sci_full", "first"),
            Family_APGIV=("family_21", "first"),
            Family_Traditional=("family_22", "first"),
            Vernacular_Names=("vern_raw", lambda x: ", ".join(sorted(set(x)))),
            Origin=("Origin", lambda x: ", ".join(sorted(set(x.dropna().astype(str).str.strip())))),
            Citations_n=("informant_id", "count"),
        )
        .reset_index()
        .rename(columns={"species": "Species"})
    )
    meta_sp["FC_pct"] = (meta_sp["Citations_n"] / n_inf * 100.0).round(2)
    meta_sp["Bayesian_Salience"] = meta_sp["Species"].map(theta_map).round(4)
    meta_sp["Salience_CrI_Lo"] = meta_sp["Species"].map(theta_lo_map).round(4)
    meta_sp["Salience_CrI_Hi"] = meta_sp["Species"].map(theta_hi_map).round(4)

    master_df = (
        meta_sp.merge(vrpi_df, on="Species")
        .sort_values(["Citations_n", "VRPI"], ascending=[False, False])
        .reset_index(drop=True)
    )
    master_df.insert(0, "FC_Rank", np.arange(1, n_sp + 1))
    master_df["VRPI_Rank"] = master_df["VRPI"].rank(ascending=False, method="min").astype(int)
    master_df.to_csv(os.path.join(tab_dir, "Supplementary_Table_S1_All_41_Species.csv"), index=False)

    export_supplementary_excel(
        os.path.join(tab_dir, "Supplementary_Table_S1_Complete_Analysis.xlsx"),
        master_df, fiv_21, fiv_22, sym_table, icf_table, motif_table, lit_df
    )

    vrpi_sens = compute_vrpi_weight_sensitivity(master_df, min_citations=2)
    k_ge_05_cit = int(lit_df[lit_df["Chem_Confirm_K"] >= 0.5]["Survey_Citations_n"].sum())
    summary = {
        "n_informants": n_inf,
        "men_n": int((informants["Gender"] == "Man").sum()),
        "women_n": int((informants["Gender"] == "Woman").sum()),
        "age_min": int(informants["Age"].min()),
        "age_max": int(informants["Age"].max()),
        "age_mean": round(float(informants["Age"].mean()), 2),
        "age_sd": round(float(informants["Age"].std()), 2),
        "age_vs_species_r": round(float(r_age), 4),
        "age_vs_species_p": round(float(p_age), 4),
        "n_species": n_sp,
        "n_families_apgiv": int(fiv_21.shape[0]),
        "n_families_trad": int(fiv_22.shape[0]),
        "n_citations": n_cit,
        "mean_citations_per_herder": round(float(n_cit / n_inf), 2),
        "fabaceae_fiv": float(fiv_21.loc[fiv_21["Family"] == "Fabaceae", "FIV_pct"].values[0]),
        "literature_species_level_k10": int((lit_df["Chem_Confirm_K"] == 1.0).sum()),
        "literature_genus_mech_k05": int((lit_df["Chem_Confirm_K"] == 0.5).sum()),
        "literature_unconfirmed_k00": int((lit_df["Chem_Confirm_K"] == 0.0).sum()),
        "literature_citations_k_ge_05_n": k_ge_05_cit,
        "literature_citations_k_ge_05_pct": round(k_ge_05_cit / n_cit * 100.0, 2),
        "cct_max_rhat": round(cct_res["max_rhat"], 4),
        "cct_mean_competence": round(float(cct_res["c_mean"].mean()), 3),
        "cct_competence_indigofera_r": round(float(stats.pearsonr(cct_res["c_mean"], Y_df.values[:, 0])[0]), 4),
        "cct_competence_listlen_r": round(float(stats.pearsonr(cct_res["c_mean"], Y_df.values.sum(axis=1))[0]), 4),
        "nmi_toxidrome_species": round(nmi_geom, 4),
        "ruminant_pct": round(float((ph_df["host_class"] == "ruminant").mean() * 100), 2),
        "nodf_nontied": round(nodf_obs, 2),
        "nodf_nontied_null_mean": round(nodf_null_mean, 2),
        "nodf_nontied_z": round(nodf_z, 2),
        "nodf_nontied_p": round(nodf_p, 4),
        "nodf_strict": round(nodf_strict, 2),
        "nodf_strict_null_mean": round(nodf_strict_null_mean, 2),
        "nodf_strict_z": round(nodf_strict_z, 2),
        "nodf_strict_p": round(nodf_strict_p, 4),
        "risk_collapse_auc_targeted": round(auc_targeted, 4),
        "vrpi_weight_sensitivity": vrpi_sens,
    }
    with open(os.path.join(tab_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[6/6] Rendering all 12 publication-ready figures (300 DPI)...")
    generate_all_figures(
        fig_dir, df, informants, citations, fiv_21, master_df, cct_res,
        sym_table, icf_table, motif_table, mat_bin,
        nodf_obs, nodf_null_mean, nodf_z, nodf_p,
        x_frac, rem_frac, auc_targeted, r_age, p_age
    )
    print("Reproduction complete! Key metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
