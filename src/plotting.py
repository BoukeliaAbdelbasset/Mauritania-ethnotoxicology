"""
Publication-grade 300 DPI visualization suite (Figures 1–12) for Toxicon.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

TEAL = "#3B6E7C"
ORANGE = "#CC7A3E"
GREEN = "#4F7F42"
RED = "#A83B3B"
PURPLE = "#7F5A96"
SLATE = "#5F6B7A"


def generate_all_figures(
    fig_dir: str,
    df: pd.DataFrame,
    informants: pd.DataFrame,
    citations: pd.DataFrame,
    fiv_21: pd.DataFrame,
    master_df: pd.DataFrame,
    cct_res: dict,
    sym_table: pd.DataFrame,
    icf_table: pd.DataFrame,
    motif_table: pd.DataFrame,
    mat_bin: pd.DataFrame,
    nodf_obs: float,
    nodf_null_mean: float,
    nodf_z: float,
    nodf_p: float,
    x_frac: np.ndarray,
    rem_frac: np.ndarray,
    auc_targeted: float,
    r_age: float,
    p_age: float,
) -> None:
    os.makedirs(fig_dir, exist_ok=True)

    # --- FIG 1: Sociodemographic profile ---
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    bins = np.arange(10, 90, 10)
    men_h, _ = np.histogram(informants.loc[informants["Gender"] == "Man", "Age"], bins=bins)
    wom_h, _ = np.histogram(informants.loc[informants["Gender"] == "Woman", "Age"], bins=bins)
    y_pos = bins[:-1] + 5
    axes[0].barh(y_pos, men_h, height=7.5, color=TEAL, label="Men (n=85)")
    axes[0].barh(y_pos, -wom_h, height=7.5, color=ORANGE, label="Women (n=21)")
    axes[0].axvline(0, color="#222222", lw=0.9)
    axes[0].set_xlim(-26, 34)
    axes[0].set_xticks([-20, -10, 0, 10, 20, 30])
    axes[0].set_xticklabels(["20", "10", "0", "10", "20", "30"])
    axes[0].set_xlabel("← Women    |    Men →")
    axes[0].set_ylabel("Age (years)")
    axes[0].set_title("Age–sex structure (N = 106)", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7.6, loc="lower left")

    edu_order = ["Illiterate", "Coranic", "Primary", "Middle", "High", "University"]
    edu_counts = informants["Education level"].value_counts().reindex(edu_order, fill_value=0)
    edu_colors = [TEAL, ORANGE, GREEN, PURPLE, RED, "#3B7A9E"]
    axes[1].bar(edu_order, edu_counts.values, color=edu_colors)
    axes[1].set_xticklabels(["Illiterate", "Koranic", "Primary", "Middle", "High", "University"], rotation=38, ha="right")
    axes[1].set_ylabel("Number of informants")
    axes[1].set_title("Education level", fontweight="bold")

    axes[2].scatter(informants["Age"], informants["n_species"], color=TEAL, alpha=0.65, edgecolors="white", lw=0.4, s=40)
    m_sl, b_int = np.polyfit(informants["Age"], informants["n_species"], 1)
    xs = np.linspace(informants["Age"].min(), informants["Age"].max(), 100)
    axes[2].plot(xs, m_sl * xs + b_int, color=ORANGE, lw=2.0)
    axes[2].text(0.05, 0.88, f"r = {r_age:.2f} (p = {p_age:.3f})", transform=axes[2].transAxes, fontsize=9, color="#333333")
    axes[2].set_xlabel("Age (years)")
    axes[2].set_ylabel("Species cited")
    axes[2].set_title("Knowledge vs. age", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig1_Sociodemographic_Profile.png"))
    plt.close(fig)

    # --- FIG 2: Family Importance Value (APG IV) ---
    top12_fam = fiv_21.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    y_f = np.arange(len(top12_fam))
    ax.hlines(y_f, 0, top12_fam["FIV_pct"], color="#E5E9EC", lw=3.0, zorder=1)
    norm_c = Normalize(vmin=0, vmax=12)
    sizes = 90 + top12_fam["Species_Richness"] * 48
    sc = ax.scatter(
        top12_fam["FIV_pct"], y_f, s=sizes,
        c=top12_fam["Species_Richness"], cmap=plt.cm.YlGnBu, norm=norm_c,
        edgecolors="#1F2933", lw=0.8, zorder=3
    )
    for yv, xv, spn in zip(y_f, top12_fam["FIV_pct"], top12_fam["Species_Richness"]):
        ax.text(xv + (3.2 if xv < 95 else 4.2), yv, f"{spn} sp.", va="center", fontsize=8.8, color="#4A5568")
    ax.set_yticks(y_f)
    ax.set_yticklabels(top12_fam["Family"].tolist(), fontsize=10, fontstyle="italic")
    ax.set_xlim(0, 114)
    ax.set_xlabel("Family Importance Value (% of unique informants citing the family; N = 106)")
    ax.set_title("Botanical family spectrum of plants reported as toxic (APG IV; top 12 of 21 families)", fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Species richness")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig2_Family_Importance_Value.png"))
    plt.close(fig)

    # --- FIG 3: Bayesian Consensus Salience vs. Raw FC ---
    top14 = master_df.sort_values("Bayesian_Salience", ascending=False).head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    y_s = np.arange(len(top14))
    ax.hlines(y_s, top14["Salience_CrI_Lo"], top14["Salience_CrI_Hi"], color=TEAL, lw=2.2, zorder=2)
    ax.scatter(top14["Bayesian_Salience"], y_s, color=TEAL, s=55, zorder=3, label="Bayesian consensus salience (95% CrI)")
    ax.scatter(top14["FC_pct"] / 100.0, y_s, color=ORANGE, marker="D", s=45, zorder=4, label="Raw Frequency of Citation (FC)")
    ax.set_yticks(y_s)
    ax.set_yticklabels(top14["Species"].tolist(), fontsize=9.5, fontstyle="italic")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Perceived toxicity salience / citation proportion (0–1)")
    ax.set_title("Consensus-weighted perceived toxicity salience vs. raw citation frequency\n(Bayesian CCT-inspired agreement model; top 14 of 41 species)", fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig3_Bayesian_Consensus_Salience.png"))
    plt.close(fig)

    # --- FIG 4: Informant Cultural Competence ---
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    c_vals = cct_res["c_mean"]
    ax.hist(c_vals, bins=18, color=TEAL, edgecolor="white", lw=0.8)
    ax.axvline(c_vals.mean(), color=ORANGE, ls="--", lw=2.2, label=f"mean = {c_vals.mean():.2f}")
    ax.set_xlabel("Posterior informant cultural competence (consensus alignment)")
    ax.set_ylabel("Number of herders")
    ax.set_title("Posterior distribution of informant cultural competence (N = 106)", fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig4_Informant_Competence.png"))
    plt.close(fig)

    # --- FIG 5: Symptom-to-Plant Reporting Exclusivity ---
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    sc5 = ax.scatter(
        sym_table["Reports_n"],
        sym_table["Within_Sign_Concentration_Csign"],
        s=75 + sym_table["Cited_Taxa_Ks"] * 22,
        c=sym_table["Entropy_bits"],
        cmap="viridis_r",
        edgecolors="#1F2933",
        lw=0.8,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.axhline(0.5, color=SLATE, ls=":", lw=1.0)
    ax.axvline(22, color=SLATE, ls=":", lw=1.0)
    for _, r in sym_table.iterrows():
        lbl = r["Clinical_Sign"]
        if lbl == "Ballooning":
            dx, dy = 0.68, 0.055
        elif lbl == "Undefined":
            dx, dy = 0.78, 0.068
        elif lbl == "Diarrhea":
            dx, dy = 1.13, 0.018
        else:
            dx, dy = 1.09, 0.018
        ax.text(r["Reports_n"] * dx, r["Within_Sign_Concentration_Csign"] + dy, lbl, fontsize=9, color="#1F2933")
    ax.text(0.97, 0.94, "rare &\nconcentrated", transform=ax.transAxes, ha="right", va="top", color=GREEN, style="italic", fontsize=9)
    ax.text(0.97, 0.06, "common &\ndispersed", transform=ax.transAxes, ha="right", va="bottom", color=RED, style="italic", fontsize=9)
    ax.set_xlim(3.2, 220)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Reports citing the clinical sign (logarithmic scale)")
    ax.set_ylabel(r"Within-sign reporting concentration ($1 - H / \log_2 K_s$)")
    ax.set_title("Within-sign reporting concentration of herder-cited clinical signs", fontweight="bold")
    cbar5 = fig.colorbar(sc5, ax=ax, pad=0.02)
    cbar5.set_label("Species attribution entropy (bits)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig5_Symptom_Reporting_Concentration.png"))
    plt.close(fig)

    # --- FIG 6: Bipartite Exposure Network ---
    top16_sp = master_df.head(16)["Species"].tolist()
    sub_ph = df[(df["species"].isin(top16_sp)) & (df["host"].notna())]
    edges = sub_ph.groupby(["species", "host"]).size().reset_index(name="w")
    hosts_order = ["Sheep", "Goat", "Donkey", "Cattle", "Camel"]
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    p_pos = dict(zip(top16_sp, np.linspace(0.95, 0.05, len(top16_sp))))
    h_pos = dict(zip(hosts_order, [0.90, 0.68, 0.48, 0.28, 0.08]))
    max_w = edges["w"].max()
    for _, e in edges.iterrows():
        if e["host"] not in h_pos:
            continue
        col = TEAL if e["host"] != "Donkey" else ORANGE
        ax.plot([0.36, 0.84], [p_pos[e["species"]], h_pos[e["host"]]], color=col, lw=0.4 + 3.8 * (e["w"] / max_w), alpha=0.22 + 0.65 * (e["w"] / max_w), zorder=1)
    p_totals = citations.set_index("species").groupby(level=0).size()
    for sp in top16_sp:
        ax.scatter(0.36, p_pos[sp], s=80 + p_totals[sp] * 14, facecolor="white", edgecolor=TEAL, lw=1.6, zorder=3)
        ax.text(0.31, p_pos[sp], sp, ha="right", va="center", fontsize=8.8, fontstyle="italic")
    h_totals = sub_ph["host"].value_counts()
    for h in hosts_order:
        col = TEAL if h != "Donkey" else ORANGE
        ax.scatter(0.84, h_pos[h], s=120 + h_totals.get(h, 1) * 7.5, facecolor=col, edgecolor="#1F2933", lw=1.0, zorder=3)
        ax.text(0.90, h_pos[h], h, ha="left", va="center", fontsize=10, fontweight="bold")
    ax.text(0.36, 1.02, "PLANTS (Top 16)", ha="center", fontweight="bold", color=SLATE)
    ax.text(0.84, 1.02, "HOSTS", ha="center", fontweight="bold", color=SLATE)
    ax.plot([], [], color=TEAL, lw=2.5, label="Ruminant host")
    ax.plot([], [], color=ORANGE, lw=2.5, label="Non-ruminant host")
    ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.06))
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(-0.02, 1.06)
    ax.axis("off")
    ax.set_title("Plant–host exposure network (top 16 cited species; all 41 in Fig. 9)", fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig6_Exposure_Network.png"))
    plt.close(fig)

    # --- FIG 7: Toxidrome ICF ---
    icf_plot = icf_table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    y7 = np.arange(len(icf_plot))
    ax.barh(y7, icf_plot["ICF"], color=plt.cm.viridis(np.linspace(0.35, 0.88, len(icf_plot))), edgecolor="#222222", lw=0.7)
    ax.axvline(0.5, color=SLATE, ls=":", lw=1.0, zorder=0)
    for yv, r in zip(y7, icf_plot.itertuples()):
        ax.text(r.ICF + 0.015, yv, f"ICF = {r.ICF:.2f} ({r.Reports_Nur} rep / {r.Taxa_Nt} taxa)", va="center", fontsize=8.8, zorder=4, bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.2))
    ax.set_yticks(y7)
    ax.set_yticklabels(icf_plot["Toxidrome"], fontsize=10)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Informant Consensus Factor (ICF)")
    ax.set_title("Herder agreement on physiological toxidromes (N = 106)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig7_Toxidrome_ICF.png"))
    plt.close(fig)

    # --- FIG 8: Tripartite Motifs ---
    m_plot = motif_table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    y8 = np.arange(len(m_plot))
    bar_cols = [GREEN if z > 1.96 else (RED if z < -1.96 else SLATE) for z in m_plot["Z_score"]]
    ax.barh(y8, m_plot["Z_score"], color=bar_cols, edgecolor="#222222", lw=0.7)
    ax.axvline(0, color="#222222", lw=0.9)
    ax.axvline(1.96, color=GREEN, ls="--", lw=1.0, alpha=0.8)
    ax.axvline(-1.96, color=RED, ls="--", lw=1.0, alpha=0.8)
    for yv, r in zip(y8, m_plot.itertuples()):
        ax.text(r.Z_score + (0.18 if r.Z_score >= 0 else -0.18), yv, f"×{r.Enrichment:.2f}", va="center", ha=("left" if r.Z_score >= 0 else "right"), fontsize=8.5, color="#333333")
    ax.set_yticks(y8)
    ax.set_yticklabels(m_plot["Motif"], fontsize=9.2)
    ax.set_xlim(-9.2, 9.5)
    ax.set_xlabel("Enrichment z-score vs. degree-preserving null (2,000 permutations)")
    ax.set_title("Stereotyped toxicity motifs: organ → host-class → toxidrome", fontweight="bold")
    ax.legend(handles=[Patch(color=GREEN, label="Over-represented (z > 1.96)"), Patch(color=RED, label="Under-represented (z < −1.96)")], frameon=False, loc="lower right", fontsize=8.8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig8_Tripartite_Motifs.png"))
    plt.close(fig)

    # --- FIG 9: Packed 41x7 Nestedness Matrix ---
    row_order = mat_bin.sum(axis=1).sort_values(ascending=False).index
    col_order = mat_bin.sum(axis=0).sort_values(ascending=False).index
    M_packed = mat_bin.loc[row_order, col_order]
    fig, ax = plt.subplots(figsize=(7.5, 9.6))
    ax.imshow(M_packed.values, cmap="Blues", aspect="auto", interpolation="nearest", vmin=0, vmax=1.25)
    ax.plot(np.arange(len(col_order)), M_packed.sum(axis=0).values - 0.5, color=ORANGE, lw=2.3, label="Perfect-nestedness isocline")
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order.tolist(), fontsize=7.6, fontstyle="italic")
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order, rotation=38, ha="right", fontsize=9.5)
    ax.set_title(f"Nested structure of plant → host exposure (all 41 species)\nNon-tied NODF = {nodf_obs:.2f} (null = {nodf_null_mean:.2f}, z = {nodf_z:.2f}, p = {nodf_p:.3f})", fontweight="bold", fontsize=10.5)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig9_Nestedness_Matrix.png"))
    plt.close(fig)

    # --- FIG 10: Risk-Collapse Curve ---
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(x_frac * 100, rem_frac * 100, color=RED, lw=2.5, label=f"Generalist-first removal (R = {auc_targeted:.3f})")
    ax.plot([0, 100], [100, 0], color=TEAL, ls="--", lw=2.2, label="Random removal (R = 0.500)")
    ax.fill_between(x_frac * 100, rem_frac * 100, (1 - x_frac) * 100, color=ORANGE, alpha=0.14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% of toxic plant species removed (control effort)")
    ax.set_ylabel("% of plant–host exposure links remaining")
    ax.set_title("Risk-collapse robustness: targeted vs. random plant control (all 41 species)", fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig10_Risk_Collapse_Curve.png"))
    plt.close(fig)

    # --- FIG 11: VRPI Top 15 Multi-Citation Ranking ---
    top15_v = master_df[master_df["Citations_n"] >= 2].sort_values("VRPI", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    y11 = np.arange(len(top15_v))
    c_w = top15_v["Consensus_C"] * 0.30
    f_w = top15_v["Fidelity_F"] * 0.20
    l_w = top15_v["Lethality_L"] * 0.25
    h_w = top15_v["Host_Breadth_H"] * 0.15
    k_w = top15_v["Chem_Confirm_K"] * 0.10
    ax.barh(y11, c_w, color=TEAL, edgecolor="white", lw=0.6, label="Consensus (w=0.30)")
    ax.barh(y11, f_w, left=c_w, color=ORANGE, edgecolor="white", lw=0.6, label="Fidelity (w=0.20)")
    ax.barh(y11, l_w, left=c_w + f_w, color=GREEN, edgecolor="white", lw=0.6, label="Lethality (w=0.25)")
    ax.barh(y11, h_w, left=c_w + f_w + l_w, color=PURPLE, edgecolor="white", lw=0.6, label="Host breadth (w=0.15)")
    ax.barh(y11, k_w, left=c_w + f_w + l_w + h_w, color=RED, edgecolor="white", lw=0.6, label="Chem. confirm (w=0.10)")
    for idx_r, (yv, r) in enumerate(zip(y11, top15_v.itertuples())):
        ax.text(r.VRPI + 0.012, yv, f"#{len(top15_v) - idx_r} (boot {int(r.Boot_Rank_MultiCit_Lo)}–{int(r.Boot_Rank_MultiCit_Hi)})", va="center", fontsize=8.5, color="#4A5568")
    ax.set_yticks(y11)
    ax.set_yticklabels(top15_v["Species"].tolist(), fontsize=9.5, fontstyle="italic")
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Veterinary Risk Priority Index (weighted, 0–1)")
    ax.set_title("VRPI: composite triage ranking of top 15 multi-citation species (n ≥ 2 of 22)\n(stacked criteria; 95% EM-CCT bootstrap rank intervals within multi-citation pool)", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.48, -0.10), ncol=5, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig11_VRPI_Ranking.png"))
    plt.close(fig)

    # --- FIG 12: Criteria Fingerprint Radar Chart ---
    top5_v = master_df[master_df["Citations_n"] >= 2].sort_values("VRPI", ascending=False).head(5)
    labels = ["Consensus", "Fidelity", "Lethality", "Host\nbreadth", "Chem.\nconfirm"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    radar_cols = [TEAL, ORANGE, GREEN, PURPLE, RED]
    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw=dict(polar=True))
    for (_, r), col in zip(top5_v.iterrows(), radar_cols):
        vals = [r["Consensus_C"], r["Fidelity_F"], r["Lethality_L"], r["Host_Breadth_H"], r["Chem_Confirm_K"]]
        vals += vals[:1]
        ax.plot(angles, vals, color=col, lw=2.0, label=r["Species"])
        ax.fill(angles, vals, color=col, alpha=0.09)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color=SLATE)
    ax.set_title("Five-criterion risk fingerprint of the top-5 VRPI species", fontweight="bold", pad=22)
    leg = ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=9)
    for txt in leg.get_texts():
        txt.set_fontstyle("italic")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig12_VRPI_Top5_Radar.png"))
    plt.close(fig)
