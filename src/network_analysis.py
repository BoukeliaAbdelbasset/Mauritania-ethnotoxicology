"""
Ethnobotanical indices, information-theoretic reporting exclusivity,
tripartite motif enrichment, bipartite nestedness (NODF), and risk-collapse simulation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


def compute_fiv_table(citations: pd.DataFrame, family_col: str, n_informants: int) -> pd.DataFrame:
    """Compute informant-based Family Importance Value: FIV_f = (FC_f / N) * 100."""
    rows = []
    for fam, grp in citations.groupby(family_col):
        n_sp = grp["species"].nunique()
        n_cit = len(grp)
        n_uniq = grp["informant_id"].nunique()
        rows.append({
            "Family": fam,
            "Species_Richness": int(n_sp),
            "Total_Citations": int(n_cit),
            "Unique_Informants": int(n_uniq),
            "FIV_pct": round(n_uniq / n_informants * 100.0, 2),
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["FIV_pct", "Species_Richness", "Family"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def compute_symptom_concentration_and_exclusivity(df: pd.DataFrame, n_total_species: int, min_reports: int = 4) -> pd.DataFrame:
    """Compute Shannon entropy H(s), within-sign concentration C_sign(s), and flora-wide exclusivity S_flora(s)."""
    sym_df = df[df["symptom"].notna()].copy()
    rows = []
    for sym, cnt in sym_df["symptom"].value_counts().items():
        if cnt < min_reports:
            continue
        sub = sym_df[sym_df["symptom"] == sym]
        p_vec = sub["species"].value_counts(normalize=True).values
        H = float(-np.sum(p_vec * np.log2(p_vec)))
        if abs(H) < 1e-12:
            H = 0.0
        K_s = len(p_vec)
        s_rep = 1.0 if K_s == 1 else float(1.0 - H / np.log2(K_s))
        s_global = float(1.0 - H / np.log2(n_total_species))
        rows.append({
            "Clinical_Sign": sym,
            "Reports_n": int(cnt),
            "Cited_Taxa_Ks": int(K_s),
            "Entropy_bits": round(H, 4),
            "Within_Sign_Concentration_Csign": round(s_rep, 4),
            "Flora_Wide_Exclusivity_Sflora": round(s_global, 4),
        })
    return pd.DataFrame(rows).sort_values("Within_Sign_Concentration_Csign", ascending=False).reset_index(drop=True)


def compute_toxidrome_icf_and_nmi(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Compute Informant Consensus Factor (ICF) per toxidrome and geometric-mean NMI."""
    sym_df = df[df["symptom"].notna()].copy()
    rows = []
    for tox, grp in sym_df.groupby("toxidrome"):
        nur = len(grp)
        nt = grp["species"].nunique()
        icf = (nur - nt) / (nur - 1) if nur > 1 else np.nan
        rows.append({
            "Toxidrome": tox,
            "Reports_Nur": int(nur),
            "Taxa_Nt": int(nt),
            "ICF": round(float(icf), 4),
        })
    icf_table = pd.DataFrame(rows).sort_values("ICF", ascending=False).reset_index(drop=True)
    nmi_geom = float(normalized_mutual_info_score(sym_df["toxidrome"], sym_df["species"], average_method="geometric"))
    return icf_table, nmi_geom


def compute_tripartite_motifs(df: pd.DataFrame, n_perm: int = 2000, min_obs: int = 5, seed: int = 42) -> pd.DataFrame:
    """Test Organ -> Host-class -> Toxidrome motifs against degree-preserving permutations."""
    tri_cit = (
        df.groupby("citation_id")
        .agg(organ=("organ", "first"), host_class=("host_class", "first"), toxidrome=("toxidrome", "first"))
        .dropna()
    )
    rng = np.random.default_rng(seed)
    obs_m = tri_cit.groupby(["organ", "host_class", "toxidrome"]).size()
    o_arr = tri_cit["organ"].values
    h_arr = tri_cit["host_class"].values
    t_arr = tri_cit["toxidrome"].values

    perm_counts = {k: np.zeros(n_perm) for k in obs_m.index}
    for p in range(n_perm):
        op = rng.permutation(o_arr)
        tp = rng.permutation(t_arr)
        c_dict: dict = {}
        for oo, hh, tt in zip(op, h_arr, tp):
            key = (oo, hh, tt)
            if key in perm_counts:
                c_dict[key] = c_dict.get(key, 0) + 1
        for k, v in c_dict.items():
            perm_counts[k][p] = v

    rows = []
    for k, ov in obs_m.items():
        if ov < min_obs:
            continue
        pm = float(perm_counts[k].mean())
        ps = float(perm_counts[k].std(ddof=1))
        z = (ov - pm) / ps if ps > 0 else 0.0
        rows.append({
            "Motif": f"{k[0]} → {k[1]} → {k[2]}",
            "Observed": int(ov),
            "Null_Mean": round(pm, 2),
            "Enrichment": round(ov / pm, 2) if pm > 0 else np.nan,
            "Z_score": round(z, 2),
        })
    return pd.DataFrame(rows).sort_values("Z_score", ascending=False).reset_index(drop=True)


def compute_nodf(M: np.ndarray, strict_almeida_neto: bool = False) -> float:
    """Compute NODF nestedness metric on a binary plant x host incidence matrix."""
    M = np.asarray(M, dtype=int)
    M = M[np.argsort(-M.sum(axis=1)), :][:, np.argsort(-M.sum(axis=0))]
    nr, nc = M.shape
    rs, cs = M.sum(axis=1), M.sum(axis=0)

    total_ov = 0.0
    valid_pairs = 0
    for i in range(nr):
        for j in range(i + 1, nr):
            if rs[i] > rs[j] and rs[j] > 0:
                total_ov += (np.sum((M[i] == 1) & (M[j] == 1)) / rs[j]) * 100.0
                valid_pairs += 1
    for i in range(nc):
        for j in range(i + 1, nc):
            if cs[i] > cs[j] and cs[j] > 0:
                total_ov += (np.sum((M[:, i] == 1) & (M[:, j] == 1)) / cs[j]) * 100.0
                valid_pairs += 1

    denom = (nr * (nr - 1) / 2.0 + nc * (nc - 1) / 2.0) if strict_almeida_neto else valid_pairs
    return float(total_ov / denom) if denom > 0 else 0.0


def bascompte_null_nodf(
    M: np.ndarray,
    n_perm: int = 1000,
    strict_almeida_neto: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Probabilistic marginal-preserving Null Model 2 (Bascompte et al., 2003)."""
    rng = np.random.default_rng(seed)
    nr, nc = M.shape
    r_prob = M.sum(axis=1) / nc
    c_prob = M.sum(axis=0) / nr
    P = 0.5 * (r_prob[:, None] + c_prob[None, :])
    null_vals = np.zeros(n_perm)
    for p in range(n_perm):
        M_rand = (rng.random((nr, nc)) < P).astype(int)
        null_vals[p] = compute_nodf(M_rand, strict_almeida_neto=strict_almeida_neto)
    return null_vals


def compute_risk_collapse_curve(mat_bin: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate generalist-first plant removal vs. random removal robustness curve."""
    degrees = mat_bin.sum(axis=1).sort_values(ascending=False).values
    total_links = int(degrees.sum())
    rem_frac = (total_links - np.concatenate([[0], np.cumsum(degrees)])) / total_links
    x_frac = np.linspace(0, 1, len(rem_frac))
    auc_targeted = float(np.trapz(rem_frac, x_frac))
    return x_frac, rem_frac, auc_targeted


def compute_tripartite_motifs_expanded(df: pd.DataFrame, n_perm: int = 2000, min_obs: int = 5, seed: int = 42) -> pd.DataFrame:
    """Sensitivity analysis over all unique (citation_id, organ, host_class, toxidrome) tuples."""
    df_ff = df.copy()
    df_ff["organ_ff"] = df_ff.groupby("citation_id")["organ"].ffill()
    df_ff["tox_ff"] = df_ff.groupby("citation_id")["toxidrome"].ffill()
    tri = df_ff.dropna(subset=["organ_ff", "host_class", "tox_ff"])[
        ["citation_id", "organ_ff", "host_class", "tox_ff"]
    ].drop_duplicates()
    rng = np.random.default_rng(seed)
    obs_m = tri.groupby(["organ_ff", "host_class", "tox_ff"]).size()
    o_arr, h_arr, t_arr = tri["organ_ff"].values, tri["host_class"].values, tri["tox_ff"].values
    perm_counts = {k: np.zeros(n_perm) for k in obs_m.index}
    for p in range(n_perm):
        op, tp = rng.permutation(o_arr), rng.permutation(t_arr)
        cd: dict = {}
        for oo, hh, tt in zip(op, h_arr, tp):
            key = (oo, hh, tt)
            if key in perm_counts:
                cd[key] = cd.get(key, 0) + 1
        for k, v in cd.items():
            perm_counts[k][p] = v
    rows = []
    for k, ov in obs_m.items():
        if ov < min_obs:
            continue
        pm, ps = float(perm_counts[k].mean()), float(perm_counts[k].std(ddof=1))
        z = (ov - pm) / ps if ps > 0 else 0.0
        rows.append({
            "Motif": f"{k[0]} → {k[1]} → {k[2]}",
            "Observed": int(ov),
            "Null_Mean": round(pm, 2),
            "Enrichment": round(ov / pm, 2) if pm > 0 else np.nan,
            "Z_score": round(z, 2),
        })
    return pd.DataFrame(rows).sort_values("Z_score", ascending=False).reset_index(drop=True)
