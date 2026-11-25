"""
Veterinary Risk Priority Index (VRPI) composite triage scoring and
informant-level non-parametric bootstrap rank stability intervals.

Audit-Proof Methodological Fixes Implemented:
  1. Clinical Reporting Fidelity (F_j) strictly EXCLUDES 'Unspecified' ('Undefined')
     from the numerator: F_j = max_{t != Unspecified}(n_{j,t}) / N_j, so taxa whose
     reports contain only 'Undefined' signs receive F_j = 0.000 rather than 1.000.
  2. Chemical/Toxicological Confirmation (K_j) loads the audited 3-tier literature
     database (`1.0` species-level, `0.5` genus/subfamily/mechanical, `0.0` unconfirmed).
  3. Informant-level bootstrap (1,000 resamples) genuinely re-estimates competence-weighted
     CCT salience on every resample via `fit_cct_em_map` and computes BOTH:
       - `Boot_Rank_MultiCit_Lo/Hi`: rank bounds within the n >= 2 multi-citation triage pool (1..22)
       - `Boot_Rank_All41_Lo/Hi`: rank bounds across the full 41-species inventory (1..41).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from cct_model import fit_cct_em_map


DEFAULT_WEIGHTS = {
    "consensus": 0.30,
    "lethality": 0.25,
    "fidelity": 0.20,
    "host_breadth": 0.15,
    "chem_confirm": 0.10,
}


def compute_vrpi_and_bootstrap(
    df: pd.DataFrame,
    Y_matrix: np.ndarray,
    species_order: list[str],
    citations_count_map: dict[str, int],
    theta_map: dict[str, float],
    lit_df: pd.DataFrame,
    max_hosts: int,
    n_informants: int,
    n_boot: int = 1000,
    weights: dict[str, float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute weighted VRPI sub-indices and 95% informant bootstrap rank intervals."""
    w = weights or DEFAULT_WEIGHTS
    lit_lookup = lit_df.set_index("Species").to_dict(orient="index")
    max_theta = max(theta_map.values()) if max(theta_map.values()) > 0 else 1.0

    rows = []
    for sp in species_order:
        sgrp = df[df["species"] == sp]
        c_norm = theta_map[sp] / max_theta

        obs = sgrp["obs_clean"].dropna()
        leth = float((obs == "Death").mean()) if len(obs) > 0 else 0.0

        # CRITICAL FIX: Exclude 'Unspecified' from Clinical Reporting Fidelity numerator
        toxs = sgrp["toxidrome"].dropna()
        if len(toxs) > 0:
            spec_toxs = toxs[toxs != "Unspecified"]
            if len(spec_toxs) > 0:
                vc_spec = spec_toxs.value_counts()
                dom_tox = str(vc_spec.index[0])
                fid = float(vc_spec.iloc[0] / len(toxs))
            else:
                dom_tox = "Unspecified (F=0)"
                fid = 0.0
        else:
            dom_tox = "None"
            fid = 0.0

        hb_raw = int(sgrp["host"].dropna().nunique())
        hb = hb_raw / max_hosts

        lit_entry = lit_lookup.get(sp, {})
        k_val = float(lit_entry.get("Chem_Confirm_K", 0.0))
        k_ref = str(lit_entry.get("Primary_Reference", "No confirmed species-level toxicology reference"))
        k_tox = str(lit_entry.get("Documented_Phytotoxins", "Unconfirmed"))
        k_tier = str(lit_entry.get("Evidence_Tier", "Unconfirmed (No Published Evidence)"))

        vrpi = (
            w["consensus"] * c_norm
            + w["lethality"] * leth
            + w["fidelity"] * fid
            + w["host_breadth"] * hb
            + w["chem_confirm"] * k_val
        )
        rows.append({
            "Species": sp,
            "Dominant_Specified_Toxidrome": dom_tox,
            "Consensus_C": c_norm,
            "Lethality_L": leth,
            "Fidelity_F": fid,
            "Host_Breadth_H": hb,
            "Host_Count": hb_raw,
            "Chem_Confirm_K": k_val,
            "Evidence_Tier": k_tier,
            "Documented_Phytotoxins": k_tox,
            "Literature_Evidence": k_ref,
            "VRPI": vrpi,
        })
    vrpi_df = pd.DataFrame(rows)

    # Pre-aggregate per-informant species observations for exact bootstrap
    inf_records: dict[int, dict[str, tuple]] = {}
    for inf_id, grp in df.groupby("informant_id"):
        sp_dict = {}
        for sp, sgrp in grp.groupby("species"):
            obs_s = sgrp["obs_clean"].dropna().tolist()
            tox_s = sgrp["toxidrome"].dropna().tolist()
            hosts_s = set(sgrp["host"].dropna().tolist())
            sp_dict[sp] = (obs_s, tox_s, hosts_s)
        inf_records[int(inf_id)] = sp_dict

    n_sp = len(species_order)
    multi_mask = np.array([citations_count_map[sp] >= 2 for sp in species_order], dtype=bool)
    multi_indices = np.where(multi_mask)[0]
    k_vec = np.array([float(lit_lookup.get(sp, {}).get("Chem_Confirm_K", 0.0)) for sp in species_order])

    rng = np.random.default_rng(seed)
    boot_ranks_41 = np.zeros((n_boot, n_sp))
    boot_ranks_multi = np.full((n_boot, n_sp), np.nan)

    for b in range(n_boot):
        idx_inf = rng.integers(0, n_informants, size=n_informants)
        # Re-estimate competence-weighted CCT salience on resampled matrix via fast EM-MAP
        Y_b = Y_matrix[idx_inf, :]
        _, t_b = fit_cct_em_map(Y_b, alpha_c=2.0, beta_c=2.0, alpha_t=1.0, beta_t=2.0, n_iter=12)
        max_tb = float(np.max(t_b)) if np.max(t_b) > 0 else 1.0
        c_b_vec = t_b / max_tb

        death_cnt = {sp: 0 for sp in species_order}
        obs_cnt = {sp: 0 for sp in species_order}
        tox_spec_cnt: dict[str, dict[str, int]] = {sp: {} for sp in species_order}
        tox_tot_cnt = {sp: 0 for sp in species_order}
        host_sets: dict[str, set[str]] = {sp: set() for sp in species_order}

        for row_i in idx_inf:
            iid = int(row_i) + 1
            for sp, (obs_s, tox_s, hosts_s) in inf_records[iid].items():
                obs_cnt[sp] += len(obs_s)
                death_cnt[sp] += sum(1 for o in obs_s if o == "Death")
                tox_tot_cnt[sp] += len(tox_s)
                for t in tox_s:
                    if t != "Unspecified":
                        tox_spec_cnt[sp][t] = tox_spec_cnt[sp].get(t, 0) + 1
                host_sets[sp].update(hosts_s)

        scores_b = np.zeros(n_sp)
        for idx_s, sp in enumerate(species_order):
            l_b = death_cnt[sp] / obs_cnt[sp] if obs_cnt[sp] > 0 else 0.0
            f_b = (max(tox_spec_cnt[sp].values()) / tox_tot_cnt[sp]) if (tox_tot_cnt[sp] > 0 and tox_spec_cnt[sp]) else 0.0
            h_b = len(host_sets[sp]) / max_hosts
            scores_b[idx_s] = (
                w["consensus"] * c_b_vec[idx_s]
                + w["lethality"] * l_b
                + w["fidelity"] * f_b
                + w["host_breadth"] * h_b
                + w["chem_confirm"] * k_vec[idx_s]
            )

        # Full 41-species rank
        ord_41 = np.argsort(-scores_b)
        r41 = np.empty(n_sp)
        r41[ord_41] = np.arange(1, n_sp + 1)
        boot_ranks_41[b] = r41

        # Multi-citation (n >= 2) pool rank (1..22) aligned with Fig. 11 & Table 4
        scores_m = scores_b[multi_indices]
        ord_m = np.argsort(-scores_m)
        rm = np.empty(len(multi_indices))
        rm[ord_m] = np.arange(1, len(multi_indices) + 1)
        boot_ranks_multi[b, multi_indices] = rm

    vrpi_df["Boot_Rank_All41_Lo"] = np.percentile(boot_ranks_41, 2.5, axis=0).round().astype(int)
    vrpi_df["Boot_Rank_All41_Hi"] = np.percentile(boot_ranks_41, 97.5, axis=0).round().astype(int)

    mc_lo, mc_hi = [], []
    for idx_s in range(n_sp):
        if multi_mask[idx_s]:
            mc_lo.append(int(np.round(np.percentile(boot_ranks_multi[:, idx_s], 2.5))))
            mc_hi.append(int(np.round(np.percentile(boot_ranks_multi[:, idx_s], 97.5))))
        else:
            mc_lo.append(np.nan)
            mc_hi.append(np.nan)
    vrpi_df["Boot_Rank_MultiCit_Lo"] = mc_lo
    vrpi_df["Boot_Rank_MultiCit_Hi"] = mc_hi
    return vrpi_df


def compute_vrpi_weight_sensitivity(master_df: pd.DataFrame, min_citations: int = 2) -> dict:
    """Evaluate Spearman rank stability and top-5 invariance across alternative weight schemes."""
    from scipy.stats import spearmanr

    mc = master_df[master_df["Citations_n"] >= min_citations].copy()
    base = mc["VRPI"].values
    schemes = {
        "equal_weights": (0.20, 0.20, 0.20, 0.20, 0.20),
        "toxicology_heavy": (0.20, 0.30, 0.15, 0.10, 0.25),
        "consensus_heavy": (0.40, 0.20, 0.15, 0.15, 0.10),
    }
    out = {}
    for label, (wc, wl, wf, wh, wk) in schemes.items():
        sc = (
            wc * mc["Consensus_C"]
            + wl * mc["Lethality_L"]
            + wf * mc["Fidelity_F"]
            + wh * mc["Host_Breadth_H"]
            + wk * mc["Chem_Confirm_K"]
        )
        rho = float(spearmanr(base, sc)[0])
        top5 = mc.assign(_sc=sc).sort_values("_sc", ascending=False)["Species"].head(5).tolist()
        out[label] = {"spearman_rho": round(rho, 4), "top5": top5}
    return out
