#!/usr/bin/env python3
"""
attractor_inference_pc9.py
==========================
Complete analysis code for

  Raza W, Mehmood S, Enderlein C, Barbosa Da Silva FA.
  "Replicate libraries set the noise floor for cell-state claims in cross-sectional
   single-cell data: an empirical audit in drug-tolerant EGFR-mutant PC9 cells."

Everything is in this one file. Sub-commands (run `python attractor_inference_pc9.py -h`):

  preprocess-gse150949  raw GSE150949 counts + metadata  ->  GSE150949_PCA50.csv.gz
  gse150949             noise floor, contrasts, replicate-referenced ratios, dip test,
                        correlation dimension, trajectory distance, lineage concordance
  library-distance      all-library energy-distance matrices and PC sensitivity (10/20/50 PCs)
  gse149383             Drop-seq erlotinib pipeline: merge, QC, normalise, PCA, classifier
                        controls, dip/GMM, equal-n correlation dimension, CCM, permutation
                        entropy, transfer entropy, surrogates, random-gene control
  simulate              toggle-switch simulation benchmark (library effects, replicate ratio,
                        dip test, correlation dimension, interpolated permutation entropy)
  selftest              runs every stage on small synthetic data (no downloads needed)

Requirements: Python >= 3.10, numpy, scipy, pandas, scikit-learn, diptest.
Optional: scanpy (diffusion pseudotime only), scrublet (doublet control only).

Notes on faithfulness
---------------------
* Written from the Methods and Additional file 1 of the manuscript. Where a detail is not
  stated there, the choice made here is marked with "ASSUMPTION" in a comment.
* The dip test on GSE150949 uses a PCA fitted *within* each library. If only the archived 50
  global PCs are available, that PCA is fitted on those 50 PCs (an approximation of the
  within-library PCA on the 2,000-gene matrix).
* CCM, permutation entropy, transfer entropy and the surrogate tests are implemented here
  (the skccm package is not required).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

__version__ = "1.0.0"

MARKERS = ["EGFR", "TACSTD2", "SERPINE1", "CD24", "MET", "ALDH1A3", "SOX4", "GPRC5A", "CYP1B1"]

# =====================================================================================
# 1. Generic statistics
# =====================================================================================


def energy_distance(A, B):
    """2E|A-B| - E|A-A'| - E|B-B'| with unbiased within-sample terms (expectation 0 under H0)."""
    na, nb = len(A), len(B)
    return float(2 * cdist(A, B).mean()
                 - cdist(A, A).sum() / (na * (na - 1))
                 - cdist(B, B).sum() / (nb * (nb - 1)))


def make_clf():
    return LogisticRegression(C=1.0, max_iter=2000)  # default penalty is l2


def cv_predict(X, y, seed=0, folds=5):
    """Out-of-fold probabilities of an l2 logistic regression, 5-fold stratified CV."""
    p = np.zeros(len(y))
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        p[te] = make_clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def cv_accuracy(X, y, seed=0):
    p = cv_predict(X, y, seed)
    return float(((p > 0.5) == y).mean()), p


def cv_acc_auc(X, y, seed=0):
    acc, p = cv_accuracy(X, y, seed)
    return acc, float(roc_auc_score(y, p))


def perm_pvalue(X, y, observed_acc, n_perm=200, seed=0):
    """Label-permutation p = (r+1)/(M+1); the smallest attainable value is 1/(M+1)."""
    rng = np.random.default_rng(seed)
    r = 0
    for _ in range(n_perm):
        acc, _ = cv_accuracy(X, rng.permutation(y), seed=int(rng.integers(1 << 30)))
        r += acc >= observed_acc
    return (r + 1) / (n_perm + 1)


def heldout_accuracy(Xtr, ytr, Xte, yte):
    return float((make_clf().fit(Xtr, ytr).predict(Xte) == yte).mean())


def two_group(A, B):
    return np.vstack([A, B]), np.r_[np.zeros(len(A), int), np.ones(len(B), int)]


def rho_accuracy(acc_contrast, acc_replicate):
    """Replicate-referenced accuracy ratio (acc_contrast - 0.5) / (acc_replicate - 0.5)."""
    return (acc_contrast - 0.5) / (acc_replicate - 0.5)


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def dip_pvalue(x):
    try:
        import diptest
    except ImportError as e:  # pragma: no cover
        raise SystemExit("The 'diptest' package is required: pip install diptest") from e
    dip, p = diptest.diptest(np.asarray(x, float))
    return float(dip), float(p)


def within_pc1(X):
    """PC1 of a PCA fitted within the given cells."""
    Xc = X - X.mean(0)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[0]


def q025_975(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ---- correlation dimension --------------------------------------------------------


def radius_grid(X_ref, n_points=40, lo_pct=5, hi_pct=95, max_cells=2000, seed=0):
    """Logarithmic radius grid between the 5th and 95th percentile of pairwise distances."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_ref), min(max_cells, len(X_ref)), replace=False)
    d = pdist(X_ref[idx])
    d = d[d > 0]
    return np.geomspace(np.percentile(d, lo_pct), np.percentile(d, hi_pct), n_points)


def correlation_dimension(X, radii, window=(0.15, 0.60), return_curve=False):
    """D2 = slope of ln C(r) vs ln r over 15-60% of the log radius grid.
    C(r) = fraction of distinct cell pairs (i != j) closer than r."""
    d = np.sort(pdist(X))
    C = np.searchsorted(d, radii, side="right") / len(d)
    k = len(radii) - 1
    lo, hi = int(round(window[0] * k)), int(round(window[1] * k))
    sel = np.zeros(len(radii), bool)
    sel[lo:hi + 1] = True
    sel &= C > 0
    if sel.sum() < 3:
        return (np.nan, C) if return_curve else np.nan
    slope = np.polyfit(np.log(radii[sel]), np.log(C[sel]), 1)[0]
    return (float(slope), C) if return_curve else float(slope)


# =====================================================================================
# 2. Normalisation / feature selection / PCA (shared by both datasets)
# =====================================================================================


def log_normalise(X, scale=1e4):
    """x' = ln(1 + 1e4 * x / sum_g x)   (Eq. 1).  X: sparse cells x genes."""
    X = sparse.csr_matrix(X, dtype=np.float64)
    tot = np.asarray(X.sum(1)).ravel()
    tot[tot == 0] = 1
    Xn = sparse.diags(scale / tot) @ X
    Xn = sparse.csr_matrix(Xn)
    Xn.data = np.log1p(Xn.data)
    return Xn


def dispersion_z(Xlog, n_bins=20, mean_range=None):
    """Normalised dispersion (z-score of var/mean within expression-level bins)."""
    mean = np.asarray(Xlog.mean(0)).ravel()
    sq = np.asarray(Xlog.multiply(Xlog).mean(0)).ravel()
    var = np.maximum(sq - mean ** 2, 0)
    ok = mean > 0
    if mean_range is not None:
        ok &= (mean >= mean_range[0]) & (mean <= mean_range[1])
    disp = np.full(len(mean), np.nan)
    disp[ok] = var[ok] / mean[ok]
    z = np.full(len(mean), -np.inf)
    if ok.sum() == 0:
        return z, mean
    bins = pd.cut(mean[ok], n_bins, labels=False)
    zz = np.zeros(ok.sum())
    d = disp[ok]
    for b in np.unique(bins):
        m = bins == b
        sd = d[m].std()
        zz[m] = (d[m] - d[m].mean()) / sd if (m.sum() > 1 and sd > 0) else 0.0
    z[ok] = zz
    return z, mean


def select_hvg(Xlog, n_top, exclude=None, **kw):
    z, _ = dispersion_z(Xlog, **kw)
    if exclude is not None and len(exclude):
        z[np.asarray(exclude)] = -np.inf
    return np.argsort(-z)[:n_top]


def regress_out(X, covariates):
    """Per-gene OLS on [1, covariates]; return residuals (equivalent to scanpy regress_out)."""
    D = np.column_stack([np.ones(len(X))] + [np.asarray(c, float) for c in covariates])
    B, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ B


def standardise(X):
    sd = X.std(0)
    sd[sd == 0] = 1
    return (X - X.mean(0)) / sd


def pca_scores(X, n_comp=50, seed=0):
    n_comp = min(n_comp, min(X.shape) - 1)
    return PCA(n_components=n_comp, svd_solver="randomized", random_state=seed).fit_transform(X)


# =====================================================================================
# 3. GSE150949 (10x, replicate libraries)
# =====================================================================================

LIBS_150949 = ["0", "3_rep1", "3_rep2", "7_rep1", "7_rep2",
               "14_rep1_high", "14_rep2_high", "14_rep1_med", "14_rep2_med",
               "14_rep1_low", "14_rep2_low"]
ARMS = {"D0": ["0"], "D3": ["3_rep1", "3_rep2"], "D7": ["7_rep1", "7_rep2"],
        "D14high": ["14_rep1_high", "14_rep2_high"], "D14med": ["14_rep1_med", "14_rep2_med"],
        "D14low": ["14_rep1_low", "14_rep2_low"]}
CONTRASTS = [("D0", "D3"), ("D3", "D7"), ("D7", "D14med"), ("D3", "D14med"),
             ("D14high", "D14med"), ("D14med", "D14low"), ("D14high", "D14low")]


def _open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def preprocess_gse150949(counts_path, meta_path, out_path, sep=None, chunk_genes=250,
                         n_hvg=2000, n_pcs=50, lib_col="sample_name", seed=0):
    """GEO counts (genes x cells, csv/tsv[.gz], first column = gene) + metadata (index = cell id)
    -> 50 PCs per cell.  Three streamed passes of 250 genes keep memory low.
    Steps: scale to 1e4/cell, log, 2000 most variable genes (dispersion normalised in 20
    expression bins; genes with mean log expression 0.0125-3), regress log UMI + mito fraction,
    standardise, randomised-SVD PCA."""
    if sep is None:
        sep = "\t" if ".tsv" in str(counts_path) or ".txt" in str(counts_path) else ","
    meta = pd.read_csv(meta_path, index_col=0, sep=None, engine="python")

    def chunks():
        return pd.read_csv(counts_path, sep=sep, index_col=0, chunksize=chunk_genes)

    # pass 1: per-cell totals and mitochondrial totals
    tot = None
    mito = None
    cells = None
    for ch in chunks():
        if cells is None:
            cells = ch.columns.astype(str)
            tot = np.zeros(len(cells))
            mito = np.zeros(len(cells))
        v = ch.to_numpy(float)
        tot += v.sum(0)
        m = ch.index.astype(str).str.upper().str.startswith("MT-")
        mito += v[m].sum(0)
    meta = meta.loc[cells] if set(cells) <= set(meta.index) else meta.reindex(cells)
    mito_frac = mito / np.maximum(tot, 1)
    logdepth = np.log(np.maximum(tot, 1))
    # pass 2: gene mean / second moment of log-normalised data
    names, means, sqs = [], [], []
    for ch in chunks():
        v = np.log1p(1e4 * ch.to_numpy(float) / np.maximum(tot, 1))
        names += list(ch.index.astype(str))
        means.append(v.mean(1))
        sqs.append((v ** 2).mean(1))
    mean = np.concatenate(means)
    var = np.maximum(np.concatenate(sqs) - mean ** 2, 0)
    ok = (mean >= 0.0125) & (mean <= 3)
    disp = np.where(ok, var / np.where(mean > 0, mean, 1), np.nan)
    bins = pd.cut(mean[ok], 20, labels=False)
    z = np.full(len(mean), -np.inf)
    zz = np.zeros(ok.sum())
    d = disp[ok]
    for b in np.unique(bins):
        mk = bins == b
        sd = d[mk].std()
        zz[mk] = (d[mk] - d[mk].mean()) / sd if (mk.sum() > 1 and sd > 0) else 0
    z[ok] = zz
    keep = np.sort(np.argsort(-z)[:n_hvg])
    keep_set = set(keep.tolist())
    # pass 3: collect selected genes
    X = np.zeros((len(cells), len(keep)), np.float32)
    row = 0
    for ch in chunks():
        v = np.log1p(1e4 * ch.to_numpy(float) / np.maximum(tot, 1))
        for i in range(v.shape[0]):
            if row + i in keep_set:
                X[:, np.searchsorted(keep, row + i)] = v[i]
        row += v.shape[0]
    R = regress_out(X.astype(np.float64), [logdepth, mito_frac])
    pcs = pca_scores(standardise(R), n_pcs, seed)
    out = meta.copy()
    out["log_depth"] = logdepth
    out["mito_frac"] = mito_frac
    for i in range(pcs.shape[1]):
        out[f"PC{i + 1}"] = pcs[:, i]
    out.to_csv(out_path)
    print(f"PC1 vs log depth r = {np.corrcoef(pcs[:, 0], logdepth)[0, 1]:.3f}; "
          f"wrote {out_path} ({len(out)} cells)")
    return out


def load_pca(path, lib_col="sample_name"):
    d = pd.read_csv(path, index_col=0, low_memory=False)
    pcs = [c for c in d.columns if c.startswith("PC") and c[2:].isdigit()]
    pcs = sorted(pcs, key=lambda c: int(c[2:]))
    lib = d[lib_col].astype(str).to_numpy()
    return d, d[pcs].to_numpy(float), lib


def analyse_gse150949(pca_path, outdir, lib_col="sample_name", barcode_col="lineage_barcode",
                      n_cells=1300, n_draws=20, n_pcs=20, seed=20240101,
                      d2_cells=2000, d2_draws=8, n_shuffles=200):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta, P, lib = load_pca(pca_path, lib_col)
    libs = [l for l in LIBS_150949 if l in set(lib)] or sorted(set(lib))
    idx_of = {l: np.where(lib == l)[0] for l in libs}
    rng = np.random.default_rng(seed)
    Xk = P[:, :n_pcs]

    # ---------- noise floor and contrasts, 20 random draws ----------
    floor_rows, contrast_rows, dir_rows = [], [], []
    for draw in range(n_draws):
        S = {l: Xk[rng.choice(idx_of[l], min(n_cells, len(idx_of[l])), replace=False)] for l in libs}
        # split-half floor: two halves of 2,600 cells of one library
        for l in libs:
            big = Xk[rng.choice(idx_of[l], min(2 * n_cells, len(idx_of[l])), replace=False)]
            h = len(big) // 2
            A, B = big[:h], big[h:2 * h]
            Xg, yg = two_group(A, B)
            acc, _ = cv_accuracy(Xg, yg, seed=draw)
            floor_rows.append(dict(draw=draw, kind="split_half", name=l, accuracy=acc,
                                   energy_distance=energy_distance(A, B)))
        # replicate libraries of one condition
        rep_acc, rep_ed = {}, {}
        for a, ls in ARMS.items():
            if len(ls) == 2 and all(x in S for x in ls):
                Xg, yg = two_group(S[ls[0]], S[ls[1]])
                rep_acc[a], _ = cv_accuracy(Xg, yg, seed=draw)
                rep_ed[a] = energy_distance(S[ls[0]], S[ls[1]])
                floor_rows.append(dict(draw=draw, kind="replicate", name=a, accuracy=rep_acc[a],
                                       energy_distance=rep_ed[a]))
                # direction of the replicate difference
                dvec = S[ls[1]].mean(0) - S[ls[0]].mean(0)
                u = dvec / np.linalg.norm(dvec)
                acc_dir, _ = cv_accuracy(np.vstack([S[ls[0]], S[ls[1]]]) @ u[:, None], yg, seed=draw)
                dir_rows.append(dict(draw=draw, condition=a, acc_20pc=rep_acc[a], acc_direction=acc_dir,
                                     share_of_above_chance=(acc_dir - .5) / (rep_acc[a] - .5),
                                     share_sq_shift_PC1_3=float((dvec[:3] ** 2).sum() / (dvec ** 2).sum())))
        # Day 0 halves (one library only)
        d0 = Xk[rng.choice(idx_of[ARMS["D0"][0]], min(2 * n_cells, len(idx_of[ARMS["D0"][0]])), replace=False)]
        h0 = len(d0) // 2
        S0a, S0b = d0[:h0], d0[h0:2 * h0]
        S["D0_half1"], S["D0_half2"] = S0a, S0b

        def arm_libs(a):
            return [S0a, S0b] if a == "D0" else [S[l] for l in ARMS[a]]

        def arm_lib_pair(a):  # distinct sampled libraries for ED (Day 0: the sampled library)
            return [S[ARMS[a][0]]] if a == "D0" else [S[l] for l in ARMS[a]]

        for a, b in CONTRASTS:
            A1 = arm_lib_pair(a)[0] if a != "D0" else S[ARMS["D0"][0]]
            B1 = arm_lib_pair(b)[0]
            Xg, yg = two_group(A1, B1)
            acc_single, _ = cv_accuracy(Xg, yg, seed=draw)
            # held-out replicate (train on replicate 1 of both arms, test on replicate 2, and reverse)
            la, lb = arm_libs(a), arm_libs(b)
            Xtr, ytr = two_group(la[0], lb[0])
            Xte, yte = two_group(la[1], lb[1])
            acc_held = 0.5 * (heldout_accuracy(Xtr, ytr, Xte, yte) + heldout_accuracy(Xte, yte, Xtr, ytr))
            reps = [x for x in (a, b) if x in rep_acc]
            acc_rep = float(np.mean([rep_acc[x] for x in reps]))
            ed_rep = float(np.mean([rep_ed[x] for x in reps]))
            ed_between = float(np.mean([energy_distance(x, y) for x in arm_lib_pair(a) for y in arm_lib_pair(b)]))
            contrast_rows.append(dict(
                draw=draw, contrast=f"{a} vs {b}", acc_single=acc_single, acc_replicate=acc_rep,
                acc_heldout=acc_held, rho_accuracy=rho_accuracy(acc_single, acc_rep),
                ed_between=ed_between, ed_replicate=ed_rep, rho_ED=ed_between / ed_rep))
    floor = pd.DataFrame(floor_rows)
    contr = pd.DataFrame(contrast_rows)
    dirs = pd.DataFrame(dir_rows)
    floor.to_csv(outdir / "S8_noise_floor_draws.csv", index=False)
    contr.to_csv(outdir / "S9_contrasts_draws.csv", index=False)
    dirs.to_csv(outdir / "S9_replicate_direction_draws.csv", index=False)
    contr.groupby("contrast", sort=False).agg(["mean"]).droplevel(1, axis=1).drop(columns="draw") \
        .assign(rho_ED_lo=contr.groupby("contrast", sort=False)["rho_ED"].quantile(.025),
                rho_ED_hi=contr.groupby("contrast", sort=False)["rho_ED"].quantile(.975)) \
        .to_csv(outdir / "Table2_contrasts.csv")
    floor.groupby(["kind", "name"], sort=False)[["accuracy", "energy_distance"]].agg(["mean", "min", "max"]) \
        .to_csv(outdir / "Table1_noise_floor.csv")

    # ---------- dip test within each library and in pooled replicate pairs ----------
    dip_rows = []
    for l in libs:
        dip, p = dip_pvalue(within_pc1(P[idx_of[l]]))
        dip_rows.append(dict(unit=l, kind="library", dip=dip, p=p))
    for a, ls in ARMS.items():
        if len(ls) == 2 and all(x in idx_of for x in ls):
            dip, p = dip_pvalue(within_pc1(P[np.r_[idx_of[ls[0]], idx_of[ls[1]]]]))
            dip_rows.append(dict(unit=a, kind="pooled_replicates", dip=dip, p=p))
    pd.DataFrame(dip_rows).to_csv(outdir / "S10_dip_test.csv", index=False)

    # ---------- correlation dimension (2,000 cells/library, 8 draws, shared radius grid) ----------
    d2_rows = []
    for npc in (10, 15, 20, 30):
        ref = P[rng.choice(len(P), min(2000, len(P)), replace=False), :npc]
        radii = radius_grid(ref, seed=seed)
        for l in libs:
            vals = [correlation_dimension(
                P[rng.choice(idx_of[l], min(d2_cells, len(idx_of[l])), replace=False), :npc], radii)
                for _ in range(d2_draws)]
            d2_rows.append(dict(library=l, n_pcs=npc, D2=float(np.nanmean(vals))))
    pd.DataFrame(d2_rows).to_csv(outdir / "S10_correlation_dimension.csv", index=False)

    # ---------- trajectory distance from the Day 0 centroid (50 PCs) ----------
    if ARMS["D0"][0] in idx_of:
        npc50 = min(50, P.shape[1])
        c0 = P[idx_of[ARMS["D0"][0]], :npc50].mean(0)
        td = [dict(library=l, mean_distance_from_D0_centroid=float(
            np.linalg.norm(P[idx_of[l], :npc50] - c0, axis=1).mean()),
            centroid_distance_from_D0=float(np.linalg.norm(P[idx_of[l], :npc50].mean(0) - c0)))
            for l in libs]
        pd.DataFrame(td).to_csv(outdir / "S10_trajectory_distance.csv", index=False)

    # ---------- within-library lineage concordance (20 PCs, 200 shuffles) ----------
    if barcode_col in meta.columns:
        lin_rows = []
        for l in libs:
            ii = idx_of[l]
            bc = meta[barcode_col].to_numpy()[ii]
            ok = pd.notna(bc)
            ii, bc = ii[ok], bc[ok].astype(str)
            vc = pd.Series(bc).value_counts()
            shared = np.isin(bc, vc[vc >= 2].index)
            ii, bc = ii[shared], bc[shared]
            if len(ii) < 4:
                continue
            X = Xk[ii]
            obs = mean_sq_dist_within_groups(X, bc)
            nulls = np.array([mean_sq_dist_within_groups(X, rng.permutation(bc)) for _ in range(n_shuffles)])
            lin_rows.append(dict(library=l, n_cells=len(ii), n_clones=len(set(bc)),
                                 ratio_obs_over_shuffled=obs / nulls.mean(),
                                 p=(int((nulls <= obs).sum()) + 1) / (n_shuffles + 1)))
        pd.DataFrame(lin_rows).to_csv(outdir / "S10_lineage_concordance.csv", index=False)
    else:
        warnings.warn(f"column '{barcode_col}' not found: lineage concordance skipped")
    library_distance_matrix(pca_path, outdir, lib_col=lib_col)
    print("GSE150949 analysis written to", outdir)


def mean_sq_dist_within_groups(X, labels):
    """Mean squared Euclidean distance over all pairs of cells sharing a label.
    Uses sum_{i<j}|xi-xj|^2 = n * sum_i |xi - mean|^2 within each group."""
    total, pairs = 0.0, 0
    for g in pd.unique(labels):
        Xg = X[labels == g]
        n = len(Xg)
        if n < 2:
            continue
        total += n * ((Xg - Xg.mean(0)) ** 2).sum()
        pairs += n * (n - 1) // 2
    return total / pairs


def library_distance_matrix(pca_path, outdir, lib_col="sample_name", n=800, ndraw=5, pcs=(10, 20, 50)):
    """All-library energy-distance matrices (diagonal = split-half) and the PC sensitivity of the
    contrast / replicate distance ratio.  800 cells per library, 5 draws, seed 1."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(pca_path, index_col=0, low_memory=False)
    lib = d[lib_col].astype(str)
    pcn = [f"PC{i}" for i in range(1, 51) if f"PC{i}" in d.columns]
    rng = np.random.default_rng(1)
    libs = sorted(lib.unique())
    rows = []
    for npc in pcs:
        npc = min(npc, len(pcn))
        S = {l: [d.loc[lib == l, pcn[:npc]].values[rng.choice((lib == l).sum(), n, replace=False)]
                 for _ in range(ndraw)] for l in libs}
        M = pd.DataFrame(index=libs, columns=libs, dtype=float)
        for i, a in enumerate(libs):
            for b in libs[i:]:
                v = [energy_distance(S[a][k][:n // 2], S[a][k][n // 2:]) if a == b
                     else energy_distance(S[a][k], S[b][k]) for k in range(ndraw)]
                M.loc[a, b] = M.loc[b, a] = np.mean(v)
        M.to_csv(outdir / f"libdist_{npc}pc.csv")
        rep = {g: M.loc[ARMS[g][0], ARMS[g][1]] for g in ARMS
               if len(ARMS[g]) == 2 and all(x in M.index for x in ARMS[g])}
        for a, b in CONTRASTS:
            if not all(x in M.index for x in ARMS[a] + ARMS[b]):
                continue
            pairs = [M.loc[x, y] for x in ARMS[a] for y in ARMS[b]]
            reps = [rep[g] for g in (a, b) if g in rep]
            rows.append(dict(npc=npc, contrast=f"{a} vs {b}", between=np.mean(pairs),
                             replicate=np.mean(reps), ratio=np.mean(pairs) / np.mean(reps),
                             min_pair=min(pairs), min_pair_over_max_rep=min(pairs) / max(reps)))
    pd.DataFrame(rows).to_csv(outdir / "sensitivity_PCs_ratio.csv", index=False)


# =====================================================================================
# 4. GSE149383 (Drop-seq, one library per condition)
# =====================================================================================

DAYS = {"Day0": 0, "Day1": 1, "Day2": 2, "Day4": 4, "Day9": 9, "Day11": 11}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def read_dge(path, sep="\t"):
    """Per-sample digital gene expression file (genes x cells, first column = gene symbol).
    Returns sparse cells x genes, gene names, cell barcodes."""
    df = pd.read_csv(path, sep=sep, index_col=0)
    return sparse.csr_matrix(df.to_numpy(np.float64).T), list(df.index.astype(str)), list(df.columns.astype(str))


def merge_samples(samples, dge_dir, sep="\t"):
    """samples: DataFrame with columns file, condition, library (+ optional day).
    Duplicate re-uploads (identical MD5, or identical cell-barcode set) are excluded.
    Merge on the union of observed gene symbols (implicit zeros)."""
    seen_md5, seen_bc, keep, log = {}, {}, [], []
    mats = []
    for _, r in samples.iterrows():
        path = Path(dge_dir) / r["file"]
        h = md5(path)
        M, genes, bcs = read_dge(path, sep)
        key = frozenset(bcs)
        if h in seen_md5 or key in seen_bc:
            log.append(f"EXCLUDED duplicate re-upload: {r['file']}")
            continue
        seen_md5[h] = r["file"]
        seen_bc[key] = r["file"]
        mats.append((r, M, genes, bcs))
    union = sorted(set(itertools.chain.from_iterable(g for _, _, g, _ in mats)))
    gi = {g: i for i, g in enumerate(union)}
    blocks, obs = [], []
    for r, M, genes, bcs in mats:
        cols = np.array([gi[g] for g in genes])
        B = sparse.csr_matrix((M.data, M.indices, M.indptr), shape=M.shape).tocoo()
        B = sparse.csr_matrix((B.data, (B.row, cols[B.col])), shape=(M.shape[0], len(union)))
        blocks.append(B)
        obs += [dict(cell=f"{r['library']}:{b}", condition=r["condition"], library=r["library"]) for b in bcs]
    return sparse.vstack(blocks).tocsr(), np.array(union), pd.DataFrame(obs).set_index("cell"), log


def qc_filter(X, genes, obs, min_counts=300, min_genes=30, max_mito=0.30):
    tot = np.asarray(X.sum(1)).ravel()
    ngenes = np.asarray((X > 0).sum(1)).ravel()
    mt = np.array([g.upper().startswith("MT-") for g in genes])
    mito = np.asarray(X[:, np.where(mt)[0]].sum(1)).ravel() / np.maximum(tot, 1) if mt.any() else np.zeros(len(tot))
    keep = (tot > 0) & (tot >= min_counts) & (ngenes >= min_genes) & (mito < max_mito)
    obs = obs.copy()
    obs["total_counts"], obs["n_genes"], obs["mito_frac"] = tot, ngenes, mito
    return X[keep], obs[keep]


def process_counts(X, genes, obs, n_hvg=2000, n_pcs=50, markers=MARKERS, include_markers=True, seed=0):
    """Normalise, select genes, regress out log depth and mito fraction, standardise, PCA.
    include_markers=True : top n_hvg non-marker genes + force-included markers (p = 2,009).
    include_markers=False: top n_hvg genes with the markers excluded (marker-free axis)."""
    Xl = log_normalise(X)
    gl = {g: i for i, g in enumerate(genes)}
    mk = [gl[m] for m in markers if m in gl]
    sel = select_hvg(Xl, n_hvg, exclude=mk)
    if include_markers:
        sel = np.unique(np.r_[sel, mk])
    D = np.asarray(Xl[:, sel].todense())
    logd = np.log(np.maximum(obs["total_counts"].to_numpy(), 1))
    R = regress_out(D, [logd, obs["mito_frac"].to_numpy()])
    pcs = pca_scores(standardise(R), n_pcs, seed)
    pc1_raw = pca_scores(standardise(D), 1, seed)[:, 0]
    return dict(pcs=pcs, sel=sel, Xlog=Xl, corr_before=float(np.corrcoef(pc1_raw, logd)[0, 1]),
                corr_after=float(np.corrcoef(pcs[:, 0], logd)[0, 1]))


def depth_match(depth_a, depth_b, rng, n_bins=10):
    """Ten joint depth-quantile bins; downsample both groups to the smaller count in each bin."""
    edges = np.quantile(np.r_[depth_a, depth_b], np.linspace(0, 1, n_bins + 1))
    ba = np.clip(np.searchsorted(edges, depth_a, side="right") - 1, 0, n_bins - 1)
    bb = np.clip(np.searchsorted(edges, depth_b, side="right") - 1, 0, n_bins - 1)
    ia, ib = [], []
    for k in range(n_bins):
        a, b = np.where(ba == k)[0], np.where(bb == k)[0]
        m = min(len(a), len(b))
        if m:
            ia.append(rng.choice(a, m, replace=False))
            ib.append(rng.choice(b, m, replace=False))
    return (np.concatenate(ia) if ia else np.array([], int)), (np.concatenate(ib) if ib else np.array([], int))


def binomial_thin(X, target, rng):
    """Binomial count thinning of each cell to a common depth (cells below the target are kept)."""
    X = sparse.csr_matrix(X).copy()
    tot = np.asarray(X.sum(1)).ravel()
    p = np.minimum(1.0, target / np.maximum(tot, 1))
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    X.data = rng.binomial(X.data.astype(np.int64), p[rows]).astype(np.float64)
    X.eliminate_zeros()
    return X


def classify_pair(P, y_idx_a, y_idx_b, covariates, n_perm, seed, n_pcs=20):
    A, B = P[y_idx_a, :n_pcs], P[y_idx_b, :n_pcs]
    X, y = two_group(A, B)
    acc, auc = cv_acc_auc(X, y, seed)
    p = perm_pvalue(X, y, acc, n_perm, seed)
    C = np.vstack([covariates[y_idx_a], covariates[y_idx_b]])
    cacc, cauc = cv_acc_auc((C - C.mean(0)) / np.where(C.std(0) == 0, 1, C.std(0)), y, seed)
    return dict(n_per_group=len(A), accuracy=acc, auc=auc, perm_p=p, covariate_only_accuracy=cacc,
                covariate_only_auc=cauc)


# ---- binned series, CCM, entropies, surrogates ------------------------------------


def embed(x, E=2, tau=1):
    n = len(x) - (E - 1) * tau
    return np.column_stack([x[i * tau:i * tau + n] for i in range(E)])


def ccm_skill(target, source, E=2, tau=1, lib_lengths=range(8, 23, 2), train_frac=0.75):
    """Convergent cross mapping skill: predict `target` from the delay-embedded shadow manifold
    of `source` (E=2, tau=1, E+1 nearest neighbours, exponential weights).  Of 29 embedded
    vectors 75% (22) form the library and 7 the test set; skill (Pearson r) is the maximum over
    library lengths 8..22 step 2.  (Re-implementation of the skccm procedure.)"""
    M = embed(np.asarray(source, float), E, tau)
    y = np.asarray(target, float)[(E - 1) * tau:]
    ntr = int(round(train_frac * len(M)))
    Mtr, ytr, Mte, yte = M[:ntr], y[:ntr], M[ntr:], y[ntr:]
    best = -np.inf
    for L in lib_lengths:
        L = min(L, ntr)
        d = cdist(Mte, Mtr[:L])
        k = min(E + 1, L)
        idx = np.argsort(d, 1)[:, :k]
        dd = np.take_along_axis(d, idx, 1)
        w = np.exp(-dd / np.maximum(dd[:, [0]], 1e-12))
        w /= w.sum(1, keepdims=True)
        pred = (w * ytr[:L][idx]).sum(1)
        if np.std(pred) == 0 or np.std(yte) == 0:
            r = 0.0
        else:
            r = float(np.corrcoef(pred, yte)[0, 1])
        best = max(best, r)
    return best


def permutation_entropy(x, m=3):
    x = np.asarray(x, float)
    pats = np.array([np.argsort(x[i:i + m], kind="stable") for i in range(len(x) - m + 1)])
    _, c = np.unique(pats, axis=0, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(m)))


def transfer_entropy(source, target, levels=3):
    """T_{Y->X} = sum p(x_{t+1},x_t,y_t) log[p(x_{t+1}|x_t,y_t)/p(x_{t+1}|x_t)] on a tertile split."""
    def disc(v):
        q = np.quantile(v, np.linspace(0, 1, levels + 1)[1:-1])
        return np.digitize(v, q)
    x, y = disc(np.asarray(target, float)), disc(np.asarray(source, float))
    xn, xt, yt = x[1:], x[:-1], y[:-1]
    n = len(xn)
    from collections import Counter
    c3 = Counter(zip(xn, xt, yt))
    c2a = Counter(zip(xt, yt))
    c2b = Counter(zip(xn, xt))
    c1 = Counter(xt)
    te = 0.0
    for (a, b, c), v in c3.items():
        te += (v / n) * math.log((v / c2a[(b, c)]) / (c2b[(a, b)] / c1[b]))
    return te


def surrogate_shuffle(x, rng):
    return rng.permutation(x)


def surrogate_phase(x, rng):
    f = np.fft.rfft(x - np.mean(x))
    ph = rng.uniform(0, 2 * np.pi, len(f))
    ph[0] = 0
    if len(x) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=len(x)) + np.mean(x)


def surrogate_iaaft(x, rng, n_iter=100):
    x = np.asarray(x, float)
    amp = np.abs(np.fft.rfft(x))
    srt = np.sort(x)
    s = rng.permutation(x)
    for _ in range(n_iter):
        f = np.fft.rfft(s)
        s = np.fft.irfft(amp * np.exp(1j * np.angle(f)), n=len(x))
        s = srt[np.argsort(np.argsort(s))]
    return s


def surrogate_twin(x, rng, m=2, rec_rate=0.10):
    """Twin surrogates (Thiel et al. 2006): jump between points of the embedded series that have
    identical recurrence neighbourhoods."""
    x = np.asarray(x, float)
    E = embed(x, m, 1)
    n = len(E)
    D = cdist(E, E)
    R = D <= np.quantile(D[np.triu_indices(n, 1)], rec_rate)
    twins = [np.where((R == R[:, [i]]).all(0))[0] for i in range(n)]
    i = int(rng.integers(n))
    out = []
    for _ in range(len(x)):
        j = int(rng.choice(twins[i]))
        out.append(x[j + (m - 1)] if j + m - 1 < len(x) else x[-1])
        i = j + 1 if j + 1 < n else int(rng.integers(n))
    return np.array(out)


def surrogate_p(stat_fn, obs, x, kind, M, rng, upper=True):
    """p = (r+1)/(M+1), r = number of surrogates at least as extreme as the observation."""
    fn = dict(shuffle=surrogate_shuffle, phase=surrogate_phase, iaaft=surrogate_iaaft, twin=surrogate_twin)[kind]
    vals = np.array([stat_fn(fn(x, rng)) for _ in range(M)])
    r = (vals >= obs).sum() if upper else (vals <= obs).sum()
    return (int(r) + 1) / (M + 1)


def build_series(key, values, n_bins=30, key_range=None):
    """Equal-width bins on the ordering key; empty bins linearly interpolated.
    Returns (series, occupied mask)."""
    lo, hi = key_range if key_range is not None else (key.min(), key.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    b = np.clip(np.searchsorted(edges, key, side="right") - 1, 0, n_bins - 1)
    s = np.full(n_bins, np.nan)
    for k in range(n_bins):
        m = b == k
        if m.any():
            s[k] = values[m].mean()
    occ = ~np.isnan(s)
    if occ.sum() >= 2:
        s = np.interp(np.arange(n_bins), np.where(occ)[0], s[occ])
    return s, occ


def ordering_key(day, axis):
    """key = day + within-day axis value rescaled to [0, 1]."""
    key = np.zeros(len(day))
    for d in np.unique(day):
        m = day == d
        a = axis[m]
        rng_ = a.max() - a.min()
        key[m] = d + ((a - a.min()) / rng_ if rng_ > 0 else 0.0)
    return key


def analyse_gse149383(samples_path, dge_dir, outdir, sep="\t", seed=0, n_perm=200, n_surr=200,
                      d2_n=None, d2_draws=200, qc=(300, 30, 0.30), n_random_genes=300, tag=""):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples = pd.read_csv(samples_path, sep=None, engine="python")
    X, genes, obs, log = merge_samples(samples, dge_dir, sep)
    for l in log:
        print(l)
    n_raw = X.shape[0]
    X, obs = qc_filter(X, genes, obs, *qc)
    print(f"raw cells {n_raw}; after QC {X.shape[0]}")
    obs["log_depth"] = np.log(obs["total_counts"])
    obs.groupby("condition").size().rename("cells").to_csv(outdir / f"Table5_cells_per_condition{tag}.csv")

    main = process_counts(X, genes, obs, include_markers=True, seed=seed)
    free = process_counts(X, genes, obs, include_markers=False, seed=seed)
    P, Pf = main["pcs"], free["pcs"]
    print(f"PC1 vs depth: before {main['corr_before']:.3f}, after regression {main['corr_after']:.3f}")
    cond = obs["condition"].to_numpy()
    libc = obs["library"].to_numpy()

    # ---------- trajectory measures ----------
    c0 = P[cond == "Day0", :50].mean(0)
    obs["dist_D0"] = np.linalg.norm(P[:, :50] - c0, axis=1)
    cf0 = Pf[cond == "Day0", :50].mean(0)
    axis = np.linalg.norm(Pf[:, :50] - cf0, axis=1)  # marker-free trajectory axis
    obs["axis_marker_free"] = axis
    r_axes = float(np.corrcoef(obs["dist_D0"], axis)[0, 1])
    try:  # optional diffusion pseudotime
        import anndata as ad
        import scanpy as sc
        a = ad.AnnData(np.zeros((len(P), 1)))
        a.obsm["X_pca"] = P[:, :50]
        sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
        sc.tl.diffmap(a)
        a.uns["iroot"] = int(np.argmin(np.linalg.norm(P[:, :50] - c0, axis=1)))
        sc.tl.dpt(a)
        obs["dpt"] = a.obs["dpt_pseudotime"].to_numpy()
    except Exception as e:  # scanpy optional
        print("diffusion pseudotime skipped:", type(e).__name__)
    obs.groupby("condition")[[c for c in ("dist_D0", "dpt") if c in obs]].mean() \
        .to_csv(outdir / f"trajectory_by_condition{tag}.csv")

    # ---------- classifier controls (depth-matched) ----------
    cov = obs[["log_depth", "mito_frac", "n_genes"]].to_numpy(float)
    depth = obs["total_counts"].to_numpy()
    lib_names = sorted(obs.loc[obs["condition"] == "Day11", "library"].unique())
    pairs = [("Day19_withdrawn vs Day19_retreated", cond == "Day19_withdrawn", cond == "Day19_retreated"),
             ("Day0 vs Day19_withdrawn", cond == "Day0", cond == "Day19_withdrawn"),
             ("Day0 vs Day19_retreated", cond == "Day0", cond == "Day19_retreated")]
    if len(lib_names) >= 2:
        pairs.append((f"Day11 {lib_names[0]} vs {lib_names[1]}",
                      (cond == "Day11") & (libc == lib_names[0]), (cond == "Day11") & (libc == lib_names[1])))
    t6, matched = [], {}
    for name, ma, mb in pairs:
        ia, ib = np.where(ma)[0], np.where(mb)[0]
        sa, sb = depth_match(depth[ia], depth[ib], rng)
        ga, gb = ia[sa], ib[sb]
        matched[name] = (ga, gb)
        res = classify_pair(P, ga, gb, cov, 1000, seed)
        res.update(pair=name, median_depth_a=float(np.median(depth[ga])), median_depth_b=float(np.median(depth[gb])),
                   mannwhitney_depth_p=float(stats.mannwhitneyu(depth[ga], depth[gb]).pvalue))
        t6.append(res)
    t6 = pd.DataFrame(t6)
    t6.to_csv(outdir / f"Table6_classifier_depth_matched{tag}.csv", index=False)
    accs = dict(zip(t6["pair"], t6["accuracy"]))
    wr = accs.get("Day19_withdrawn vs Day19_retreated")
    rep_key = next((k for k in accs if k.startswith("Day11")), None)
    if wr is not None and rep_key:
        rho = rho_accuracy(wr, accs[rep_key])
        pd.DataFrame([dict(contrast_accuracy=wr, replicate_accuracy=accs[rep_key], rho_accuracy=rho)]) \
            .to_csv(outdir / f"replicate_referenced_ratio{tag}.csv", index=False)
        print(f"replicate-referenced ratio rho = {rho:.2f}")

    # depth / doublet controls
    t7 = []
    Xthin = binomial_thin(X, 600, rng)
    obs_thin = obs.copy()
    obs_thin["total_counts"] = np.asarray(Xthin.sum(1)).ravel()
    obs_thin["mito_frac"] = obs_thin["mito_frac"]
    keep_thin = obs_thin["total_counts"].to_numpy() > 0
    Pt = np.zeros_like(P)
    Pt_rows = np.where(keep_thin)[0]
    pt = process_counts(Xthin[Pt_rows], genes, obs_thin.iloc[Pt_rows], include_markers=True, seed=seed)["pcs"]
    Pt[Pt_rows, :pt.shape[1]] = pt
    for name in (pairs[0][0], pairs[-1][0]):
        ga, gb = matched[name]
        r = classify_pair(Pt, ga, gb, cov, 200, seed)
        t7.append(dict(pair=name, check="binomial thinning to 600 counts", accuracy=r["accuracy"],
                       auc=r["auc"], perm_p=r["perm_p"]))
    try:
        import scrublet as scr
        dbl = np.zeros(X.shape[0], bool)
        for l in np.unique(libc):
            m = np.where(libc == l)[0]
            sc_ = scr.Scrublet(X[m])
            _, pred = sc_.scrub_doublets(verbose=False)
            dbl[m] = pred
        obs["predicted_doublet"] = dbl
        for name in (pairs[0][0], pairs[-1][0]):
            ga, gb = matched[name]
            ga, gb = ga[~dbl[ga]], gb[~dbl[gb]]
            r = classify_pair(P, ga, gb, cov, 200, seed)
            t7.append(dict(pair=name, check="Scrublet doublets removed", accuracy=r["accuracy"],
                           auc=r["auc"], perm_p=r["perm_p"]))
    except Exception as e:
        print("Scrublet control skipped:", type(e).__name__)
    pd.DataFrame(t7).to_csv(outdir / f"Table7_depth_doublet_controls{tag}.csv", index=False)

    # ---------- unimodality: dip test and 1- vs 2-component mixtures ----------
    ga, gb = matched[pairs[0][0]]
    sel = np.r_[ga, gb]
    truth = np.r_[np.zeros(len(ga), int), np.ones(len(gb), int)]
    dip, p = dip_pvalue(P[sel, 0])
    g1 = GaussianMixture(1, random_state=seed).fit(P[sel, :2])
    g2 = GaussianMixture(2, random_state=seed, n_init=5).fit(P[sel, :2])
    lab = g2.predict(P[sel, :2])
    agree = max((lab == truth).mean(), (lab != truth).mean())
    pd.DataFrame([dict(dip=dip, dip_p=p, delta_BIC_1_minus_2=g1.bic(P[sel, :2]) - g2.bic(P[sel, :2]),
                       cluster_label_agreement=float(agree),
                       supervised_accuracy=float(t6.loc[0, "accuracy"]))]) \
        .to_csv(outdir / f"dip_gmm_reversibility{tag}.csv", index=False)

    # ---------- correlation dimension at equal n ----------
    conds = [c for c in ["Day0", "Day1", "Day2", "Day4", "Day9", "Day11", "Day19_withdrawn", "Day19_retreated"]
             if (cond == c).any()]
    n_eq = d2_n or int(min((cond == c).sum() for c in conds))
    d2_rows, pcs_rows = [], []
    for npc in (10, 15, 20, 30):
        radii = radius_grid(P[:, :npc], max_cells=2000, seed=seed)
        for c in conds:
            ii = np.where(cond == c)[0]
            nd = d2_draws if npc == 15 else 50
            v = np.array([correlation_dimension(P[rng.choice(ii, n_eq, replace=False), :npc], radii)
                          for _ in range(nd)])
            (d2_rows if npc == 15 else pcs_rows).append(
                dict(condition=c, n_pcs=npc, cells_available=len(ii), n_equal=n_eq, D2=np.nanmean(v),
                     lo=q025_975(v[~np.isnan(v)])[0], hi=q025_975(v[~np.isnan(v)])[1]))
            if npc == 15:
                pcs_rows.append(d2_rows[-1] | dict(condition=c))
    pd.DataFrame(d2_rows).to_csv(outdir / f"TableE1_D2_equal_n{tag}.csv", index=False)
    pd.DataFrame(pcs_rows).to_csv(outdir / f"S1_D2_vs_n_PCs{tag}.csv", index=False)
    # label-permutation test of D2(withdrawn) - D2(re-treated) at equal n
    iw, ir = np.where(cond == "Day19_withdrawn")[0], np.where(cond == "Day19_retreated")[0]
    radii = radius_grid(P[:, :15], seed=seed)

    def d2diff(a, b):
        return (correlation_dimension(P[rng.choice(a, n_eq, replace=False), :15], radii)
                - correlation_dimension(P[rng.choice(b, n_eq, replace=False), :15], radii))
    if len(iw) and len(ir):
        obs_d = np.mean([d2diff(iw, ir) for _ in range(20)])
        pool = np.r_[iw, ir]
        null = []
        for _ in range(n_perm):
            pp = rng.permutation(pool)
            null.append(d2diff(pp[:len(iw)], pp[len(iw):]))
        pv = (int((np.abs(null) >= abs(obs_d)).sum()) + 1) / (n_perm + 1)
        pd.DataFrame([dict(diff_withdrawn_minus_retreated=obs_d, perm_p=pv)]) \
            .to_csv(outdir / f"D2_withdrawn_vs_retreated_perm{tag}.csv", index=False)
    # first pass (superseded): bootstrap at each condition's own n, 80% with replacement
    fp = []
    for c in conds:
        ii = np.where(cond == c)[0]
        radii_c = radius_grid(P[ii, :15], max_cells=2000, seed=seed)
        v = []
        for _ in range(200):
            b = rng.choice(ii, max(int(0.8 * len(ii)), 5), replace=True)
            Xb = P[b, :15]
            Xb = np.unique(Xb, axis=0)
            v.append(correlation_dimension(Xb, radii_c))
        fp.append(dict(condition=c, n=len(ii), D2_first_pass=np.nanmean(v), lo=np.nanpercentile(v, 2.5),
                       hi=np.nanpercentile(v, 97.5)))
    pd.DataFrame(fp).to_csv(outdir / f"S2_D2_first_pass{tag}.csv", index=False)

    # ---------- binned series: CCM, permutation entropy, transfer entropy ----------
    traj = obs[obs["condition"].isin(DAYS)].copy()
    day = traj["condition"].map(DAYS).to_numpy(float)
    key = ordering_key(day, traj["axis_marker_free"].to_numpy())
    kr = (0.0, float(max(DAYS.values()) + 1))
    Xlog = main["Xlog"][np.where(obs["condition"].isin(DAYS))[0]]
    gl = {g: i for i, g in enumerate(genes)}
    ser_axis, occ = build_series(key, traj["axis_marker_free"].to_numpy(), key_range=kr)
    series = {"axis": ser_axis}
    for m in MARKERS:
        if m in gl:
            series[m] = build_series(key, np.asarray(Xlog[:, gl[m]].todense()).ravel(), key_range=kr)[0]
    print(f"binned series: {occ.sum()} of 30 bins occupied")
    ccm_rows = []
    for m in [g for g in MARKERS if g in series]:
        for direction, tgt, src in (("gene->trajectory", series[m], ser_axis),
                                    ("trajectory->gene", ser_axis, series[m])):
            sk = ccm_skill(tgt, src)
            row = dict(marker=m, direction=direction, skill=sk)
            fn = lambda s_, src=src: ccm_skill(s_, src)
            for kind in ("shuffle", "phase", "iaaft", "twin"):
                row[f"p_{kind}"] = surrogate_p(fn, sk, tgt, kind, n_surr, rng)
            ccm_rows.append(row)
    ccm = pd.DataFrame(ccm_rows)
    ccm["q_BH_18"] = benjamini_hochberg(ccm["p_shuffle"].to_numpy())
    ccm.to_csv(outdir / f"S4_CCM{tag}.csv", index=False)
    # permutation entropy: all 30 bins and occupied bins only
    pe_rows = []
    for name, s in series.items():
        pe = permutation_entropy(s)
        p_all = surrogate_p(permutation_entropy, pe, s, "shuffle", n_surr, rng, upper=False)
        s_occ = build_series(key, traj["axis_marker_free"].to_numpy() if name == "axis" else
                             np.asarray(Xlog[:, gl[name]].todense()).ravel(), key_range=kr)
        # occupied bins only: recompute without interpolation
        lo, hi = kr
        edges = np.linspace(lo, hi, 31)
        b = np.clip(np.searchsorted(edges, key, side="right") - 1, 0, 29)
        vals = traj["axis_marker_free"].to_numpy() if name == "axis" else np.asarray(Xlog[:, gl[name]].todense()).ravel()
        so = np.array([vals[b == k].mean() for k in range(30) if (b == k).any()])
        pe_o = permutation_entropy(so)
        p_occ = surrogate_p(permutation_entropy, pe_o, so, "shuffle", 1000, rng, upper=False)
        pe_rows.append(dict(series=name, PE_all_bins=pe, p_all_bins=p_all, PE_occupied=pe_o, p_occupied=p_occ))
    pe_df = pd.DataFrame(pe_rows)
    pe_df["q_BH_all_bins"] = benjamini_hochberg(pe_df["p_all_bins"].to_numpy())
    pe_df.to_csv(outdir / f"S6_permutation_entropy{tag}.csv", index=False)
    # transfer entropy (source shuffled, target intact); p = fraction of surrogates >= observed
    te_rows = []
    for m in [g for g in MARKERS if g in series]:
        for direction, src, tgt in (("gene->trajectory", series[m], ser_axis),
                                    ("trajectory->gene", ser_axis, series[m])):
            te = transfer_entropy(src, tgt)
            nul = np.array([transfer_entropy(rng.permutation(src), tgt) for _ in range(n_surr)])
            te_rows.append(dict(marker=m, direction=direction, TE=te, p=float((nul >= te).mean())))
    te_df = pd.DataFrame(te_rows)
    te_df["q_BH_18"] = benjamini_hochberg(te_df["p"].to_numpy())
    te_df.to_csv(outdir / f"S7_transfer_entropy{tag}.csv", index=False)

    # ---------- random-gene control (matched for mean expression and detection rate) ----------
    Xl_traj = Xlog.tocsc()
    mean_e = np.asarray(Xl_traj.mean(0)).ravel()
    det = np.asarray((Xl_traj > 0).mean(0)).ravel()
    feat = np.column_stack([(mean_e - mean_e.mean()) / (mean_e.std() or 1), (det - det.mean()) / (det.std() or 1)])
    mk_idx = {gl[m] for m in MARKERS if m in gl}
    rg_rows = []
    for m in ("EGFR", "SOX4"):
        if m not in gl:
            continue
        i = gl[m]
        dist = np.linalg.norm(feat - feat[i], axis=1)
        cand = [j for j in np.argsort(dist) if j not in mk_idx and mean_e[j] > 0][:max(5 * n_random_genes, 50)]
        pick = rng.choice(cand, min(n_random_genes, len(cand)), replace=False)
        skills = np.array([ccm_skill(build_series(key, np.asarray(Xl_traj[:, j].todense()).ravel(), key_range=kr)[0],
                                     ser_axis) for j in pick])
        obs_sk = float(ccm.query("marker == @m and direction == 'gene->trajectory'")["skill"].iloc[0])
        r = int((skills >= obs_sk).sum())
        rg_rows.append(dict(marker=m, skill=obs_sk, random_mean=skills.mean(), random_sd=skills.std(),
                            random_p95=np.percentile(skills, 95),
                            percentile=100 * (skills < obs_sk).mean(), p_one_sided=(r + 1) / (len(skills) + 1)))
    pd.DataFrame(rg_rows).to_csv(outdir / f"TableE2_random_gene_control{tag}.csv", index=False)

    # ---------- direct marker comparison (withdrawn vs re-treated, depth matched) ----------
    ga, gb = matched[pairs[0][0]]
    mk_rows = []
    for m in MARKERS:
        if m in gl:
            va = np.asarray(main["Xlog"][ga][:, gl[m]].todense()).ravel()
            vb = np.asarray(main["Xlog"][gb][:, gl[m]].todense()).ravel()
            mk_rows.append(dict(marker=m, mannwhitney_p=float(stats.mannwhitneyu(va, vb).pvalue),
                                mean_withdrawn=va.mean(), mean_retreated=vb.mean()))
    pd.DataFrame(mk_rows).to_csv(outdir / f"markers_withdrawn_vs_retreated{tag}.csv", index=False)
    pd.DataFrame([dict(r_marker_free_vs_marker_included_axis=r_axes,
                       pc1_depth_r_before=main["corr_before"], pc1_depth_r_after=main["corr_after"])]) \
        .to_csv(outdir / f"depth_correction_summary{tag}.csv", index=False)
    print("GSE149383 analysis written to", outdir)


# =====================================================================================
# 5. Simulation benchmark (two-gene toggle switch, negative-binomial counts, library effects)
#    NOTE: reconstructed from the description in the main text; Additional file 2 defines the
#    original settings, so exact numbers differ.
# =====================================================================================

STRUCTURES = ["low_rank", "high_dim", "depth_only", "sparse", "mixed"]


class SimWorld:
    def __init__(self, n_genes=200, rank=3, seed=0):
        r = np.random.default_rng(seed)
        self.G = n_genes
        self.base = r.normal(0.8, 0.7, n_genes)
        self.load = np.zeros((n_genes, 2))
        self.load[0, 0] = self.load[1, 1] = 1.0
        resp = r.random(n_genes) < 0.3
        self.load[2:][resp[2:]] = r.normal(0, 0.35, (resp[2:].sum(), 2))
        self.W = r.normal(0, 1, (rank, n_genes))
        self.sparse_mask = r.random(n_genes) < 0.1


def sim_latent(n, bistable, shift, rng, sigma=0.35, dt=0.05, steps=300):
    """Euler-Maruyama toggle switch.  Hill coefficient 4 (bistable) or 1 (monostable)."""
    a, K, h = 3.0 * (1 + shift), 1.0, (4 if bistable else 1)
    u, v = rng.uniform(0.1, 3, n), rng.uniform(0.1, 3, n)
    for _ in range(steps):
        du = a / (1 + (v / K) ** h) - u
        dv = a / (1 + (u / K) ** h) - v
        u = np.abs(u + du * dt + sigma * np.sqrt(dt) * rng.standard_normal(n))
        v = np.abs(v + dv * dt + sigma * np.sqrt(dt) * rng.standard_normal(n))
    return np.column_stack([u, v])


def library_effect(world, structure, scale, rng):
    G = world.G
    z = rng.standard_normal(world.W.shape[0])
    if structure == "low_rank":
        return scale * (z[:2] @ world.W[:2]) / np.sqrt(2), 0.0
    if structure == "high_dim":
        return scale * rng.standard_normal(G), 0.0
    if structure == "depth_only":
        return np.zeros(G), scale * rng.standard_normal()
    if structure == "sparse":
        e = np.zeros(G)
        e[world.sparse_mask] = 3 * scale * rng.standard_normal(world.sparse_mask.sum())
        return e, 0.0
    if structure == "mixed":
        return scale * (z[:2] @ world.W[:2]) / np.sqrt(2) + 0.5 * scale * rng.standard_normal(G), 0.0
    raise ValueError(structure)


def sim_library(world, structure, scale, n_cells, rng, bistable=False, shift=0.0, theta=0.5):
    z = sim_latent(n_cells, bistable, shift, rng)
    eff, dshift = library_effect(world, structure, scale, rng)
    size = np.exp(rng.normal(0, 0.3, n_cells) + dshift)
    logmu = world.base[None, :] + 0.5 * (z - 1.5) @ world.load.T + eff[None, :]
    mu = np.exp(logmu) * size[:, None]
    r = 1.0 / theta
    return rng.negative_binomial(r, r / (r + mu)).astype(np.float64), z


def sim_pca(count_blocks, n_pcs=20, seed=0):
    X = np.vstack(count_blocks)
    X = sparse.csr_matrix(X)
    tot = np.asarray(X.sum(1)).ravel()
    keep = tot > 0
    Xl = np.asarray(log_normalise(X).todense())
    R = regress_out(Xl, [np.log(np.maximum(tot, 1))])
    pcs = pca_scores(standardise(R), n_pcs, seed)
    out, k = [], 0
    for b in count_blocks:
        out.append(pcs[k:k + len(b)])
        k += len(b)
    return out


_CAL = {}


def calibrate_scale(world, structure, target=0.93, n_cells=195, seed=0):
    """Bisect the library-effect scale so two same-condition libraries separate at ~target accuracy."""
    key = (structure, target, n_cells, world.G)
    if key in _CAL:
        return _CAL[key]
    lo, hi = 0.0, 6.0
    for _ in range(9):
        mid = (lo + hi) / 2
        accs = []
        for rep in range(3):
            r = np.random.default_rng(seed + 100 * rep + 1)
            a, _ = sim_library(world, structure, mid, n_cells, r)
            b, _ = sim_library(world, structure, mid, n_cells, r)
            pa, pb = sim_pca([a, b], 20, seed)
            X, y = two_group(pa, pb)
            accs.append(cv_accuracy(X, y, seed)[0])
        if np.mean(accs) < target:
            lo = mid
        else:
            hi = mid
    _CAL[key] = (lo + hi) / 2
    return _CAL[key]


def library_permutation_ratio_test(ED, n_libs):
    """Permute library -> arm labels (all balanced splits); statistic = across/within distance ratio."""
    L = 2 * n_libs
    def stat(arm):
        a = np.array(arm)
        across = [ED[i, j] for i in range(L) for j in range(i + 1, L) if a[i] != a[j]]
        within = [ED[i, j] for i in range(L) for j in range(i + 1, L) if a[i] == a[j]]
        return np.mean(across) / max(np.mean(within), 1e-9)
    obs = stat([0] * n_libs + [1] * n_libs)
    splits = [s for s in itertools.combinations(range(L), n_libs) if 0 in s]
    nulls = []
    for s in splits:
        arm = [1] * L
        for i in s:
            arm[i] = 0
        nulls.append(stat(arm))
    return obs, (int((np.array(nulls) >= obs - 1e-12).sum())) / len(nulls)


def sim_once(world, structure, scale, n_libs, effect, rng, n_cells=195, perms=100):
    shift = {"none": 0.0, "small": 0.12, "large": 0.6}[effect]
    A = [sim_library(world, structure, scale, n_cells, rng)[0] for _ in range(n_libs)]
    B = [sim_library(world, structure, scale, n_cells, rng, shift=shift)[0] for _ in range(n_libs)]
    P = sim_pca(A + B, 20, int(rng.integers(1 << 20)))
    Xs, ys = two_group(P[0], P[n_libs])
    acc_single, _ = cv_accuracy(Xs, ys, 0)
    p_single = perm_pvalue(Xs, ys, acc_single, perms, 0)
    Xr, yr = two_group(P[0], P[1])
    acc_rep, _ = cv_accuracy(Xr, yr, 0)
    L = 2 * n_libs
    ED = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            ED[i, j] = ED[j, i] = energy_distance(P[i], P[j])
    across = [ED[i, j] for i in range(n_libs) for j in range(n_libs, L)]
    within = [ED[i, j] for i in range(n_libs) for j in range(i + 1, n_libs)] + \
             [ED[i, j] for i in range(n_libs, L) for j in range(i + 1, L)]
    out = dict(structure=structure, n_libs=n_libs, effect=effect, scale=scale, acc_single=acc_single,
               p_single=p_single, called_separable=bool(p_single < 0.05), acc_replicate=acc_rep,
               rho_accuracy=rho_accuracy(acc_single, acc_rep), rho_ED=float(np.mean(across) / max(np.mean(within), 1e-9)))
    if n_libs >= 3:
        _, out["library_perm_p"] = library_permutation_ratio_test(ED, n_libs)
    return out


def sim_dip_once(world, structure, scale, n_libs, rng, n_cells=300):
    """Bistable arm vs monostable arm: per-library dip test (all libraries must agree) vs pooled."""
    res = {}
    for truth, bist in (("bistable", True), ("monostable", False)):
        libs = [sim_library(world, structure, scale, n_cells, rng, bistable=bist) for _ in range(n_libs)]
        P = sim_pca([c for c, _ in libs], 20, int(rng.integers(1 << 20)))
        per = [dip_pvalue(within_pc1(p))[1] < 0.05 for p in P]
        pooled = dip_pvalue(within_pc1(np.vstack(P)))[1] < 0.05
        res[truth] = dict(per_library_all=all(per), pooled=pooled)
    return res


def sim_d2_vs_pcs(world, structure, scale, rng, n_cells=600, depths=(0.5, 1.0, 2.0)):
    rows = []
    for dm in depths:
        c, _ = sim_library(world, structure, scale, n_cells, rng, bistable=False)
        c = rng.binomial(c.astype(int), min(1.0, dm / 2.0)) if dm < 2.0 else c
        P = sim_pca([c], 30, 0)[0]
        for npc in (5, 10, 20, 30):
            radii = radius_grid(P[:, :npc], max_cells=600)
            rows.append(dict(depth_factor=dm, n_pcs=npc, D2=correlation_dimension(P[:, :npc], radii), latent_dim=2))
    return rows


def sim_interpolated_pe(rng, n_sim=100, n_bins=30, n_occ=17):
    """Permutation entropy of binned series: noise-only series, with 13 of 30 bins filled by linear
    interpolation, versus the same series on occupied bins only."""
    rows = []
    occ_idx = np.sort(rng.choice(n_bins, n_occ, replace=False))
    for _ in range(n_sim):
        raw = rng.standard_normal(n_occ)
        full = np.interp(np.arange(n_bins), occ_idx, raw)
        for kind, s in (("interpolated_30_bins", full), ("occupied_only", raw)):
            pe = permutation_entropy(s)
            p = surrogate_p(permutation_entropy, pe, s, "shuffle", 100, rng, upper=False)
            rows.append(dict(kind=kind, PE=pe, p=p, significant=p < 0.05))
    return rows


def run_simulation(outdir, n_sim=20, seed=0, n_cells=195, structures=STRUCTURES, n_libs_list=(2, 3, 4)):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    world = SimWorld(seed=seed)
    cal = []
    rows, dip_rows = [], []
    for st in structures:
        scale = calibrate_scale(world, st, 0.93, n_cells, seed)
        cal.append(dict(structure=st, scale=scale))
        for nl in n_libs_list if st == "low_rank" else (2,):
            for eff in ("none", "small", "large"):
                for _ in range(n_sim):
                    rows.append(sim_once(world, st, scale, nl, eff, rng, n_cells))
    sims = pd.DataFrame(rows)
    sims.to_csv(outdir / "S11_simulation_runs.csv", index=False)
    pd.DataFrame(cal).to_csv(outdir / "S12_calibration.csv", index=False)
    # summaries: single-library false positives, null threshold for rho_ED, power
    summ = []
    for (st, nl), g in sims.groupby(["structure", "n_libs"]):
        null = g[g.effect == "none"]
        thr = float(np.percentile(null["rho_ED"], 95)) if len(null) else np.nan
        for eff in ("none", "small", "large"):
            e = g[g.effect == eff]
            row = dict(structure=st, n_libs=nl, effect=eff, single_library_called_separable=e["called_separable"].mean(),
                       mean_rho_accuracy=e["rho_accuracy"].mean(), null_rho_ED_p95=thr,
                       frac_rho_ED_above_null_p95=(e["rho_ED"] > thr).mean())
            if "library_perm_p" in e and e["library_perm_p"].notna().any():
                row["power_library_perm_p_lt_0.05"] = (e["library_perm_p"] < 0.05).mean()
            summ.append(row)
    pd.DataFrame(summ).to_csv(outdir / "S12_simulation_summary.csv", index=False)
    # bimodality: per-library (must agree) versus pooled libraries
    st = "low_rank"
    scale = calibrate_scale(world, st, 0.93, n_cells, seed)
    for _ in range(max(5, n_sim // 2)):
        r = sim_dip_once(world, st, scale, 2, rng)
        for truth, v in r.items():
            dip_rows.append(dict(truth=truth, **v))
    dd = pd.DataFrame(dip_rows)
    dd.groupby("truth")[["per_library_all", "pooled"]].mean().to_csv(outdir / "S12_dip_test_pooled_vs_per_library.csv")
    d2 = pd.DataFrame(sim_d2_vs_pcs(world, st, scale, rng))
    d2.to_csv(outdir / "S12_D2_vs_PCs_depth.csv", index=False)
    pe = pd.DataFrame(sim_interpolated_pe(rng, n_sim=max(20, n_sim)))
    pe.groupby("kind")["significant"].mean().to_csv(outdir / "S12_permutation_entropy_interpolation.csv")
    print("simulation written to", outdir)


# =====================================================================================
# 6. Self-test on synthetic data (no downloads)
# =====================================================================================


def _synthetic_gse150949(path, n_per_lib=1500, seed=0):
    r = np.random.default_rng(seed)
    rows = []
    sizes = {l: n_per_lib for l in LIBS_150949}
    centres = {"0": 0, "3": 3, "7": 2.5, "14": 3.2}
    for l in LIBS_150949:
        base = centres[l.split("_")[0]]
        lib_shift = r.normal(0, 0.25, 50)
        pcs = r.normal(0, 1, (sizes[l], 50)) * np.r_[np.full(5, 2.0), np.ones(45)] + lib_shift
        pcs[:, 0] += base * 2
        if "low" in l:
            pcs[:, 1] += 1.5
        if "high" in l:
            pcs[:, 1] -= 1.5
        bc = r.integers(0, n_per_lib // 4, sizes[l])
        centres_bc = r.normal(0, 0.8, (n_per_lib // 4, 50))
        pcs = pcs + centres_bc[bc] * 0.8
        for i in range(sizes[l]):
            rows.append([f"{l}_{i}", l, f"{l}_{bc[i]}"] + list(pcs[i]))
    df = pd.DataFrame(rows, columns=["cell", "sample_name", "lineage_barcode"] + [f"PC{i}" for i in range(1, 51)])
    df.set_index("cell").to_csv(path)


def _synthetic_gse149383(dirpath, seed=0, n_genes=600):
    r = np.random.default_rng(seed)
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    genes = [f"G{i}" for i in range(n_genes - 12)] + MARKERS + ["MT-CO1", "MT-ND1", "MT-ATP6"]
    sheet = []
    spec = [("Day0", "Day0", 400), ("Day1", "Day1", 300), ("Day2", "Day2", 250), ("Day4", "Day4", 200),
            ("Day9", "Day9", 300), ("Day11_A", "Day11", 300), ("Day11_B", "Day11", 300),
            ("Day19_withdrawn", "Day19_withdrawn", 350), ("Day19_retreated", "Day19_retreated", 350)]
    base = r.gamma(1.0, 0.6, len(genes))
    for i, (lib, cond, n) in enumerate(spec):
        eff = np.ones(len(genes))
        eff[:60] = np.exp(r.normal(0.8 * i / 9, 0.3, 60))
        eff *= np.exp(r.normal(0, 0.2, len(genes)))
        depth = np.exp(r.normal(6.0, 0.5, n))
        mu = base[None, :] * eff[None, :]
        mu = mu / mu.sum(1, keepdims=True) * depth[:, None]
        counts = r.poisson(mu).T
        df = pd.DataFrame(counts, index=genes, columns=[f"{lib}_cell{j}" for j in range(n)])
        df.index.name = "GENE"
        fn = f"{lib}.txt.gz"
        df.to_csv(dirpath / fn, sep="\t")
        sheet.append(dict(file=fn, condition=cond, library=lib))
    # duplicate re-upload that must be excluded
    (dirpath / "Day0_dup.txt.gz").write_bytes((dirpath / "Day0.txt.gz").read_bytes())
    sheet.append(dict(file="Day0_dup.txt.gz", condition="Day0", library="Day0_dup"))
    pd.DataFrame(sheet).to_csv(dirpath / "samples.tsv", sep="\t", index=False)


def selftest():
    tmp = Path(tempfile.mkdtemp(prefix="attractor_selftest_"))
    print("self-test directory:", tmp)
    # ---- unit checks
    r = np.random.default_rng(0)
    A, B = r.normal(0, 1, (300, 5)), r.normal(0, 1, (300, 5))
    assert abs(energy_distance(A, B)) < 0.2, "energy distance should be ~0 for identical distributions"
    assert energy_distance(A, B + 1.5) > 1, "energy distance should be large for shifted data"
    q = benjamini_hochberg([0.005, 0.01, 0.04, 0.5])
    assert np.all(q >= np.array([0.005, 0.01, 0.04, 0.5]) - 1e-12) and q[-1] <= 1
    X = r.normal(0, 1, (1500, 8))
    radii = radius_grid(X, max_cells=800)
    d2 = correlation_dimension(X[:800], radii)
    assert 2 < d2 < 9, d2
    s = np.sin(np.linspace(0, 6, 30)) + r.normal(0, .05, 30)
    assert 0 <= permutation_entropy(s) <= 1
    assert ccm_skill(s, s) > 0.5
    for kind in ("shuffle", "phase", "iaaft", "twin"):
        sur = dict(shuffle=surrogate_shuffle, phase=surrogate_phase, iaaft=surrogate_iaaft,
                   twin=surrogate_twin)[kind](s, r)
        assert len(sur) == len(s)
    assert transfer_entropy(s, s) >= 0
    # ---- GSE150949
    pca = tmp / "GSE150949_PCA50.csv.gz"
    _synthetic_gse150949(pca)
    res = tmp / "res150949"
    analyse_gse150949(pca, res, n_cells=400, n_draws=3, d2_cells=400, d2_draws=2, n_shuffles=30)
    fl = pd.read_csv(res / "S8_noise_floor_draws.csv")
    half = fl[fl.kind == "split_half"]["accuracy"]
    assert 0.40 < half.mean() < 0.60, "split halves should be at chance"
    assert fl[fl.kind == "replicate"]["accuracy"].mean() > half.mean()
    ct = pd.read_csv(res / "S9_contrasts_draws.csv")
    assert (ct["rho_ED"] > 1).mean() > 0.5
    # ---- raw 10x-style preprocessing round trip
    cnt = pd.DataFrame(np.random.default_rng(1).poisson(0.8, (300, 400)),
                       index=[f"g{i}" for i in range(297)] + ["MT-A", "MT-B", "MT-C"],
                       columns=[f"c{i}" for i in range(400)])
    cnt.to_csv(tmp / "counts.csv.gz")
    pd.DataFrame({"sample_name": ["0"] * 400, "lineage_barcode": ["x"] * 400},
                 index=cnt.columns).to_csv(tmp / "meta.csv")
    out = preprocess_gse150949(tmp / "counts.csv.gz", tmp / "meta.csv", tmp / "pca_from_counts.csv.gz", n_hvg=100,
                               n_pcs=10)
    assert "PC10" in out.columns
    # ---- GSE149383
    dge = tmp / "dge"
    _synthetic_gse149383(dge)
    res2 = tmp / "res149383"
    analyse_gse149383(dge / "samples.tsv", dge, res2, n_perm=20, n_surr=30, d2_draws=10, n_random_genes=20)
    t5 = pd.read_csv(res2 / "Table5_cells_per_condition.csv")
    assert "Day0" in set(t5["condition"]) and t5["cells"].sum() < 3000, "duplicate upload must be excluded"
    for fn in ("Table6_classifier_depth_matched.csv", "S4_CCM.csv", "S6_permutation_entropy.csv",
               "S7_transfer_entropy.csv", "TableE1_D2_equal_n.csv", "TableE2_random_gene_control.csv"):
        assert (res2 / fn).exists(), fn
    # ---- simulation
    res3 = tmp / "sim"
    run_simulation(res3, n_sim=3, structures=["low_rank", "high_dim"], n_libs_list=(2, 3), n_cells=120)
    assert (res3 / "S12_simulation_summary.csv").exists()
    print("SELFTEST PASSED. Outputs in", tmp)


# =====================================================================================
# 7. Command line
# =====================================================================================


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("preprocess-gse150949", help="GEO counts + metadata -> 50 PCs per cell")
    s.add_argument("--counts", required=True, help="genes x cells csv/tsv(.gz), first column = gene symbol")
    s.add_argument("--meta", required=True, help="per-cell metadata csv (index = cell id, has the library column)")
    s.add_argument("--out", default="GSE150949_PCA50.csv.gz")
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("gse150949", help="noise floor, contrasts, dip, D2, trajectory, lineage")
    s.add_argument("--pca", required=True)
    s.add_argument("--out", default="results")
    s.add_argument("--lib-col", default="sample_name")
    s.add_argument("--barcode-col", default="lineage_barcode")
    s.add_argument("--n-cells", type=int, default=1300)
    s.add_argument("--n-draws", type=int, default=20)
    s.add_argument("--seed", type=int, default=20240101)

    s = sub.add_parser("library-distance", help="all-library energy distance matrices, 10/20/50 PCs")
    s.add_argument("--pca", required=True)
    s.add_argument("--out", default="results")
    s.add_argument("--lib-col", default="sample_name")

    s = sub.add_parser("gse149383", help="Drop-seq erlotinib pipeline")
    s.add_argument("--samples", required=True,
                   help="tsv/csv with columns file, condition, library. conditions: Day0 Day1 Day2 Day4 Day9 Day11 "
                        "Day19_withdrawn Day19_retreated; the two Day 11 depositions get distinct library names")
    s.add_argument("--dge-dir", required=True)
    s.add_argument("--out", default="results_gse149383")
    s.add_argument("--sep", default="\t")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--qc-counts", type=int, default=300)
    s.add_argument("--qc-genes", type=int, default=30)
    s.add_argument("--qc-mito", type=float, default=0.30)
    s.add_argument("--tag", default="", help="suffix for output files (use with a second QC threshold, "
                                              "e.g. --qc-counts 250 --qc-genes 25 --qc-mito 0.35 --tag _qc2)")

    s = sub.add_parser("simulate", help="toggle-switch simulation benchmark")
    s.add_argument("--out", default="results_simulation")
    s.add_argument("--n-sim", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)

    sub.add_parser("selftest", help="run everything on small synthetic data")
    a = ap.parse_args(argv)

    if a.cmd == "preprocess-gse150949":
        preprocess_gse150949(a.counts, a.meta, a.out, seed=a.seed)
    elif a.cmd == "gse150949":
        analyse_gse150949(a.pca, a.out, a.lib_col, a.barcode_col, a.n_cells, a.n_draws, seed=a.seed)
    elif a.cmd == "library-distance":
        library_distance_matrix(a.pca, a.out, a.lib_col)
    elif a.cmd == "gse149383":
        analyse_gse149383(a.samples, a.dge_dir, a.out, a.sep, a.seed,
                          qc=(a.qc_counts, a.qc_genes, a.qc_mito), tag=a.tag)
    elif a.cmd == "simulate":
        run_simulation(a.out, a.n_sim, a.seed)
    elif a.cmd == "selftest":
        selftest()


if __name__ == "__main__":
    main()
