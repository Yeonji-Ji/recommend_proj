import pandas as pd
import numpy as np


def precision_recall_ndcg_at_k(df, k=10, label="label", score="pred"):
    precs, recs, ndcgs = [], [], []

    for u, g in df.groupby("u", sort=False):
        g = g.sort_values(
            score, ascending=False
        )  # sort the result data based on prediction score
        topk = g.head(k)  # pick top k scored (predicted) movies
        hits = int(
            topk[label].sum()
        )  # movies actually rated over 4.0 in the predicted topk
        total_pos = int(g[label].sum())  # movies actually rated over 4.0
        precs.append(hits / k)  # Precision
        if total_pos > 0:
            recs.append(hits / total_pos)  # Recall

        rel = topk[label].to_numpy()
        if rel.size:
            discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
            dcg = float((rel * discounts).sum())
            ideal = np.sort(rel)[::-1]
            idcg = float((ideal * discounts).sum()) if ideal.sum() > 0 else 0.0
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)  # NDCG

    P = float(np.mean(precs)) if precs else 0.0
    R = float(np.mean(recs)) if recs else 0.0
    N = float(np.mean(ndcgs)) if ndcgs else 0.0

    return P, R, N


def metrics_over_K(df, ks, label="label", score="pred"):
    rows = []
    for k in ks:
        P, R, N = precision_recall_ndcg_at_k(df, k=k, label=label, score=score)
        rows.append({"K": k, "Precision": P, "Recall": R, "NDCG": N})
    return pd.DataFrame(rows)
