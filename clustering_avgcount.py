#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def parse_args():
    p = argparse.ArgumentParser(description="Build LOSO cluster assignments for Model 4 (_avgcount flow).")
    p.add_argument(
        "--input",
        default="/home/runner/work/BikeCount/BikeCount/PreparedForModel_micro_nomicro.csv",
        help="Input CSV path",
    )
    p.add_argument(
        "--output",
        default="/home/runner/work/BikeCount/BikeCount/model4_outputs/clusters_avgcount.csv",
        help="Output cluster CSV path",
    )
    p.add_argument("--max-k", type=int, default=12, help="Maximum k to consider for silhouette selection")
    return p.parse_args()


MANUAL_DIRECTIONALITY_MAP = {
    "CB1101": "inbound",
    "CB1142": "inbound",
    "CB1143": "outbound",
    "CEK18": "inbound",
    "CEK049": "balanced",
    "CEE016": "balanced",
    "CB1599": "inbound",
    "CB1699": "outbound",
    "CB2105": "outbound",
    "CJM90": "inbound",
    "CEV011": "outbound",
    "CB02411": "balanced",
    "CJE181": "inbound",
    "CAT17": "outbound",
    "CLW239": "balanced",
    "COM205": "balanced",
    "CVT387": "inbound",
    "CEK31": "inbound",
}

DIR_TO_CODE = {"outbound": 0, "balanced": 1, "inbound": 2}


def choose_k_by_silhouette(x: np.ndarray, max_k: int) -> int:
    n = x.shape[0]
    if n <= 2:
        return 1
    k_upper = min(max_k, n - 1)
    if k_upper < 2:
        return 1
    best_k = 2
    best_s = -np.inf
    for k in range(2, k_upper + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(x)
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(x, labels, metric="euclidean")
        if s > best_s:
            best_s = s
            best_k = k
    return best_k


def station_aggregate(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.groupby("FeatureID", as_index=False)[feature_cols].mean(numeric_only=True)
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    required = {
        "FeatureID",
        "Build 125m",
        "Tree Proportion",
        "Plant Proportion",
        "Grass Proportion",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df["FeatureID"] = df["FeatureID"].astype(str)
    if "Directionality_encoded" not in df.columns:
        df["Directionality_encoded"] = 0.0
    df["Directionality_encoded"] = pd.to_numeric(df["Directionality_encoded"], errors="coerce").fillna(0.0)

    df["dir_label"] = (
        df["FeatureID"].str.upper().map(MANUAL_DIRECTIONALITY_MAP).fillna("balanced")
    )
    df["ManualFlowDirection_encoded"] = df["dir_label"].map(DIR_TO_CODE).astype(float)

    feature_cols = [
        "Build 125m",
        "Tree Proportion",
        "Plant Proportion",
        "Grass Proportion",
        "Directionality_encoded",
        "ManualFlowDirection_encoded",
    ]

    stations = sorted(df["FeatureID"].dropna().unique().tolist())
    rows = []

    for test_station in stations:
        train_df = df[df["FeatureID"] != test_station].copy()
        test_df = df[df["FeatureID"] == test_station].copy()
        if train_df.empty or test_df.empty:
            continue

        train_station = station_aggregate(train_df, feature_cols).fillna(0.0)
        x_cols = feature_cols
        x_train_raw = train_station[x_cols].astype(float).to_numpy()
        mu = np.nanmean(x_train_raw, axis=0)
        sd = np.nanstd(x_train_raw, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        x_train = (np.nan_to_num(x_train_raw, nan=0.0) - mu) / sd

        k_use = choose_k_by_silhouette(x_train, max_k=max(1, int(args.max_k)))
        km = KMeans(n_clusters=max(1, k_use), random_state=42, n_init=10)
        train_labels = km.fit_predict(x_train)
        train_station["cluster_id"] = train_labels.astype(int)

        # Held-out station assignment: no held-out count leakage.
        test_static = station_aggregate(test_df, feature_cols)
        if test_static.empty:
            continue
        x_test = test_static[x_cols].astype(float).to_numpy()
        x_test = (np.nan_to_num(x_test, nan=0.0) - mu) / sd
        test_label = int(km.predict(x_test)[0])

        for _, r in train_station[["FeatureID", "cluster_id"]].iterrows():
            rows.append(
                {
                    "Fold_Test_Station": str(test_station),
                    "FeatureID": str(r["FeatureID"]),
                    "cluster_id": int(r["cluster_id"]),
                    "is_test": 0,
                }
            )
        rows.append(
            {
                "Fold_Test_Station": str(test_station),
                "FeatureID": str(test_station),
                "cluster_id": int(test_label),
                "is_test": 1,
            }
        )

    if not rows:
        raise RuntimeError("No clusters generated.")

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"Wrote clusters to {out_path}")


if __name__ == "__main__":
    main()
