#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def parse_args():
    p = argparse.ArgumentParser(description="Run Model 4 only and export metrics/plots.")
    p.add_argument(
        "--input",
        default="/home/runner/work/BikeCount/BikeCount/PreparedForModel_micro_nomicro.csv",
        help="Input CSV path",
    )
    p.add_argument(
        "--output-dir",
        default="/home/runner/work/BikeCount/BikeCount/model4_outputs",
        help="Directory for outputs",
    )
    p.add_argument("--n-clusters", type=int, default=4, help="Number of teacher/feature clusters")
    p.add_argument("--max-stations", type=int, default=0, help="Limit number of test stations (0 = all)")
    return p.parse_args()


def safe_qcut(values: pd.Series, q: int) -> pd.Series:
    uq = values.nunique()
    bins = min(max(2, q), uq) if uq > 1 else 1
    if bins == 1:
        return pd.Series(np.zeros(len(values), dtype=int), index=values.index)
    return pd.qcut(values, q=bins, labels=False, duplicates="drop").astype(int)


def mode_or_default(s: pd.Series, default: int = 0) -> int:
    if s.empty:
        return default
    m = s.mode(dropna=True)
    if len(m) == 0:
        return default
    return int(m.iloc[0])


def build_station_static(df: pd.DataFrame) -> pd.DataFrame:
    static_cols = [
        "Latitude", "Longitude", "Tree Proportion", "Grass Proportion", "Plant Proportion", "NDVI",
        "Build 25m", "Build 125m", "Build 250m", "Veg 25m", "Veg 125m", "Veg 250m",
        "Street 25m", "Street 125m", "Street 250m", "RoadType_encoded", "Directionality_encoded", "Shade",
    ]
    existing = [c for c in static_cols if c in df.columns]
    station_static = df.groupby("FeatureID", as_index=False)[existing].mean(numeric_only=True)
    return station_static


def encode_feature_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out[c] = df[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            out[c] = df[c].astype("string").fillna("missing").astype("category").cat.codes.astype(float)
    return out


def estimate_station_mean(test_station: str, train_station_stats: pd.DataFrame, station_static: pd.DataFrame,
                          feature_cluster_of_test: int, feature_cluster_train: pd.DataFrame) -> float:
    test_row = station_static[station_static["FeatureID"] == test_station]
    if test_row.empty:
        return float(train_station_stats["station_mean"].mean())

    mapped = feature_cluster_train[feature_cluster_train["feature_cluster"] == feature_cluster_of_test]["FeatureID"].tolist()
    donors = train_station_stats[train_station_stats["FeatureID"].isin(mapped)].copy()
    if donors.empty:
        donors = train_station_stats.copy()

    donor_static = station_static[station_static["FeatureID"].isin(donors["FeatureID"])].copy()
    merged = donors.merge(donor_static, on="FeatureID", how="left")

    static_cols = [c for c in merged.columns if c not in {"FeatureID", "station_mean"} and pd.api.types.is_numeric_dtype(merged[c])]
    if not static_cols:
        return float(donors["station_mean"].mean())

    tv = test_row.iloc[0][static_cols].astype(float).values
    X = merged[static_cols].astype(float).fillna(0.0).values
    d = np.sqrt(np.sum((X - tv) ** 2, axis=1))
    w = np.exp(-d)
    if np.all(w <= 0):
        return float(donors["station_mean"].mean())
    return float(np.average(merged["station_mean"].values, weights=w))


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "date_parsed" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date_parsed"], errors="coerce")
    else:
        df["date_parsed"] = pd.to_datetime(df["DateCET"], format="%d-%m-%y", errors="coerce")
        if "hourCET" in df.columns:
            df["date_parsed"] = df["date_parsed"] + pd.to_timedelta(df["hourCET"].fillna(0).astype(int), unit="h")

    df = df.dropna(subset=["FeatureID", "Count", "date_parsed"]).copy()
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0.0)
    df = df[df["Count"] >= 0].copy()

    station_ids = sorted(df["FeatureID"].astype(str).unique().tolist())
    if args.max_stations and args.max_stations > 0:
        station_ids = station_ids[:args.max_stations]

    station_static = build_station_static(df)

    row_feature_cols = [
        "hourCET", "mth", "weekday", "weCET", "School_holiday", "weeknum", "year_",
        "Ta_nomicro", "RH_nomicro", "rain_nomicro", "hot_nomicro", "cold_nomicro", "windy_nomicro",
        "wind_speed_ucc", "solar_bxl",
        "Latitude", "Longitude", "Tree Proportion", "Grass Proportion", "Plant Proportion", "NDVI",
        "Build 25m", "Build 125m", "Build 250m", "Veg 25m", "Veg 125m", "Veg 250m",
        "Street 25m", "Street 125m", "Street 250m", "RoadType_encoded", "Directionality_encoded", "Shade",
        "Green", "Tempcode",
    ]

    all_preds = []

    for i, test_station in enumerate(station_ids, start=1):
        train_df = df[df["FeatureID"].astype(str) != test_station].copy()
        test_df = df[df["FeatureID"].astype(str) == test_station].copy()
        if train_df.empty or test_df.empty:
            continue

        station_mean_train = (
            train_df.groupby("FeatureID", as_index=False)["Count"].mean().rename(columns={"Count": "station_mean"})
        )

        train_df = train_df.merge(station_mean_train, on="FeatureID", how="left")
        train_df["profile_target"] = train_df["Count"] / train_df["station_mean"].replace(0, np.nan)
        train_df["profile_target"] = train_df["profile_target"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        train_df["profile_target"] = train_df["profile_target"].clip(lower=0.0, upper=train_df["profile_target"].quantile(0.995))

        station_teacher = station_mean_train.copy()
        station_teacher["teacher_cluster"] = safe_qcut(station_teacher["station_mean"], args.n_clusters)

        static_train = station_static[station_static["FeatureID"].isin(station_mean_train["FeatureID"])].copy()
        static_cols = [c for c in static_train.columns if c != "FeatureID"]
        Xs = static_train[static_cols].fillna(0.0).astype(float)

        n_clusters = min(args.n_clusters, len(static_train)) if len(static_train) > 0 else 1
        n_clusters = max(1, n_clusters)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        static_train["feature_cluster"] = km.fit_predict(Xs)

        cluster_map_df = static_train.merge(station_teacher[["FeatureID", "teacher_cluster"]], on="FeatureID", how="left")
        cluster_map = (
            cluster_map_df.groupby("feature_cluster")["teacher_cluster"].agg(lambda s: mode_or_default(s, 0)).to_dict()
        )

        static_test = station_static[station_static["FeatureID"] == test_station].copy()
        if static_test.empty:
            continue
        test_feature_cluster = int(km.predict(static_test[static_cols].fillna(0.0).astype(float))[0])
        _pred_teacher_cluster = int(cluster_map.get(test_feature_cluster, 0))

        X_train = encode_feature_frame(train_df, row_feature_cols)
        X_test = encode_feature_frame(test_df, row_feature_cols)

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=8,
            max_iter=250,
            min_samples_leaf=40,
            random_state=42,
        )
        model.fit(X_train.values, train_df["profile_target"].values)
        pred_profile = model.predict(X_test.values)
        pred_profile = np.clip(pred_profile, 0.0, None)

        est_station_mean = estimate_station_mean(
            test_station=test_station,
            train_station_stats=station_mean_train,
            station_static=station_static,
            feature_cluster_of_test=test_feature_cluster,
            feature_cluster_train=static_train[["FeatureID", "feature_cluster"]],
        )
        est_station_mean = max(est_station_mean, 1e-6)

        pred_count = pred_profile * est_station_mean
        fold_out = test_df[["FeatureID", "date_parsed", "Count"]].copy()
        fold_out["Pred_Count"] = pred_count
        fold_out["Fold_Test_Station"] = test_station
        fold_out["Pred_Feature_Cluster"] = test_feature_cluster
        fold_out["Pred_Teacher_Cluster"] = _pred_teacher_cluster
        all_preds.append(fold_out)

        print(f"[{i}/{len(station_ids)}] done {test_station} with {len(test_df)} rows")

    if not all_preds:
        raise RuntimeError("No predictions were generated.")

    pred_df = pd.concat(all_preds, axis=0, ignore_index=True)
    pred_df["Pred_Count"] = pred_df["Pred_Count"].clip(lower=0.0)

    y_true = pred_df["Count"].values
    y_pred = pred_df["Pred_Count"].values

    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    metrics = pd.DataFrame([{"Model": "Model4", "R2": r2, "RMSE": rmse, "Rows": len(pred_df), "Stations": pred_df["FeatureID"].nunique()}])
    metrics.to_csv(out_dir / "model4_metrics.csv", index=False)
    pred_df.sort_values(["date_parsed", "FeatureID"]).to_csv(out_dir / "model4_predictions_full.csv", index=False)

    import matplotlib.pyplot as plt

    daily = pred_df.copy()
    daily["date"] = daily["date_parsed"].dt.date
    daily_agg = daily.groupby("date", as_index=False)[["Count", "Pred_Count"]].sum()

    plt.figure(figsize=(14, 5))
    plt.plot(pd.to_datetime(daily_agg["date"]), daily_agg["Count"], label="Actual", linewidth=1.4)
    plt.plot(pd.to_datetime(daily_agg["date"]), daily_agg["Pred_Count"], label="Predicted", linewidth=1.2)
    plt.title(f"Model 4 - Full Year Daily Total (R2={r2:.4f}, RMSE={rmse:.4f})")
    plt.xlabel("Date")
    plt.ylabel("Daily total count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "model4_full_year_daily.png", dpi=180)
    plt.close()

    monthly = pred_df.copy()
    monthly["month"] = monthly["date_parsed"].dt.month
    monthly["date"] = monthly["date_parsed"].dt.date

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharey=False)
    axes = axes.flatten()
    for m in range(1, 13):
        ax = axes[m - 1]
        mdf = monthly[monthly["month"] == m]
        if mdf.empty:
            ax.set_title(f"Month {m} (no data)")
            ax.axis("off")
            continue
        mday = mdf.groupby("date", as_index=False)[["Count", "Pred_Count"]].sum()
        x = pd.to_datetime(mday["date"])
        ax.plot(x, mday["Count"], label="Actual", linewidth=1.0)
        ax.plot(x, mday["Pred_Count"], label="Pred", linewidth=1.0)
        ax.set_title(f"Month {m}")
        ax.tick_params(axis="x", labelrotation=45)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Model 4 - Monthly Daily Totals (12-Month Panels)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "model4_12month_panels.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=5, alpha=0.25)
    lim = max(np.max(y_true), np.max(y_pred))
    plt.plot([0, lim], [0, lim], "r--", linewidth=1)
    plt.xlabel("Actual Count")
    plt.ylabel("Predicted Count")
    plt.title(f"Model 4 Pred vs Actual (R2={r2:.4f}, RMSE={rmse:.4f})")
    plt.tight_layout()
    plt.savefig(out_dir / "model4_scatter.png", dpi=180)
    plt.close()

    print("\nModel 4 completed.")
    print(f"R2   : {r2:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
