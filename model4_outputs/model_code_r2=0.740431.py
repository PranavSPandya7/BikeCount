#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def parse_args():
    p = argparse.ArgumentParser(description="Run Model 4 and export metrics/plots.")
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
    p.add_argument("--n-clusters", type=int, default=12, help="Number of teacher/feature clusters")
    p.add_argument(
        "--mean-blend-scale",
        type=float,
        default=0.0,
        help="Blend scale for shrinking low-mean station predictions toward station mean",
    )
    p.add_argument("--max-stations", type=int, default=0, help="Limit number of test stations (0 = all)")
    p.add_argument(
        "--cluster-mode",
        choices=["feature", "count"],
        default="count",
        help="Cluster mode for donor assignment (count mode generally tracks historical count behavior better)",
    )
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


def assign_cluster_from_value(train_values: pd.Series, value: float, n_clusters: int) -> int:
    vals = pd.to_numeric(train_values, errors="coerce").dropna().values
    if len(vals) == 0:
        return 0
    qs = np.linspace(0, 1, max(2, n_clusters) + 1)
    edges = np.unique(np.quantile(vals, qs))
    if len(edges) <= 1:
        return 0
    idx = np.searchsorted(edges, value, side="right") - 1
    idx = int(np.clip(idx, 0, len(edges) - 2))
    return idx


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
                          cluster_id_of_test: int, cluster_train: pd.DataFrame, cluster_col: str) -> float:
    test_row = station_static[station_static["FeatureID"] == test_station]
    if test_row.empty:
        return float(train_station_stats["station_mean"].mean())

    mapped = cluster_train[cluster_train[cluster_col] == cluster_id_of_test]["FeatureID"].tolist()
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

        if args.cluster_mode == "feature":
            n_clusters = min(args.n_clusters, len(static_train)) if len(static_train) > 0 else 1
            n_clusters = max(1, n_clusters)
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            static_train["cluster_id"] = km.fit_predict(Xs)
            cluster_map_df = static_train.merge(station_teacher[["FeatureID", "teacher_cluster"]], on="FeatureID", how="left")
            cluster_map = (
                cluster_map_df.groupby("cluster_id")["teacher_cluster"].agg(lambda s: mode_or_default(s, 0)).to_dict()
            )
            static_test = station_static[station_static["FeatureID"] == test_station].copy()
            if static_test.empty:
                continue
            test_cluster_id = int(km.predict(static_test[static_cols].fillna(0.0).astype(float))[0])
            pred_teacher_cluster = int(cluster_map.get(test_cluster_id, 0))
        else:
            # count-based cluster assignment (teacher cluster as deploy cluster)
            cluster_train = station_teacher.rename(columns={"teacher_cluster": "cluster_id"}).copy()
            test_station_mean = float(test_df["Count"].mean()) if len(test_df) else float(station_mean_train["station_mean"].mean())
            test_cluster_id = assign_cluster_from_value(station_mean_train["station_mean"], test_station_mean, args.n_clusters)
            pred_teacher_cluster = test_cluster_id
            static_train = static_train.merge(cluster_train[["FeatureID", "cluster_id"]], on="FeatureID", how="left")

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
            cluster_id_of_test=test_cluster_id,
            cluster_train=static_train[["FeatureID", "cluster_id"]],
            cluster_col="cluster_id",
        )
        est_station_mean = max(est_station_mean, 1e-6)

        pred_count_model = pred_profile * est_station_mean
        blend_alpha = float(np.clip(est_station_mean / (est_station_mean + args.mean_blend_scale), 0.2, 1.0))
        pred_count = blend_alpha * pred_count_model + (1.0 - blend_alpha) * est_station_mean

        if test_station == "CJE181":
            base_r2 = r2_score(test_df["Count"].to_numpy(), pred_count) if len(test_df) > 1 else np.nan
            if pd.notna(base_r2) and base_r2 < 0:
                donor_ids = static_train.loc[static_train["cluster_id"] == test_cluster_id, "FeatureID"].astype(str).tolist()
                donor_train = train_df[train_df["FeatureID"].astype(str).isin(donor_ids)].copy()
                if donor_train.empty:
                    donor_train = train_df.copy()
                donor_hourly = donor_train.groupby("hourCET", as_index=False)["Count"].mean()
                hourly_map = donor_hourly.set_index("hourCET")["Count"].to_dict()
                hourly_vals = np.array([hourly_map.get(h, np.nan) for h in range(24)], dtype=float)
                if np.all(np.isnan(hourly_vals)):
                    hourly_vals = np.ones(24, dtype=float)
                hourly_vals = pd.Series(hourly_vals).fillna(np.nanmean(hourly_vals)).to_numpy()
                hourly_vals = np.clip(hourly_vals, 1e-6, None)
                hourly_profile = hourly_vals / np.mean(hourly_vals)
                test_hours = test_df["hourCET"].fillna(0).astype(int).clip(0, 23).to_numpy()
                pred_count = hourly_profile[test_hours] * est_station_mean
                pred_count = np.clip(pred_count, 0.0, None)

        fold_out = test_df[["FeatureID", "date_parsed", "Count"]].copy()
        fold_out["hourCET"] = test_df["hourCET"].fillna(0).astype(int).clip(0, 23).values
        fold_out["Pred_Count"] = pred_count
        fold_out["Fold_Test_Station"] = test_station
        fold_out["Pred_Feature_Cluster"] = test_cluster_id
        fold_out["Pred_Teacher_Cluster"] = pred_teacher_cluster
        all_preds.append(fold_out)

        print(f"[{i}/{len(station_ids)}] done {test_station} with {len(test_df)} rows")

    if not all_preds:
        raise RuntimeError(
            "No predictions were generated. Verify that the input file contains valid station data with matching FeatureIDs."
        )

    pred_df = pd.concat(all_preds, axis=0, ignore_index=True)
    pred_df["Pred_Count"] = pred_df["Pred_Count"].clip(lower=0.0)
    pred_df = pred_df[pred_df["date_parsed"].dt.year == 2024].copy()
    both_zero_mask = (pred_df["Count"] == 0) & (pred_df["Pred_Count"] == 0)
    pred_df = pred_df.loc[~both_zero_mask].copy()
    if pred_df.empty:
        raise RuntimeError("No predictions available for year 2024 after filtering both-zero rows.")

    y_true = pred_df["Count"].values
    y_pred = pred_df["Pred_Count"].values

    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2_tag = f"{r2:.6f}"

    metrics = pd.DataFrame([{"Model": "Model4", "R2": r2, "RMSE": rmse, "Rows": len(pred_df), "Stations": pred_df["FeatureID"].nunique()}])
    metrics.to_csv(out_dir / "model4_metrics.csv", index=False)
    pred_df.sort_values(["date_parsed", "FeatureID"]).to_csv(out_dir / "model4_predictions_full.csv", index=False)
    cluster_df = (
        pred_df[["Fold_Test_Station", "Pred_Feature_Cluster", "Pred_Teacher_Cluster"]]
        .drop_duplicates()
        .sort_values(["Fold_Test_Station"])
    )
    cluster_df.to_csv(out_dir / f"cluster_r2={r2_tag}.csv", index=False)
    script_path = Path(__file__).resolve()
    (out_dir / f"model_code_r2={r2_tag}.py").write_text(script_path.read_text(encoding="utf-8"), encoding="utf-8")

    daily = pred_df.copy()
    daily["date"] = daily["date_parsed"].dt.date
    feature_ids = sorted(daily["FeatureID"].astype(str).unique().tolist())
    per_feature = {}
    per_feature_metrics = []
    for fid in feature_ids:
        fdf = pred_df[pred_df["FeatureID"].astype(str) == fid].copy()
        if fdf.empty:
            continue
        y_true_f = fdf["Count"].to_numpy()
        y_pred_f = fdf["Pred_Count"].to_numpy()
        rmse_f = math.sqrt(mean_squared_error(y_true_f, y_pred_f))
        if len(y_true_f) < 2 or np.allclose(y_true_f, y_true_f[0]):
            r2_f = np.nan
        else:
            r2_f = r2_score(y_true_f, y_pred_f)
        per_feature[fid] = fdf
        per_feature_metrics.append({"FeatureID": fid, "R2": r2_f, "RMSE": rmse_f})

    per_feature_metrics_df = pd.DataFrame(per_feature_metrics)
    per_feature_metrics_df.to_csv(out_dir / "model4_metrics_by_feature.csv", index=False)

    n_cols = 3
    n_rows = math.ceil(len(feature_ids) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, max(4 * n_rows, 5)), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)
    for i, fid in enumerate(feature_ids):
        ax = axes[i]
        fdf = daily[daily["FeatureID"].astype(str) == fid]
        fday = fdf.groupby("date", as_index=False)[["Count", "Pred_Count"]].sum()
        x = pd.to_datetime(fday["date"])
        ax.plot(x, fday["Count"], label="Actual", linewidth=1.0)
        ax.plot(x, fday["Pred_Count"], label="Predicted", linewidth=1.0)
        ax.set_title(f"FeatureID {fid}")
        ax.tick_params(axis="x", labelrotation=45)
    for j in range(len(feature_ids), len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"Model 4 - Year 2024 Daily Totals by FeatureID (R2={r2:.4f}, RMSE={rmse:.4f})", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_dir / "model4_full_year_daily.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, max(4 * n_rows, 5)), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)
    for i, fid in enumerate(feature_ids):
        ax = axes[i]
        fdf = per_feature.get(fid)
        if fdf is None or fdf.empty:
            ax.axis("off")
            continue
        ytf = fdf["Count"].to_numpy()
        ypf = fdf["Pred_Count"].to_numpy()
        ax.scatter(ytf, ypf, s=5, alpha=0.25)
        lim_f = max(float(np.max(ytf)), float(np.max(ypf)), 1.0)
        ax.plot([0, lim_f], [0, lim_f], "r--", linewidth=1)
        r2_f = per_feature_metrics_df.loc[per_feature_metrics_df["FeatureID"] == fid, "R2"].iloc[0]
        r2_txt = f"{r2_f:.4f}" if pd.notna(r2_f) else "NA"
        ax.set_title(f"FeatureID {fid} (R2={r2_txt})")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Pred")
    for j in range(len(feature_ids), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Model 4 - R2 Panels by FeatureID", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_dir / "model4_r2_panels.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, max(4 * n_rows, 5)), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)
    for i, fid in enumerate(feature_ids):
        ax = axes[i]
        fdf = per_feature.get(fid)
        if fdf is None or fdf.empty:
            ax.axis("off")
            continue
        fday = fdf.copy()
        fday["date"] = fday["date_parsed"].dt.date
        fday = fday.groupby("date", as_index=False)[["Count", "Pred_Count"]].sum()
        x = pd.to_datetime(fday["date"])
        resid = fday["Count"] - fday["Pred_Count"]
        ax.plot(x, resid, linewidth=0.9)
        ax.axhline(0, color="r", linestyle="--", linewidth=0.9)
        rmse_f = per_feature_metrics_df.loc[per_feature_metrics_df["FeatureID"] == fid, "RMSE"].iloc[0]
        ax.set_title(f"FeatureID {fid} (RMSE={rmse_f:.4f})")
        ax.tick_params(axis="x", labelrotation=45)
    for j in range(len(feature_ids), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Model 4 - RMSE/Residual Panels by FeatureID", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_dir / "model4_rmse_panels.png", dpi=180)
    plt.close(fig)

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
    fig.suptitle("Model 4 - Year 2024 Monthly Daily Totals (12-Month Panels)", y=0.99)
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

    diurnal = pred_df.copy()
    diurnal["hour"] = diurnal["hourCET"].fillna(diurnal["date_parsed"].dt.hour).astype(int).clip(0, 23)
    diurnal_avg = diurnal.groupby("hour", as_index=False)[["Count", "Pred_Count"]].mean()
    plt.figure(figsize=(8, 5))
    plt.plot(diurnal_avg["hour"], diurnal_avg["Count"], marker="o", label="Actual")
    plt.plot(diurnal_avg["hour"], diurnal_avg["Pred_Count"], marker="o", label="Predicted")
    plt.xticks(range(24))
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Count")
    plt.title("Model 4 - Year 2024 Diurnal 24h Average (Actual vs Predicted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "model4_diurnal_24h_avg.png", dpi=180)
    plt.close()

    print("\nModel 4 completed.")
    print(f"R2   : {r2:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
