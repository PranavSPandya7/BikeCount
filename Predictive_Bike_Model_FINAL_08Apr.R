library(data.table)
library(lubridate)
library(randomForest)

# 1. SETUP & DATA LOADING
setwd("C:/Users/pandya/OneDrive - UCL/Data shared/Response 30Mar")
MOD_DIR <- "Model_final"; PLOT_DIR <- "Plots_final"
MODEL_DIRS <- c("1_LR", "2_LR_micro", "3_LR_nomicro",
                "4_RF_noweather_24h_VP",
                "5_RF_micro_24h_VP", "6_RF_nomicro_24h_VP")
model_subdirs <- c("RDS", "fit", "R2_plots", "RMSE_plots")
all_model_paths <- as.vector(outer(file.path(MOD_DIR, MODEL_DIRS), model_subdirs, file.path))
shared_paths <- file.path(MOD_DIR, c("Metrics", "12month_plot", "Full_year_plot"))
invisible(lapply(c(MOD_DIR, PLOT_DIR, all_model_paths, shared_paths,
                   file.path(PLOT_DIR, "Model_Fits")),
                 dir.create, showWarnings = FALSE, recursive = TRUE))

bike_dt <- fread("PreparedForModel_micro_nomicro.csv")

# HOLIDAY INTEGRATION
holidays_requested <- c("IsaWeekHoliday", "SummerHoliday", "WinterHoliday", "EasterHoliday", "AllSaintsHolidays",
                        "SummerHolidayProf", "IsaSatHoliday", "WinterHolidayProf", "SpringHoliday", "CrocusHoliday", "IsaHoliday", "IsaSunHoliday", "IsaSummerHoliday", "IsaWinterHoliday")
existing_holidays <- intersect(holidays_requested, names(bike_dt))
if(length(existing_holidays) > 0) {
  bike_dt[, IsHoliday := as.numeric(rowSums(.SD, na.rm = TRUE) > 0), .SDcols = existing_holidays]
} else {
  bike_dt[, IsHoliday := 0]
}

bike_dt[, UTC1 := as.POSIXct(UTC1, format='%d-%m-%y %H:%M', tz='UTC')]
bike_dt[, DateCET := as.Date(UTC1 - 3600, tz='CET')]
bike_dt[, `:=`(month = month(DateCET), year = year(DateCET), fmonth = factor(month(DateCET), 1:12),
               fweekday = factor(as.numeric(strftime(DateCET, '%u')) - 1, 0:6),
               Trend = as.numeric(as.Date(DateCET) - as.Date('2022-01-01')))]

# STATION_CLUSTERS <- list(
#   "1" = c("CJE181", "CVT387", "CEV011", "CLW239"),
#   "2" = c("CB1699", "CB1599", "COM205"),
#   "3" = c("CEK18", "CEE016", "CAT17", "CEK31"),
#   "4" = c("CB2105", "CEK049", "CB1143"),
#   "5" = c("CB1142", "CB02411", "CB1101", "CJM90"))

STATION_CLUSTERS <- list(
  "1" = c("CJE181", "CVT387", "CLW239"),
  "2" = c("CEV011", "CB1699", "CB1599"),
  "3" = c("COM205", "CEK18", "CEE016"),
  "4" = c("CAT17", "CEK31", "CB2105"),
  "5" = c("CEK049", "CB1143", "CB02411"),
  "6" = c("CB1142", "CB1101", "CJM90")
)

cluster_map <- rbindlist(lapply(names(STATION_CLUSTERS), function(c) data.table(FeatureID = STATION_CLUSTERS[[c]], Cluster = as.integer(c))))
bike_dt[, Cluster := NULL]
bike_dt <- merge(bike_dt, cluster_map, by = "FeatureID", all.x = TRUE)

# 2. PREPROCESSING
setnames(bike_dt, c("Build 125m", "Tree Proportion", "Grass Proportion", "Plant Proportion", "RoadType_encoded"),
         c("Build125m", "Tree_Pct", "Grass_Pct", "Plant_Pct", "RoadType_encoded"), skip_absent = TRUE)

# GVI (Green View Index) = sum of tree, grass, plant pixel proportions × 100 (%)
bike_dt[, GVI := (Tree_Pct + Grass_Pct + Plant_Pct) * 100]

bike_dt[, Volume_Profile := Count / (mean(Count, na.rm = TRUE) + 0.1), by = FeatureID]

FEATURES <- c("fmonth", "Ta", "RH", "windy", "Tmrt", "UTCI", "rain", "Shade",
              "RoadType_encoded", "Directionality_encoded", "Build125m", "GVI", "hot", "cold", "weCET", "IsHoliday")
for(v in setdiff(FEATURES, "fmonth")) bike_dt[, (v) := suppressWarnings(as.numeric(as.character(get(v))))][is.na(get(v)), (v) := 0]

nomicro_cols <- c("Ta_nomicro", "Tmrt_nomicro", "UTCI_nomicro", "RH_nomicro", "rain_nomicro", "hot_nomicro", "cold_nomicro", "windy_nomicro")
for(v in nomicro_cols) {
  if(!v %in% names(bike_dt)) stop(paste("Column", v, "not found in data!"))
  bike_dt[, (v) := suppressWarnings(as.numeric(as.character(get(v))))][is.na(get(v)), (v) := 0]
}

lr_holiday_cols <- c("IsaWeekHoliday", "SummerHoliday", "IsaSatHoliday", "HeureHiver", "WinterHoliday", "EasterHoliday", "AllSaintsHolidays")
for(hcol in lr_holiday_cols) {
  if(!hcol %in% names(bike_dt)) bike_dt[, (hcol) := 0]
  bike_dt[, (hcol) := suppressWarnings(as.numeric(as.character(get(hcol))))][is.na(get(hcol)), (hcol) := 0]
}

# 3. LINEAR REGRESSION MODELS (per-station, per-hour)

# Model 1: LR (no weather)
cat("Training Model 1: LR (no weather)...\n")
bike_dt[, Prediction_LR := {
  formula_LR <- as.formula("log(Count + 1) ~ Trend + fmonth + weCET + IsaWeekHoliday + SummerHoliday + IsaSatHoliday + HeureHiver + WinterHoliday + EasterHoliday + AllSaintsHolidays")
  fit_val <- tryCatch(
    pmax(0, exp(predict(lm(formula_LR, data = .SD, subset = Count > 0), newdata = .SD)) - 1),
    error = function(e) rep(0, nrow(.SD)))
  for(i in 1:5) {
    outliers <- log(Count + 1) - log(fit_val + 1)
    sub_idx  <- Count > 0 & abs(outliers) <= 1
    wts      <- pmax(1e-6, log(fit_val + 1))
    fit_val  <- tryCatch(
      pmax(0, exp(predict(lm(formula_LR, data = .SD, subset = sub_idx, weights = wts), newdata = .SD)) - 1),
      error = function(e) fit_val)
  }
  saveRDS(.SD[, .(UTC1, FeatureID, fit = fit_val)], file.path(MOD_DIR, "1_LR/RDS", paste0(FeatureID[1], "_LR.rds")))
  fit_val
}, by = .(FeatureID, hourCET)]

# Model 2: LR micro
cat("Training Model 2: LR micro...\n")
bike_dt[, Prediction_LR_micro := {
  formula_LR_micro <- as.formula("log(Count + 1) ~ Trend + fmonth + weCET + rain + Shade + windy + cold + hot + IsaWeekHoliday + SummerHoliday + IsaSatHoliday + HeureHiver + WinterHoliday + EasterHoliday + AllSaintsHolidays")
  fit_val <- tryCatch(
    pmax(0, exp(predict(lm(formula_LR_micro, data = .SD, subset = Count > 0), newdata = .SD)) - 1),
    error = function(e) rep(0, nrow(.SD)))
  for(i in 1:5) {
    outliers <- log(Count + 1) - log(fit_val + 1)
    sub_idx  <- Count > 0 & abs(outliers) <= 1
    wts      <- pmax(1e-6, log(fit_val + 1))
    fit_val  <- tryCatch(
      pmax(0, exp(predict(lm(formula_LR_micro, data = .SD, subset = sub_idx, weights = wts), newdata = .SD)) - 1),
      error = function(e) fit_val)
  }
  saveRDS(.SD[, .(UTC1, FeatureID, fit = fit_val)], file.path(MOD_DIR, "2_LR_micro/RDS", paste0(FeatureID[1], "_LR_micro.rds")))
  fit_val
}, by = .(FeatureID, hourCET)]

# Model 3: LR nomicro
cat("Training Model 3: LR nomicro...\n")
bike_dt[, Prediction_LR_nomicro := {
  formula_LR_nomicro <- as.formula("log(Count + 1) ~ Trend + fmonth + weCET + rain_nomicro + Shade + windy_nomicro + cold_nomicro + hot_nomicro + IsaWeekHoliday + SummerHoliday + IsaSatHoliday + HeureHiver + WinterHoliday + EasterHoliday + AllSaintsHolidays")
  fit_val <- tryCatch(
    pmax(0, exp(predict(lm(formula_LR_nomicro, data = .SD, subset = Count > 0), newdata = .SD)) - 1),
    error = function(e) rep(0, nrow(.SD)))
  for(i in 1:5) {
    outliers <- log(Count + 1) - log(fit_val + 1)
    sub_idx  <- Count > 0 & abs(outliers) <= 1
    wts      <- pmax(1e-6, log(fit_val + 1))
    fit_val  <- tryCatch(
      pmax(0, exp(predict(lm(formula_LR_nomicro, data = .SD, subset = sub_idx, weights = wts), newdata = .SD)) - 1),
      error = function(e) fit_val)
  }
  saveRDS(.SD[, .(UTC1, FeatureID, fit = fit_val)], file.path(MOD_DIR, "3_LR_nomicro/RDS", paste0(FeatureID[1], "_LR_nomicro.rds")))
  fit_val
}, by = .(FeatureID, hourCET)]

# Data Imputation for gaps >= 6 hours
bike_dt[, gap_id := rleid(Count == 0), by = FeatureID]
bike_dt[, Count := as.numeric(Count)]
gap_meta <- bike_dt[, .N, by = .(FeatureID, gap_id)][N >= 6, paste0(FeatureID, "_", gap_id)]
bike_dt[paste0(FeatureID, "_", gap_id) %in% gap_meta & Count == 0, Count := Prediction_LR]

# Recompute Volume_Profile after imputation
bike_dt[, Volume_Profile := Count / (mean(Count, na.rm = TRUE) + 0.1), by = FeatureID]

# ═══════════════════════════════════════════════════════════════════════════════
# 4. RANDOM FOREST TRAINING (LOO by station)
# ═══════════════════════════════════════════════════════════════════════════════

# Model 4: RF no-weather (temporal + land-use only)
formula_rf_noweather_VP <- as.formula("log(Volume_Profile + 0.01) ~ fmonth +  IsaWeekHoliday + SummerHoliday + IsaSatHoliday + HeureHiver + WinterHoliday + EasterHoliday + AllSaintsHolidays + weCET + Shade + RoadType_encoded + Directionality_encoded + Build125m + GVI")

# Model 5: RF micro VP
formula_rf_micro_VP  <- as.formula("log(Volume_Profile + 0.01) ~ fmonth + IsHoliday + weCET + Ta + RH + windy + Tmrt + UTCI + rain + Shade + RoadType_encoded + Directionality_encoded + Build125m + GVI + hot + cold")

# Model 6: RF nomicro VP
formula_rf_nomicro_VP <- as.formula("log(Volume_Profile + 0.01) ~ fmonth + IsHoliday + weCET + Ta_nomicro + RH_nomicro + windy_nomicro + Tmrt_nomicro + UTCI_nomicro + rain_nomicro + Shade + RoadType_encoded + Directionality_encoded + Build125m + GVI + hot_nomicro + cold_nomicro")

unique_ids <- unique(bike_dt$FeatureID)
for(i in seq_along(unique_ids)) {
  thisid <- unique_ids[i]
  cat(sprintf("[%d/%d] Training RF for station: %s\n", i, length(unique_ids), thisid))

  cid        <- bike_dt[FeatureID == thisid, Cluster[1]]
  cluster_ids <- STATION_CLUSTERS[[as.character(cid)]]
  train_dt   <- bike_dt[FeatureID %in% cluster_ids & FeatureID != thisid]

  # No-weather VP (Model 4)
  models_vp_noweather <- lapply(0:23, function(h) {
    sub_dt <- train_dt[hourCET == h & Volume_Profile > 0]
    if(nrow(sub_dt) > 2) {
      if(nrow(sub_dt) > 5000) sub_dt <- sub_dt[sample(.N, 5000)]
      tryCatch(randomForest(formula_rf_noweather_VP, data = sub_dt, ntree = 30, mtry = 5, nodesize = 5), error = function(e) NULL)
    } else NULL
  })
  names(models_vp_noweather) <- 0:23
  saveRDS(models_vp_noweather, file.path(MOD_DIR, "4_RF_noweather_24h_VP/RDS", paste0(thisid, "_RF_VP_noweather.rds")))

  # Micro VP (Model 5)
  models_vp_micro <- lapply(0:23, function(h) {
    sub_dt <- train_dt[hourCET == h & Volume_Profile > 0]
    if(nrow(sub_dt) > 2) {
      if(nrow(sub_dt) > 5000) sub_dt <- sub_dt[sample(.N, 5000)]
      tryCatch(randomForest(formula_rf_micro_VP, data = sub_dt, ntree = 30, mtry = 5, nodesize = 5), error = function(e) NULL)
    } else NULL
  })
  names(models_vp_micro) <- 0:23
  saveRDS(models_vp_micro, file.path(MOD_DIR, "5_RF_micro_24h_VP/RDS", paste0(thisid, "_RF_VP_micro.rds")))

  # Nomicro VP (Model 6)
  models_vp_nomicro <- lapply(0:23, function(h) {
    sub_dt <- train_dt[hourCET == h & Volume_Profile > 0]
    if(nrow(sub_dt) > 2) {
      if(nrow(sub_dt) > 5000) sub_dt <- sub_dt[sample(.N, 5000)]
      tryCatch(randomForest(formula_rf_nomicro_VP, data = sub_dt, ntree = 30, mtry = 5, nodesize = 5), error = function(e) NULL)
    } else NULL
  })
  names(models_vp_nomicro) <- 0:23
  saveRDS(models_vp_nomicro, file.path(MOD_DIR, "6_RF_nomicro_24h_VP/RDS", paste0(thisid, "_RF_VP_nomicro.rds")))

  cat(" - RF models done for:", thisid, "\n")
}

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PREDICTION — 3 RF columns
# ═══════════════════════════════════════════════════════════════════════════════
cat("\nGenerating RF predictions...\n")
bike_dt[, c("RF_noweather_24h_VP",
            "RF_micro_24h_VP", "RF_nomicro_24h_VP") := {

  vp_nw <- readRDS(file.path(MOD_DIR, "4_RF_noweather_24h_VP/RDS", paste0(FeatureID[1], "_RF_VP_noweather.rds")))
  vp_mi <- readRDS(file.path(MOD_DIR, "5_RF_micro_24h_VP/RDS",  paste0(FeatureID[1], "_RF_VP_micro.rds")))
  vp_no <- readRDS(file.path(MOD_DIR, "6_RF_nomicro_24h_VP/RDS", paste0(FeatureID[1], "_RF_VP_nomicro.rds")))

  cid <- Cluster[1]
  # 3-year cluster average (Models 4/5/6)
  cluster_avg_volume <- bike_dt[Cluster == cid & FeatureID != FeatureID[1], mean(Count, na.rm = TRUE)]

  p_vp_nw <- p_vp_mi <- p_vp_no <- numeric(.N)

  for(h in 0:23) {
    idx <- hourCET == h
    if(any(idx)) {
      h_chr <- as.character(h)
      if(!is.null(vp_nw[[h_chr]])) {
        shape_nw <- pmax(0, exp(predict(vp_nw[[h_chr]], .SD[idx])) - 0.01)
        p_vp_nw[idx] <- shape_nw * cluster_avg_volume
      }
      if(!is.null(vp_mi[[h_chr]])) {
        shape_mi <- pmax(0, exp(predict(vp_mi[[h_chr]], .SD[idx])) - 0.01)
        p_vp_mi[idx] <- shape_mi * cluster_avg_volume
      }
      if(!is.null(vp_no[[h_chr]])) {
        shape_no <- pmax(0, exp(predict(vp_no[[h_chr]], .SD[idx])) - 0.01)
        p_vp_no[idx] <- shape_no * cluster_avg_volume
      }
    }
  }
  list(p_vp_nw, p_vp_mi, p_vp_no)
}, by = FeatureID]

# Save
bike_dt[, gap_id := NULL]
fwrite(bike_dt, file.path(MOD_DIR, "bikedata_filled.csv"))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. METRICS (all on log scale)
# ═══════════════════════════════════════════════════════════════════════════════
calc_stats <- function(act, prd, hourCET = NULL, time_filter = NULL, exclude_na = TRUE, exclude_zero = TRUE) {
  idx <- rep(TRUE, length(act))
  if(!is.null(time_filter) && !is.null(hourCET)) idx <- idx & hourCET %in% time_filter
  if(exclude_na) idx <- idx & !is.na(act) & !is.na(prd)
  if(exclude_zero) idx <- idx & act > 0 & prd > 0
  if(sum(idx) < 2) return(c(NA, NA))
  a <- act[idx]; p <- prd[idx]
  r2       <- round(1 - sum((a - p)^2) / sum((a - mean(a))^2), 3)
  rmse_pct <- round((exp(sqrt(mean((a - p)^2))) - 1) * 100, 1)
  c(r2, rmse_pct)
}

metrics <- bike_dt[Count > 0 & Prediction_LR > 0 & Prediction_LR_micro > 0, {
  obs_log <- log(Count + 1)
  m_lr       <- calc_stats(obs_log, log(pmax(0.1, Prediction_LR)          + 1), hourCET = hourCET, exclude_zero = FALSE)
  m_lr_mi    <- calc_stats(obs_log, log(pmax(0.1, Prediction_LR_micro)    + 1), hourCET = hourCET, exclude_zero = FALSE)
  m_lr_no    <- calc_stats(obs_log, log(pmax(0.1, Prediction_LR_nomicro)  + 1), hourCET = hourCET, exclude_zero = FALSE)
  rf_log     <- function(col) log(pmax(0.1, get(col)) + 1)
  m4  <- calc_stats(obs_log, rf_log("RF_noweather_24h_VP"),          hourCET = hourCET, exclude_zero = FALSE)
  m5  <- calc_stats(obs_log, rf_log("RF_micro_24h_VP"),              hourCET = hourCET, exclude_zero = FALSE)
  m6  <- calc_stats(obs_log, rf_log("RF_nomicro_24h_VP"),            hourCET = hourCET, exclude_zero = FALSE)
  .(R2_LR = m_lr[1], R2_LR_micro = m_lr_mi[1], R2_LR_nomicro = m_lr_no[1],
    R2_RF_noweather_24h_VP = m4[1],
    R2_RF_micro_24h_VP = m5[1], R2_RF_nomicro_24h_VP = m6[1],
    RMSE_pct_LR = m_lr[2], RMSE_pct_LR_micro = m_lr_mi[2], RMSE_pct_LR_nomicro = m_lr_no[2],
    RMSE_pct_RF_noweather_24h_VP = m4[2],
    RMSE_pct_RF_micro_24h_VP = m5[2], RMSE_pct_RF_nomicro_24h_VP = m6[2],
    Mean_Count = round(mean(Count, na.rm=TRUE), 1), SD_Count = round(sd(Count, na.rm=TRUE), 1),
    Cluster = Cluster[1])
}, by = FeatureID]

metric_cols <- c("R2_LR","R2_LR_micro","R2_LR_nomicro",
  "R2_RF_noweather_24h_VP",
  "R2_RF_micro_24h_VP","R2_RF_nomicro_24h_VP",
  "RMSE_pct_LR","RMSE_pct_LR_micro","RMSE_pct_LR_nomicro",
  "RMSE_pct_RF_noweather_24h_VP",
  "RMSE_pct_RF_micro_24h_VP","RMSE_pct_RF_nomicro_24h_VP")
overall <- metrics[, lapply(.SD, mean, na.rm=TRUE), .SDcols = c(metric_cols,"Mean_Count","SD_Count")][, `:=`(FeatureID="OVERALL_mean", Cluster=NA)]
fwrite(rbind(metrics[order(FeatureID)], overall, use.names=TRUE, fill=TRUE), file.path(MOD_DIR, "Metrics/Performance_Metrics.csv"))

# Per-hour metrics
cat("Computing per-hour metrics...\n")
model_pred_cols <- list(
  LR          = "Prediction_LR",
  LR_micro    = "Prediction_LR_micro",
  LR_nomicro  = "Prediction_LR_nomicro",
  RF_noweather_24h_VP    = "RF_noweather_24h_VP",
  RF_micro_24h_VP        = "RF_micro_24h_VP",
  RF_nomicro_24h_VP      = "RF_nomicro_24h_VP"
)
# --- Per-station per-hour metrics (avoids mixing between/within-station variance) ---
# R2 computed within each station at each hour, then averaged across stations.
# This is the honest metric: baseline = that station's own mean at hour h.
perhour_by_station <- rbindlist(lapply(0:23, function(h) {
  rbindlist(lapply(unique_ids, function(sid) {
    sub <- bike_dt[hourCET == h & FeatureID == sid & Count > 0]
    if(nrow(sub) < 10) return(NULL)
    row <- data.table(hourCET = h, FeatureID = sid)
    for(label in names(model_pred_cols)) {
      col <- model_pred_cols[[label]]
      if(!col %in% names(sub)) { row[, (paste0(label,"_R2")) := NA_real_]; row[, (paste0(label,"_RMSE")) := NA_real_]; next }
      ok  <- !is.na(sub[[col]]) & sub[[col]] > 0
      if(sum(ok) < 3) { row[, (paste0(label,"_R2")) := NA_real_]; row[, (paste0(label,"_RMSE")) := NA_real_]; next }
      a <- log(sub$Count[ok] + 1)
      p <- log(sub[[col]][ok] + 1)
      baseline <- mean(a)  # this station's own mean at this hour
      ss_res   <- sum((a - p)^2)
      ss_tot   <- sum((a - baseline)^2)
      r2       <- round(if(ss_tot > 0) 1 - ss_res / ss_tot else NA_real_, 3)
      rmse_pct <- round((exp(sqrt(mean((a - p)^2))) - 1) * 100, 1)
      row[, (paste0(label,"_R2"))   := r2]
      row[, (paste0(label,"_RMSE")) := rmse_pct]
    }
    row
  }))
}))
fwrite(perhour_by_station, file.path(MOD_DIR, "Metrics/Performance_Metrics_PerHour.csv"))

# Console output
cat("\n=== MODEL COMPARISON (mean R2 / RMSE%, log scale) ===\n")
cat(sprintf("Model 1   LR (no weather):                  R2 = %.3f,  RMSE%% = %.1f\n", overall$R2_LR, overall$RMSE_pct_LR))
cat(sprintf("Model 2   LR (micro weather):              R2 = %.3f,  RMSE%% = %.1f\n", overall$R2_LR_micro, overall$RMSE_pct_LR_micro))
cat(sprintf("Model 3   LR (nomicro weather):            R2 = %.3f,  RMSE%% = %.1f\n", overall$R2_LR_nomicro, overall$RMSE_pct_LR_nomicro))
cat(sprintf("Model 4   RF no-weather 24h VP (3yr avg):  R2 = %.3f,  RMSE%% = %.1f\n", overall$R2_RF_noweather_24h_VP, overall$RMSE_pct_RF_noweather_24h_VP))
cat(sprintf("Model 5   RF micro  24h VP (3yr avg):      R2 = %.3f,  RMSE%% = %.1f\n", overall$R2_RF_micro_24h_VP, overall$RMSE_pct_RF_micro_24h_VP))
cat(sprintf("Model 6   RF nomicro 24h VP (3yr avg):     R2 = %.3f,  RMSE%% = %.1f\n", overall$R2_RF_nomicro_24h_VP, overall$RMSE_pct_RF_nomicro_24h_VP))
cat("=====================================================\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
cat("Generating Visualizations...\n")

# Daily fit plots
fit_configs <- list(
  list(col = "Prediction_LR",           dir = "1_LR/fit",                           suffix = "LR",           title = "M1 LR (no weather)",      lcol = 4),
  list(col = "Prediction_LR_micro",     dir = "2_LR_micro/fit",                     suffix = "LR_micro",     title = "M2 LR (micro)",           lcol = 3),
  list(col = "Prediction_LR_nomicro",   dir = "3_LR_nomicro/fit",                   suffix = "LR_nomicro",   title = "M3 LR (nomicro)",         lcol = "dodgerblue"),
  list(col = "RF_noweather_24h_VP",     dir = "4_RF_noweather_24h_VP/fit",          suffix = "RF_noweather", title = "M4 RF no-weather VP",     lcol = "brown"),
  list(col = "RF_micro_24h_VP",         dir = "5_RF_micro_24h_VP/fit",              suffix = "RF_micro_VP",  title = "M5 RF micro VP",          lcol = 2),
  list(col = "RF_nomicro_24h_VP",       dir = "6_RF_nomicro_24h_VP/fit",            suffix = "RF_nomicro_VP", title = "M6 RF nomicro VP",       lcol = "coral")
)

for(id in unique_ids) {
  sub <- bike_dt[FeatureID == id]
  if(nrow(sub) == 0) next
  for(cfg in fit_configs) {
    tryCatch({
      agg <- sub[, .(act = sum(Count, na.rm = TRUE), prd = sum(get(cfg$col), na.rm = TRUE)), by = DateCET][order(DateCET)]
      svg(file.path(MOD_DIR, cfg$dir, paste0(id, "_", cfg$suffix, "_fit.svg")), width = 10, height = 6)
      par(mfrow = c(2,1), mar = c(3,3,2,1))
      plot(agg$DateCET, agg$act, type = "l", main = paste(id, cfg$title, "Fit"), ylab = "Count", xlab = "Date")
      lines(agg$DateCET, agg$prd, col = cfg$lcol)
      plot(agg$DateCET, agg$act - agg$prd, type = "l", main = paste("Residuals", cfg$title), ylab = "Act - Prd", xlab = "Date"); abline(h = 0, col = 2, lty = 2)
      dev.off()
    }, error = function(e) { if(dev.cur() > 1) dev.off() })
  }
}

# 12-month summary plots
month_plot_models <- list(
  "1_LR"                       = list(col = "Prediction_LR",            title = "LR (no weather)",       lcol = 4),
  "2_LR_micro"                 = list(col = "Prediction_LR_micro",      title = "LR (micro)",            lcol = 3),
  "3_LR_nomicro"               = list(col = "Prediction_LR_nomicro",    title = "LR (nomicro)",          lcol = "dodgerblue"),
  "4_RF_noweather_24h_VP"      = list(col = "RF_noweather_24h_VP",      title = "RF no-weather VP",      lcol = "brown"),
  "5_RF_micro_24h_VP"          = list(col = "RF_micro_24h_VP",          title = "RF micro VP",           lcol = 2),
  "6_RF_nomicro_24h_VP"        = list(col = "RF_nomicro_24h_VP",        title = "RF nomicro VP",         lcol = "coral")
)

for(model_key in names(month_plot_models)) {
  model_cfg <- month_plot_models[[model_key]]
  pred_col <- model_cfg$col
  invisible(lapply(unique_ids, function(id) {
    model_dir_path <- file.path(MOD_DIR, "12month_plot", model_key)
    dir.create(model_dir_path, recursive = TRUE, showWarnings = FALSE)
    svg(file.path(model_dir_path, paste0(id, ".svg")), 12, 9)
    par(mfrow=c(3,4), mar=c(3,3,2,1))
    for(m in 1:12) {
      agg <- bike_dt[FeatureID == id & month == m, .(act = mean(Count), prd = mean(get(pred_col))), by = hourCET][order(hourCET)]
      if(nrow(agg) > 0) {
        ylim_max <- max(c(agg$act, agg$prd), na.rm = TRUE) * 1.2
        plot(agg$hourCET, agg$act, main=paste("M", m), pch=16, xlab="Hour", ylab="Count", ylim=c(0, ylim_max))
        lines(agg$hourCET, agg$prd, col=model_cfg$lcol, lwd=2)
        if(m == 1) legend("topright", legend=c("Actual", model_cfg$title), col=c("black", model_cfg$lcol),
                         lty=c(NA,1), pch=c(16,NA), cex=0.5, bty="n")
      } else plot.new()
    }
    dev.off()
  }))
}

# R2/RMSE scatter plots
scatter_types <- list(
  LR          = list(col = "Prediction_LR",       model_dir = "1_LR",                       file_sfx = "",                     rgb_col = rgb(0, 0, 1, 0.3)),
  LR_micro    = list(col = "Prediction_LR_micro", model_dir = "2_LR_micro",                 file_sfx = "_LR_micro",           rgb_col = rgb(0, 0.6, 0, 0.3)),
  LR_nomicro  = list(col = "Prediction_LR_nomicro", model_dir = "3_LR_nomicro",             file_sfx = "_LR_nomicro",         rgb_col = rgb(0, 0.4, 0.8, 0.3)),
  RF_noweather_24h_VP = list(col = "RF_noweather_24h_VP", model_dir = "4_RF_noweather_24h_VP", file_sfx = "_RF_noweather", rgb_col = rgb(0.5, 0.25, 0, 0.3)),
  RF_micro_24h_VP        = list(col = "RF_micro_24h_VP",       model_dir = "5_RF_micro_24h_VP",        file_sfx = "_RF_micro_VP",        rgb_col = rgb(1, 0, 0, 0.3)),
  RF_nomicro_24h_VP      = list(col = "RF_nomicro_24h_VP",     model_dir = "6_RF_nomicro_24h_VP",      file_sfx = "_RF_nomicro_VP",      rgb_col = rgb(1, 0.4, 0.4, 0.3))
)
for(type in names(scatter_types)) {
  cfg <- scatter_types[[type]]
  for(met in c("R2", "RMSE")) {
    subdir <- file.path(cfg$model_dir, paste0(met, "_plots"))
    dir.create(file.path(MOD_DIR, subdir), recursive = TRUE, showWarnings = FALSE)
    f_path <- file.path(MOD_DIR, subdir, paste0(met, "_Comparison_All_Stations", cfg$file_sfx, ".svg"))
    metric_col <- if(met == "RMSE") paste0("RMSE_pct_", type) else paste0("R2_", type)
    tryCatch({
      svg(f_path, 20, 20)
      par(mfrow=c(6,6), mar=c(4,4,2,1))
      for(id in metrics[order(Cluster, FeatureID), FeatureID]) {
        sub <- bike_dt[FeatureID == id]
        val <- metrics[FeatureID == id, get(metric_col)]
        idx <- sub$Count > 0 & !is.na(sub[[cfg$col]])
        if(any(idx)) {
          m_val <- max(c(sub$Count[idx], sub[[cfg$col]][idx]))
          plot(sub$Count[idx], sub[[cfg$col]][idx], main=paste0(id, " (", met, "=", val, ")"),
               xlim=c(0, m_val), ylim=c(0, m_val), pch=16, col=cfg$rgb_col)
          abline(0, 1, lty=2, col=2)
        } else plot.new()
      }
      dev.off()
    }, error = function(e) {
      if(dev.cur() > 1) dev.off()
    })
  }
}

# Full Year Plots
cat("Generating Full Year Plots...\n")
n_stations_all <- length(unique_ids); ncols <- 6
nrows <- ceiling(n_stations_all / ncols)

fy_types <- list(
  LR          = list(col = "Prediction_LR",       lcol = 4),
  LR_micro    = list(col = "Prediction_LR_micro", lcol = 3),
  LR_nomicro  = list(col = "Prediction_LR_nomicro", lcol = "dodgerblue"),
  RF_noweather_24h_VP    = list(col = "RF_noweather_24h_VP",    lcol = "brown"),
  RF_micro_24h_VP        = list(col = "RF_micro_24h_VP",       lcol = 2),
  RF_nomicro_24h_VP      = list(col = "RF_nomicro_24h_VP",     lcol = "coral")
)
for(type in names(fy_types)) {
  p_col    <- fy_types[[type]]$col
  line_col <- fy_types[[type]]$lcol
  svg(file.path(MOD_DIR, "Full_year_plot", paste0("All_Stations_Full_Year_", type, ".svg")),
      width = 20, height = 3.5 * nrows)
  par(mfrow = c(nrows, ncols), mar = c(3, 3, 2, 1), mgp = c(2, 0.8, 0))
  for(thisid in unique_ids) {
    sub <- bike_dt[FeatureID == thisid]
    agg <- sub[, .(Act = mean(Count, na.rm = TRUE),
                   Prd = mean(get(p_col), na.rm = TRUE)), by = hourCET][order(hourCET)]
    if(nrow(agg) > 0) {
      m_val <- max(c(agg$Act, agg$Prd), na.rm = TRUE) * 1.1
      plot(agg$hourCET, agg$Act, main = paste0(thisid, " - ", type),
           ylim = c(0, m_val), xlab = "Hour", ylab = "Count", pch = 16, xlim = c(0, 23))
      lines(agg$hourCET, agg$Prd, col = line_col, lwd = 2)
    } else plot.new()
  }
  dev.off()
}

cat("\nAnalysis Complete. Files saved to: ", normalizePath(MOD_DIR), "\n")
