#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  library(stats)
}))

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[[1]] else "/home/runner/work/BikeCount/BikeCount/PreparedForModel_micro_nomicro.csv"
output_path <- if (length(args) >= 2) args[[2]] else "/home/runner/work/BikeCount/BikeCount/model4_outputs/clusters_avgcount.csv"
n_clusters <- if (length(args) >= 3) as.integer(args[[3]]) else 12L
if (is.na(n_clusters) || n_clusters < 1) n_clusters <- 12L

df <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)

required_cols <- c(
  "FeatureID", "Count", "year_",
  "Build 125m", "Tree Proportion", "Plant Proportion", "Grass Proportion"
)
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
}

df$FeatureID <- as.character(df$FeatureID)
df$Count <- suppressWarnings(as.numeric(df$Count))
df$year_ <- suppressWarnings(as.integer(df$year_))

stations <- sort(unique(df$FeatureID))
rows_out <- list()
row_i <- 1L

for (test_station in stations) {
  train_df <- df[df$FeatureID != test_station, , drop = FALSE]
  test_df <- df[df$FeatureID == test_station, , drop = FALSE]
  if (nrow(train_df) == 0 || nrow(test_df) == 0) next

  train_feats <- aggregate(
    train_df[, c("Build 125m", "Tree Proportion", "Plant Proportion", "Grass Proportion")],
    by = list(FeatureID = train_df$FeatureID),
    FUN = function(x) mean(suppressWarnings(as.numeric(x)), na.rm = TRUE)
  )

  train_hist <- train_df[train_df$year_ %in% c(2022L, 2023L), c("FeatureID", "Count"), drop = FALSE]
  train_mean <- aggregate(
    train_hist$Count,
    by = list(FeatureID = train_hist$FeatureID),
    FUN = function(x) mean(suppressWarnings(as.numeric(x)), na.rm = TRUE)
  )
  names(train_mean)[2] <- "mean_count_2022_2023"

  train_station <- merge(train_feats, train_mean, by = "FeatureID", all.x = TRUE)
  train_station$mean_count_2022_2023[is.na(train_station$mean_count_2022_2023)] <- mean(train_station$mean_count_2022_2023, na.rm = TRUE)
  train_station[is.na(train_station)] <- 0

  feat_mat <- as.matrix(train_station[, c("Build 125m", "Tree Proportion", "Plant Proportion", "Grass Proportion", "mean_count_2022_2023"), drop = FALSE])
  feat_mat <- scale(feat_mat)
  feat_mat[is.na(feat_mat)] <- 0

  k_use <- min(n_clusters, nrow(train_station))
  if (is.na(k_use) || k_use < 1) k_use <- 1L
  set.seed(42)
  km <- kmeans(feat_mat, centers = k_use, iter.max = 100, nstart = 10)
  train_station$cluster_id <- as.integer(km$cluster) - 1L

  test_feats <- aggregate(
    test_df[, c("Build 125m", "Tree Proportion", "Plant Proportion", "Grass Proportion")],
    by = list(FeatureID = test_df$FeatureID),
    FUN = function(x) mean(suppressWarnings(as.numeric(x)), na.rm = TRUE)
  )
  test_hist <- test_df[test_df$year_ %in% c(2022L, 2023L), c("FeatureID", "Count"), drop = FALSE]
  if (nrow(test_hist) == 0) {
    test_mean_count <- mean(train_station$mean_count_2022_2023, na.rm = TRUE)
  } else {
    test_mean_count <- mean(suppressWarnings(as.numeric(test_hist$Count)), na.rm = TRUE)
  }
  if (is.na(test_mean_count)) test_mean_count <- mean(train_station$mean_count_2022_2023, na.rm = TRUE)
  if (nrow(test_feats) == 0) next
  test_feats$mean_count_2022_2023 <- test_mean_count
  test_feats[is.na(test_feats)] <- 0

  test_mat <- as.matrix(test_feats[, c("Build 125m", "Tree Proportion", "Plant Proportion", "Grass Proportion", "mean_count_2022_2023"), drop = FALSE])
  center <- attr(feat_mat, "scaled:center")
  scalev <- attr(feat_mat, "scaled:scale")
  if (is.null(center)) center <- rep(0, ncol(test_mat))
  if (is.null(scalev)) scalev <- rep(1, ncol(test_mat))
  scalev[is.na(scalev) | scalev == 0] <- 1
  test_mat <- sweep(test_mat, 2, center, "-")
  test_mat <- sweep(test_mat, 2, scalev, "/")
  test_mat[is.na(test_mat)] <- 0

  d <- as.matrix(dist(rbind(km$centers, test_mat)))
  test_cluster <- which.min(d[1:nrow(km$centers), nrow(d)]) - 1L

  train_out <- data.frame(
    Fold_Test_Station = as.character(test_station),
    FeatureID = as.character(train_station$FeatureID),
    cluster_id = as.integer(train_station$cluster_id),
    is_test = 0L,
    stringsAsFactors = FALSE
  )
  test_out <- data.frame(
    Fold_Test_Station = as.character(test_station),
    FeatureID = as.character(test_station),
    cluster_id = as.integer(test_cluster),
    is_test = 1L,
    stringsAsFactors = FALSE
  )

  rows_out[[row_i]] <- train_out
  row_i <- row_i + 1L
  rows_out[[row_i]] <- test_out
  row_i <- row_i + 1L
}

if (length(rows_out) == 0) {
  stop("No clusters generated.")
}

out <- do.call(rbind, rows_out)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(out, output_path, row.names = FALSE)
cat(sprintf("Wrote clusters to %s\n", output_path))
