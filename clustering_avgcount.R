#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  library(stats)
}))

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[[1]] else "/home/runner/work/BikeCount/BikeCount/PreparedForModel_micro_nomicro.csv"
output_path <- if (length(args) >= 2) args[[2]] else "/home/runner/work/BikeCount/BikeCount/model4_outputs/clusters_avgcount.csv"
n_clusters <- if (length(args) >= 3) as.integer(args[[3]]) else 12L
if (is.na(n_clusters) || n_clusters < 1) n_clusters <- 12L

mean_silhouette <- function(X, labels) {
  n <- nrow(X)
  if (is.null(n) || n < 3) return(NA_real_)
  labs <- as.integer(labels)
  k <- length(unique(labs))
  if (k < 2) return(NA_real_)
  D <- as.matrix(dist(X))
  sil <- rep(NA_real_, n)
  for (i in seq_len(n)) {
    own <- labs[i]
    own_idx <- which(labs == own)
    own_idx <- own_idx[own_idx != i]
    a_i <- if (length(own_idx) == 0) 0 else mean(D[i, own_idx])
    b_i <- Inf
    for (other in setdiff(unique(labs), own)) {
      other_idx <- which(labs == other)
      if (length(other_idx) == 0) next
      b_i <- min(b_i, mean(D[i, other_idx]))
    }
    if (!is.finite(b_i)) b_i <- a_i
    denom <- max(a_i, b_i)
    sil[i] <- if (denom <= 0) 0 else (b_i - a_i) / denom
  }
  mean(sil, na.rm = TRUE)
}

choose_k_by_silhouette <- function(X, k_max) {
  n <- nrow(X)
  if (is.null(n) || n <= 2) return(1L)
  k_max <- min(as.integer(k_max), n - 1L)
  if (k_max < 2) return(1L)
  best_k <- 2L
  best_s <- -Inf
  for (k in 2:k_max) {
    set.seed(42)
    km <- kmeans(X, centers = k, iter.max = 100, nstart = 10)
    s <- mean_silhouette(X, km$cluster)
    if (!is.na(s) && s > best_s) {
      best_s <- s
      best_k <- as.integer(k)
    }
  }
  as.integer(best_k)
}

df <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)

required_cols <- c(
  "FeatureID", "Count", "year_",
  "Build 125m", "Tree Proportion", "Plant Proportion", "Grass Proportion", "Directionality_encoded"
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
    train_df[, c("Build 125m", "Tree Proportion", "Plant Proportion", "Grass Proportion", "Directionality_encoded")],
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

  feat_mat <- as.matrix(train_station[, c("Build 125m", "Tree Proportion", "Plant Proportion", "Grass Proportion", "Directionality_encoded", "mean_count_2022_2023"), drop = FALSE])
  feat_mat <- scale(feat_mat)
  feat_mat[is.na(feat_mat)] <- 0

  k_use <- choose_k_by_silhouette(feat_mat, n_clusters)
  if (is.na(k_use) || k_use < 1) k_use <- 1L
  set.seed(42)
  km <- kmeans(feat_mat, centers = k_use, iter.max = 100, nstart = 10)
  train_station$cluster_id <- as.integer(km$cluster) - 1L

  test_feats <- aggregate(
    test_df[, c("Build 125m", "Tree Proportion", "Plant Proportion", "Grass Proportion", "Directionality_encoded")],
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

  test_mat <- as.matrix(test_feats[, c("Build 125m", "Tree Proportion", "Plant Proportion", "Grass Proportion", "Directionality_encoded", "mean_count_2022_2023"), drop = FALSE])
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
