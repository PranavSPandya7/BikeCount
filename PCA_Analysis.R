library(data.table)
library(ggplot2)
library(ggrepel)
library(gridExtra)
library(stats)
library(cluster)

setwd("C:/Users/pandya/OneDrive - UCL/Data shared/Response 30Mar")
dir.create("PCA", showWarnings = FALSE)
data_prepared <- fread("PreparedForModel_micro_nomicro.csv")
data_prepared[, UTC1 := as.POSIXct(UTC1, format = '%d-%m-%y %H:%M', tz = 'UTC')]
data_prepared <- data_prepared[year(UTC1) %in% c(2022, 2023)]
data_prepared[, Latitude := as.numeric(as.character(Latitude))]
data_prepared[, Longitude := as.numeric(as.character(Longitude))]

# CALCULATE DISTANCE FROM CENTER in meters (Brussels: 50.8466, 4.3528)
# Equirectangular approximation: 1° lat ≈ 111320 m, 1° lon ≈ 111320*cos(50.8466°) ≈ 70340 m
data_prepared[, Distance_From_Center := sqrt(((Latitude - 50.8466) * 111320)^2 + ((Longitude - 4.3528) * 70340)^2)]

# Rename columns
for (m in list(c("Build 125m", "Build125m"), c("Build 125m", "Build125m"), c("Tree Proportion", "Tree_Pct"), c("Grass Proportion", "Grass_Pct"), c("Plant Proportion", "Plant_Pct"))) if (m[1] %in% names(data_prepared)) setnames(data_prepared, m[1], m[2])
# GVI (Green View Index) = sum of tree, grass, plant pixel proportions × 100 (%)
data_prepared[, GVI := (Tree_Pct + Grass_Pct + Plant_Pct) * 100]
if (!("Centrality_encoded" %in% names(data_prepared))) data_prepared[, Centrality_encoded := NA_real_]
feature_matrix <- data_prepared[, .(Latitude = mean(Latitude, na.rm=TRUE), Longitude = mean(Longitude, na.rm=TRUE), Distance_From_Center = mean(Distance_From_Center, na.rm=TRUE), Build125m = mean(Build125m, na.rm=TRUE), Tree_Pct = mean(Tree_Pct, na.rm=TRUE), Grass_Pct = mean(Grass_Pct, na.rm=TRUE), Plant_Pct = mean(Plant_Pct, na.rm=TRUE), GVI = mean(GVI, na.rm=TRUE), RoadType_encoded = mean(RoadType_encoded, na.rm=TRUE), Directionality_encoded = mean(Directionality_encoded, na.rm=TRUE), Centrality_encoded = mean(Centrality_encoded, na.rm=TRUE), Ta = mean(Ta, na.rm=TRUE), RH = mean(RH, na.rm=TRUE), rain = mean(rain, na.rm=TRUE), windy = mean(windy, na.rm=TRUE), Shade = mean(Shade, na.rm=TRUE), hot = mean(hot, na.rm=TRUE), cold = mean(cold, na.rm=TRUE), Tmrt = mean(Tmrt, na.rm=TRUE), UTCI = mean(UTCI, na.rm=TRUE)), by = FeatureID]

# Directionality intentionally excluded from PCA features in this run.

# Featureset --------------------
# OLD feature set (14 features)
# feature_cols <- c("Distance_From_Center", "Build125m", "Tree_Pct", "Grass_Pct", "Plant_Pct", "Ta", "RH", "windy", "Shade", "hot", "cold", "Tmrt", "RoadType_encoded", "Directionality_encoded")

# Core 4-feature spatial set (no Directionality)
feature_cols <- c("Distance_From_Center", "GVI", "Build125m", "RoadType_encoded")


feature_cols_requested <- feature_cols
feature_cols <- feature_cols[sapply(feature_cols, function(v) v %in% names(feature_matrix) && any(is.finite(feature_matrix[[v]])))]
dropped_cols <- setdiff(feature_cols_requested, feature_cols)
if (length(dropped_cols) > 0) cat(sprintf("Dropped unavailable/all-NA features: %s\n", paste(dropped_cols, collapse = ", ")))

cat(sprintf("Selected features: %s\n", paste(feature_cols, collapse = ", ")))
X <- as.matrix(feature_matrix[, ..feature_cols])

# Perform PCA
pca_result <- prcomp(X, scale = TRUE, center = TRUE)
# Use first 2 PCs for clustering (Core4 run gives K=6 best silhouette)
pca_2d <- pca_result$x[, 1:2]

# --- PCA Variance Summary ---
cat("\n=== PCA VARIANCE EXPLAINED ===\n")
pca_var <- summary(pca_result)$importance
print(round(pca_var, 4))
var_dt <- data.table(PC = paste0("PC", 1:ncol(pca_var)),
                     StdDev = pca_var[1,],
                     PropVariance = pca_var[2,],
                     CumulVariance = pca_var[3,])
fwrite(var_dt, "PCA/PCA_Variance_Explained.csv")
cat(sprintf("PC1 explains %.1f%%, PC2 explains %.1f%%, cumulative = %.1f%%\n",
            pca_var[2,1]*100, pca_var[2,2]*100, pca_var[3,2]*100))

# --- PCA Loadings ---
cat("\n=== PCA LOADINGS (PC1 & PC2) ===\n")
loadings_dt <- data.table(Feature = rownames(pca_result$rotation),
                          PC1 = round(pca_result$rotation[,1], 4),
                          PC2 = round(pca_result$rotation[,2], 4))
loadings_dt <- loadings_dt[order(-abs(PC1))]
print(loadings_dt)
fwrite(loadings_dt, "PCA/PCA_Loadings.csv")

# --- Scree Plot ---
eigenvalues <- pca_result$sdev^2
pdf("PCA/Scree_Plot.pdf", width = 10, height = 7)
par(mar = c(5, 5, 4, 3))
plot(1:length(eigenvalues), eigenvalues, type = "b", pch = 19, col = "#4ECDC4",
     xlab = "Principal Component", ylab = "Eigenvalue",
     main = "Scree Plot — PCA Eigenvalues", xaxt = "n", lwd = 2)
axis(1, at = 1:length(eigenvalues), labels = paste0("PC", 1:length(eigenvalues)))
abline(h = 1, lty = 2, col = "red")
text(1:length(eigenvalues), eigenvalues, labels = sprintf("%.2f", eigenvalues),
     pos = 3, cex = 0.8, font = 2)
dev.off()

# --- Silhouette scores for K=2 to K=10 ---
cat("\n=== SILHOUETTE SCORES ===\n")
k_values <- 2:10
silhouette_scores <- numeric(length(k_values))
for (i in seq_along(k_values)) {
  k <- k_values[i]
  set.seed(42)
  km <- kmeans(pca_2d, centers = k, nstart = 25)
  sil <- silhouette(km$cluster, dist(pca_2d))
  silhouette_scores[i] <- mean(sil[, 3])
}
best_k <- k_values[which.max(silhouette_scores)]
for (i in seq_along(k_values)) {
  marker <- ifelse(k_values[i] == best_k, " <-- BEST", "")
  cat(sprintf("  K=%d: Silhouette = %.4f%s\n", k_values[i], silhouette_scores[i], marker))
}

silhouette_data <- data.table(K = k_values, Silhouette_Score = silhouette_scores)
fwrite(silhouette_data, "PCA/Silhouette_Scores.csv")

# --- Silhouette bar plot ---
pdf("PCA/Silhouette_Scores_Plot.pdf", width = 10, height = 7)
par(mar = c(5, 5, 4, 3))
bp <- barplot(silhouette_data$Silhouette_Score,
        names.arg = paste0("K=", silhouette_data$K),
        main = "Silhouette Score by Number of Clusters (K)",
        xlab = "Number of Clusters (K)", ylab = "Silhouette Score",
        col = ifelse(silhouette_data$K == best_k, "#FF6B6B", "#4ECDC4"),
        ylim = c(0, max(silhouette_data$Silhouette_Score) * 1.15))
text(x = bp, y = silhouette_data$Silhouette_Score,
     labels = sprintf("%.4f", silhouette_data$Silhouette_Score), pos = 3, font = 2)
dev.off()

# --- K=5 Cluster Assignments ---
cat(sprintf("\n=== K=%d CLUSTER ASSIGNMENTS (Best Silhouette = %.4f) ===\n",
            best_k, max(silhouette_scores)))
set.seed(42)
km_best <- kmeans(pca_2d, centers = best_k, nstart = 25)

pca_plot_dt <- data.table(FeatureID = feature_matrix$FeatureID,
                          PC1 = pca_2d[,1], PC2 = pca_2d[,2])
for(k in 2:10) {
  set.seed(42)
  km_k <- kmeans(pca_2d, centers = k, nstart = 25)
  pca_plot_dt[, paste0("Cluster_k", k) := km_k$cluster]
}
# Use manual override clusters (km_best$cluster) for plot and CSV
pca_plot_dt[, Cluster := factor(km_best$cluster)]
fwrite(pca_plot_dt, "PCA/PCA_Station_Scores.csv")

# Print cluster assignments
for (cl in sort(unique(km_best$cluster))) {
  stns <- feature_matrix$FeatureID[km_best$cluster == cl]
  cat(sprintf("  Cluster %d (%d stations): %s\n", cl, length(stns), paste(sort(stns), collapse = ", ")))
}
cat(sprintf("\n  Cluster sizes: %s\n", paste(table(km_best$cluster), collapse = " / ")))

# Print as R code for FINAL.R
cat("\n=== COPY THIS TO Predictive_Bike_Model_FINAL.R (STATION_CLUSTERS) ===\n")
cat("STATION_CLUSTERS <- list(\n")
clusters <- sort(unique(km_best$cluster))
for (j in seq_along(clusters)) {
  cl <- clusters[j]
  stns <- sort(feature_matrix$FeatureID[km_best$cluster == cl])
  comma <- ifelse(j < length(clusters), ",", "")
  cat(sprintf('  "%d" = c(%s)%s\n', cl, paste0('"', stns, '"', collapse = ", "), comma))
}
cat(")\n")

# Optional manual override (disabled): keep kmeans(best_k) clusters in outputs.
# rf_clusters <- list(
#   "1" = c("CJE181", "CVT387", "CLW239"),
#   "2" = c("CEV011", "CB1699", "CB1599"),
#   "3" = c("COM205", "CEK18", "CEE016"),
#   "4" = c("CAT17", "CEK31", "CB2105"),
#   "5" = c("CEK049", "CB1143", "CB02411"),
#   "6" = c("CB1142", "CB1101", "CJM90")
# )
# if (exists("rf_clusters")) {
#   for (cn in names(rf_clusters)) pca_plot_dt[FeatureID %in% rf_clusters[[cn]], Cluster := factor(cn)]
#   fwrite(pca_plot_dt, "PCA/PCA_Station_Scores.csv")
# }

# --- PCA Scatter Plot ---
sil_best <- max(silhouette_scores)

# Feature vectors for biplot overlay (uses whichever feature set is active)
loadings_plot_dt <- data.table(
  Feature = rownames(pca_result$rotation),
  PC1 = pca_result$rotation[, 1],
  PC2 = pca_result$rotation[, 2]
)

axis_span <- min(diff(range(pca_plot_dt$PC1, na.rm = TRUE)),
                 diff(range(pca_plot_dt$PC2, na.rm = TRUE)))
if (!is.finite(axis_span) || axis_span <= 0) axis_span <- 1

loading_norm <- max(sqrt(loadings_plot_dt$PC1^2 + loadings_plot_dt$PC2^2), na.rm = TRUE)
if (!is.finite(loading_norm) || loading_norm <= 0) loading_norm <- 1

loading_scale <- 0.35 * axis_span / loading_norm
loadings_plot_dt[, `:=`(PC1_end = PC1 * loading_scale, PC2_end = PC2 * loading_scale)]

p <- ggplot(pca_plot_dt, aes(x = PC1, y = PC2, color = Cluster, label = FeatureID)) +
  geom_point(size = 4) +
  geom_segment(
    data = loadings_plot_dt,
    inherit.aes = FALSE,
    aes(x = 0, y = 0, xend = PC1_end, yend = PC2_end),
    arrow = grid::arrow(length = grid::unit(0.15, "cm")),
    color = "grey35",
    linewidth = 0.5
  ) +
  geom_text_repel(
    data = loadings_plot_dt,
    inherit.aes = FALSE,
    aes(x = PC1_end, y = PC2_end, label = Feature),
    color = "grey20",
    size = 3,
    box.padding = 0.25,
    point.padding = 0.1,
    min.segment.length = 0,
    segment.color = "grey60",
    show.legend = FALSE
  ) +
  geom_text_repel(size = 3.5, max.overlaps = Inf, box.padding = 0.6, min.segment.length = 0, force = 2, show.legend = FALSE, segment.color = NA) +
  scale_color_manual(values = c("1" = "#E41A1C", "2" = "#377EB8", "3" = "#4DAF4A", "4" = "#984EA3", "5" = "#FF7F00", "6" = "#000000", "7" = "#A65628", "8" = "#F781BF", "9" = "#999999", "10" = "#66C2A5")) +
  labs(title = sprintf("PCA — 18 Stations in PC1-PC2 Space (k=%d, Silhouette=%.3f)", best_k, sil_best),
       x = sprintf("PC1 (%.1f%% variance)", pca_var[2,1]*100),
       y = sprintf("PC2 (%.1f%% variance)", pca_var[2,2]*100)) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right",
        panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.6),
      axis.ticks = element_line(color = "black"))

pdf("PCA/PCA_Scatter_Plot.pdf", width = 10, height = 8)
print(p)
dev.off()

png("PCA/PCA_Scatter_Plot.png", width = 1200, height = 900, res = 120)
print(p)
dev.off()

svg("PCA/PCA_Scatter_Plot.svg", width = 10, height = 8)
print(p)
dev.off()

cat("\nPCA outputs saved to PCA/ folder\n")
cat(sprintf("Files: %s\n", normalizePath("PCA")))

# --- PCA Biplot + OSM Map combined (PCA_withmap.svg) ---
library(sf)
library(maptiles)
library(tidyterra)

# Station coordinates
map_dt <- feature_matrix[, .(FeatureID, Latitude, Longitude)]
map_dt[, Cluster := pca_plot_dt$Cluster[match(FeatureID, pca_plot_dt$FeatureID)]]

pts_sf <- st_as_sf(map_dt, coords = c("Longitude", "Latitude"), crs = 4326)
bbox <- st_bbox(pts_sf)
pad <- 0.008
bbox_padded <- c(
  xmin = as.numeric(bbox["xmin"]) - pad, ymin = as.numeric(bbox["ymin"]) - pad,
  xmax = as.numeric(bbox["xmax"]) + pad, ymax = as.numeric(bbox["ymax"]) + pad
)
bbox_sfc <- st_as_sfc(st_bbox(bbox_padded, crs = st_crs(4326)))
osm_tiles <- tryCatch(
  get_tiles(bbox_sfc, provider = "OpenStreetMap", zoom = 14, crop = TRUE),
  error = function(e) {
    message("OSM failed, trying CartoDB.Positron")
    tryCatch(get_tiles(bbox_sfc, provider = "CartoDB.Positron", zoom = 14, crop = TRUE),
             error = function(e2) NULL)
  }
)

p_map <- ggplot()
if (!is.null(osm_tiles)) {
  p_map <- p_map + tidyterra::geom_spatraster_rgb(data = osm_tiles, maxcell = Inf)
}
p_map <- p_map +
  geom_point(data = map_dt, aes(x = Longitude, y = Latitude, color = Cluster),
             size = 4, alpha = 0.95) +
  geom_text_repel(data = map_dt, aes(x = Longitude, y = Latitude, label = FeatureID),
                  size = 3.2, max.overlaps = Inf, box.padding = 0.4, segment.color = "grey40") +
  scale_color_manual(values = c("1" = "#E41A1C", "2" = "#377EB8", "3" = "#4DAF4A",
                                "4" = "#984EA3", "5" = "#FF7F00", "6" = "#000000")) +
  coord_sf(xlim = c(bbox_padded["xmin"], bbox_padded["xmax"]),
           ylim = c(bbox_padded["ymin"], bbox_padded["ymax"]),
           crs = 4326, expand = FALSE) +
  labs(title = "Station Map — Cluster Assignments on OSM",
       x = "Longitude", y = "Latitude", color = "Cluster") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "right",
        axis.line = element_line(color = "black", linewidth = 0.4),
        axis.ticks = element_line(color = "black"),
        plot.title = element_text(face = "bold", hjust = 0.5))

p_combined <- gridExtra::grid.arrange(p, p_map, ncol = 1, heights = c(1, 1.1))

svg("PCA/PCA_withmap.svg", width = 10, height = 16)
gridExtra::grid.arrange(p, p_map, ncol = 1, heights = c(1, 1.1))
dev.off()

cat("\nPCA + Map combined saved:\n")
cat(normalizePath("PCA/PCA_withmap.svg"), "\n")