

gc()
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("..")

train_flat <- fread("data/prepared/train_split_flattened.csv")
val_flat <- fread("data/prepared/val_split_flattened.csv")
test_flat <- fread("data/prepared/test_data_flattened.csv")

train_labels <- factor(train_flat$label)
val_labels <- factor(val_flat$label)
test_labels <- factor(test_flat$label)

train_flat_features <- train_flat[, -1]
val_flat_features <- val_flat[, -1]
test_flat_features <- test_flat[, -1]

gc()

add_statistical_features <- function(data) {
  data %>%
    mutate(
      mean = rowMeans(data),
      sd = apply(data, 1, sd),
      variance = apply(data, 1, var)
    )
}

train_flat_features <- add_statistical_features(train_flat_features)
gc()
val_flat_features <- add_statistical_features(val_flat_features)
gc()
test_flat_features <- add_statistical_features(test_flat_features)
gc()

add_edge_features <- function(data) {
  apply(data, 1, function(row) {
    img <- matrix(row, nrow = 28, byrow = TRUE)
    edge_horizontal <- abs(diff(img, differences = 1, axis = 1))
    edge_vertical <- abs(diff(img, differences = 1, axis = 2))
    sum(edge_horizontal) + sum(edge_vertical)
  }) %>% as.data.frame() %>% rename(edge_sum = ".")
}

train_edge_features <- add_edge_features(train_flat_features)
gc()
val_edge_features <- add_edge_features(val_flat_features)
gc()
test_edge_features <- add_edge_features(test_flat_features)
gc()

add_gradient_features <- function(data) {
  horizontal <- apply(data, 1, function(row) {
    img <- matrix(row, nrow = 28, byrow = TRUE)
    sum(abs(diff(img, axis = 1)))
  })
  vertical <- apply(data, 1, function(row) {
    img <- matrix(row, nrow = 28, byrow = TRUE)
    sum(abs(diff(img, axis = 2)))
  })
  data.frame(gradient_horizontal = horizontal, gradient_vertical = vertical)
}

train_gradient_features <- add_gradient_features(train_flat_features)
gc()
val_gradient_features <- add_gradient_features(val_flat_features)
gc()
test_gradient_features <- add_gradient_features(test_flat_features)
gc()

reduce_dimensions <- function(data, n_components) {
  pca_result <- prcomp(as.matrix(data), scale. = TRUE)
  as.data.frame(pca_result$x[, 1:n_components])
}

train_flat_pca <- reduce_dimensions(train_flat_features, n_components = 700)
gc()
val_flat_pca <- reduce_dimensions(val_flat_features, n_components = 700)
gc()
test_flat_pca <- reduce_dimensions(test_flat_features, n_components = 700)
gc()

combine_features <- function(pca_data, edge_features, gradient_features, labels) {
  combined <- cbind(pca_data, edge_features, gradient_features)
  combined$label <- labels
  combined
}

train_flat_combined <- combine_features(
  train_flat_pca, train_edge_features, train_gradient_features, train_labels
)
gc()
val_flat_combined <- combine_features(
  val_flat_pca, val_edge_features, val_gradient_features, val_labels
)
gc()
test_flat_combined <- combine_features(
  test_flat_pca, test_edge_features, test_gradient_features, test_labels
)
gc()

engineered_dir <- "data/engineered"
dir.create(engineered_dir, recursive = TRUE, showWarnings = FALSE)

write.csv(train_flat_combined, file.path(engineered_dir, "train_flat_engineered.csv"), row.names = FALSE)
gc()
write.csv(val_flat_combined, file.path(engineered_dir, "val_flat_engineered.csv"), row.names = FALSE)
gc()
write.csv(test_flat_combined, file.path(engineered_dir, "test_flat_engineered.csv"), row.names = FALSE)
gc()

cat("Feature engineering completed with additional features and 150 PCA components. Engineered datasets saved in 'data/engineered'.\n")
gc()
rm(list = ls())
