# Install and load required libraries
if (!require(dplyr)) install.packages("dplyr")
if (!require(data.table)) install.packages("data.table")
if (!require(caret)) install.packages("caret")
if (!require(reshape2)) install.packages("reshape2")
if (!require(imager)) install.packages("imager")
if (!require(FactoMineR)) install.packages("FactoMineR")

library(dplyr)
library(data.table)
library(caret)
library(reshape2)
library(imager)
library(FactoMineR)

gc()
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("..")

# Load prepared (flattened) data
train_flat <- fread("data/prepared/train_split_flattened.csv")
val_flat <- fread("data/prepared/val_split_flattened.csv")
test_flat <- fread("data/prepared/test_data_flattened.csv")

# Separate labels for classification tasks
train_labels <- factor(train_flat$label)  # Convert labels to factors
val_labels <- factor(val_flat$label)
test_labels <- factor(test_flat$label)

train_flat_features <- train_flat[, -1]
val_flat_features <- val_flat[, -1]
test_flat_features <- test_flat[, -1]

reduce_dimensions <- function(data, n_components) {
  pca_result <- prcomp(as.matrix(data), scale. = TRUE)
  reduced_data <- as.data.frame(pca_result$x[, 1:n_components])
  return(reduced_data)
}

train_flat_pca <- reduce_dimensions(train_flat_features, n_components = 200)  
val_flat_pca <- reduce_dimensions(val_flat_features, n_components = 200)
test_flat_pca <- reduce_dimensions(test_flat_features, n_components = 200)

train_flat_pca$label <- train_labels
val_flat_pca$label <- val_labels
test_flat_pca$label <- test_labels

gc()
engineered_dir <- "data/engineered"
dir.create(engineered_dir, recursive = TRUE, showWarnings = FALSE)

write.csv(train_flat_pca, file.path(engineered_dir, "train_flat_engineered.csv"), row.names = FALSE)
write.csv(val_flat_pca, file.path(engineered_dir, "val_flat_engineered.csv"), row.names = FALSE)
write.csv(test_flat_pca, file.path(engineered_dir, "test_flat_engineered.csv"), row.names = FALSE)

cat("Feature engineering completed with 200 PCA components and labels. Engineered datasets saved in 'data/engineered'.\n")
gc()
