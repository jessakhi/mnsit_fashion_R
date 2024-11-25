# Install and load necessary packages
if (!require(caret)) install.packages("caret", dependencies = TRUE)
if (!require(glmnet)) install.packages("glmnet", dependencies = TRUE)
if (!require(doParallel)) install.packages("doParallel", dependencies = TRUE)
if (!require(ranger)) install.packages("ranger", dependencies = TRUE)
if (!require(rpart)) install.packages("rpart", dependencies = TRUE)
if (!require(xgboost)) install.packages("xgboost", dependencies = TRUE)

library(caret)
library(glmnet)
library(doParallel)
library(ranger)
library(rpart)
library(xgboost)

# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(123)

# Load engineered datasets
engineered_dir <- "data/engineered"
train_data <- read.csv(file.path(engineered_dir, "train_flat_engineered.csv"))
val_data <- read.csv(file.path(engineered_dir, "val_flat_engineered.csv"))
test_data <- read.csv(file.path(engineered_dir, "test_flat_engineered.csv"))

# Prepare data
train_data$label <- factor(make.names(train_data$label))
val_data$label <- factor(make.names(val_data$label))
test_labels <- factor(make.names(test_data$label))
test_data <- test_data[, -which(names(test_data) == "label")]

# Define control parameters
control <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  allowParallel = TRUE
)

metric <- "Accuracy"
processing_times <- list()

# Logistic Regression with Elastic Net Regularization
start_time <- Sys.time()
logistic_model <- train(
  label ~ ., 
  data = train_data, 
  method = "glmnet", 
  metric = metric, 
  trControl = control,
  tuneGrid = expand.grid(
    alpha = 0.5,     # Elastic Net mixing parameter
    lambda = 0.01    # Regularization parameter
  )
)
processing_times$logistic_model <- Sys.time() - start_time

logistic_val_predictions <- predict(logistic_model, val_data)
logistic_val_accuracy <- mean(logistic_val_predictions == val_data$label)

cat("Logistic Regression Accuracy on Validation Set:", logistic_val_accuracy, "\n")
cat("Training Time:", processing_times$logistic_model, "\n")

# K-Nearest Neighbors (KNN)
start_time <- Sys.time()
knn_model <- train(
  label ~ ., 
  data = train_data, 
  method = "knn", 
  metric = metric, 
  trControl = control,
  tuneGrid = expand.grid(k = 3)  # Optimal k value
)
processing_times$knn_model <- Sys.time() - start_time

knn_val_predictions <- predict(knn_model, val_data)
knn_val_accuracy <- mean(knn_val_predictions == val_data$label)

cat("KNN Validation Accuracy:", knn_val_accuracy, "\n")
cat("KNN Training Time:", processing_times$knn_model, "\n")

# Random Forest
start_time <- Sys.time()
rf_model <- train(
  label ~ ., 
  data = train_data, 
  method = "ranger", 
  metric = metric, 
  trControl = control,
  tuneGrid = expand.grid(
    mtry = floor(sqrt(ncol(train_data) - 1)),  # Number of variables randomly sampled
    splitrule = "gini",  # Splitting rule
    min.node.size = 1    # Minimum size of terminal nodes
  ),
  num.trees = 100  # Number of trees
)
processing_times$rf_model <- Sys.time() - start_time

rf_val_predictions <- predict(rf_model, val_data)
rf_val_accuracy <- mean(rf_val_predictions == val_data$label)

cat("Random Forest Validation Accuracy:", rf_val_accuracy, "\n")
cat("Random Forest Training Time:", processing_times$rf_model, "\n")

# Decision Tree (CART)
start_time <- Sys.time()
cart_model <- train(
  label ~ ., 
  data = train_data, 
  method = "rpart", 
  metric = metric, 
  trControl = control,
  tuneGrid = expand.grid(cp = 0.01)  # Complexity parameter
)
processing_times$cart_model <- Sys.time() - start_time

cart_val_predictions <- predict(cart_model, val_data)
cart_val_accuracy <- mean(cart_val_predictions == val_data$label)

cat("CART Validation Accuracy:", cart_val_accuracy, "\n")
cat("CART Training Time:", processing_times$cart_model, "\n")

# Extreme Gradient Boosting (XGBoost)
start_time <- Sys.time()
xgb_model <- train(
  label ~ ., 
  data = train_data, 
  method = "xgbTree", 
  metric = metric, 
  trControl = control,
  tuneGrid = expand.grid(
    nrounds = 100,         # Number of boosting rounds
    max_depth = 6,         # Maximum depth of a tree
    eta = 0.1,             # Learning rate
    gamma = 0,             # Minimum loss reduction
    colsample_bytree = 0.8,# Subsample ratio of columns
    min_child_weight = 1,  # Minimum sum of instance weight
    subsample = 0.8        # Subsample ratio of training instances
  )
)
processing_times$xgb_model <- Sys.time() - start_time

xgb_val_predictions <- predict(xgb_model, val_data)
xgb_val_accuracy <- mean(xgb_val_predictions == val_data$label)

cat("XGBoost Validation Accuracy:", xgb_val_accuracy, "\n")
cat("XGBoost Training Time:", processing_times$xgb_model, "\n")

# Save models and data
if (!dir.exists("models")) dir.create("models")
save(
  logistic_model, knn_model, rf_model, cart_model, xgb_model, processing_times, 
  file = "models/trained_models.RData"
)

save(test_data, test_labels, val_data, file = "models/mnist_data.RData")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

# Summarize validation accuracies
val_accuracies <- data.frame(
  Model = c("Logistic Regression", "KNN", "Random Forest", "CART", "XGBoost"),
  Validation_Accuracy = c(logistic_val_accuracy, knn_val_accuracy, rf_val_accuracy, cart_val_accuracy, xgb_val_accuracy)
)

print(val_accuracies)
write.csv(val_accuracies, file = "val_accuracies.csv", row.names = FALSE)

# Clean up
gc()
rm(list = ls())
