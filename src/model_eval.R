

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("..")

img_dir <- "report/img"
if (!dir.exists(img_dir)) dir.create(img_dir, recursive = TRUE)

if (file.exists("models/trained_models.RData")) {
  load("models/trained_models.RData")
  load("models/mnist_data.RData")
} else {
  stop("Required files not found.")
}

benchmark_models <- function(model_name, true_labels, predicted_probs, predicted_labels, processing_time) {
  cm <- confusionMatrix(factor(predicted_labels), factor(true_labels))
  accuracy <- cm$overall["Accuracy"]
  precision <- mean(cm$byClass[,"Precision"], na.rm = TRUE)
  recall <- mean(cm$byClass[,"Recall"], na.rm = TRUE)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  true_labels_onehot <- model.matrix(~ factor(true_labels) - 1)
  log_loss <- -mean(rowSums(true_labels_onehot * log(predicted_probs + 1e-15)))
  roc_curves <- lapply(1:ncol(predicted_probs), function(class) {
    roc(as.numeric(true_labels == colnames(predicted_probs)[class]), predicted_probs[, class])
  })
  auc_values <- sapply(roc_curves, auc)
  list(
    model = model_name,
    cm = cm,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    mean_auc = mean(auc_values, na.rm = TRUE),
    log_loss = log_loss,
    processing_time = processing_time,
    roc_curves = roc_curves
  )
}

save_individual_roc <- function(model_name, roc_curves, save_dir) {
  for (i in seq_along(roc_curves)) {
    save_path <- file.path(save_dir, paste0(model_name, "_ROC_Class_", i, ".png"))
    png(save_path, width = 800, height = 600)
    plot(roc_curves[[i]], main = paste(model_name, "- ROC Curve for Class", i), legacy.axes = TRUE)
    dev.off()
  }
}

save_confusion_matrix <- function(cm, model_name, save_path) {
  cm_table <- as.data.frame(cm$table)
  p <- ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "blue") +
    labs(title = paste(model_name, "- Confusion Matrix"), x = "True Labels", y = "Predicted Labels") +
    theme_minimal()
  ggsave(save_path, plot = p, width = 8, height = 6)
}

save_f1_scores <- function(benchmark_table, save_path) {
  p <- ggplot(benchmark_table, aes(x = Model, y = F1_Score, fill = Model)) +
    geom_bar(stat = "identity", color = "black") +
    labs(title = "F1-Score Comparison", y = "F1-Score", x = "Model") +
    theme_minimal()
  ggsave(save_path, plot = p, width = 8, height = 6)
}

logistic_probs <- predict(logistic_model, test_data, type = "prob")
logistic_labels <- predict(logistic_model, test_data, type = "raw")
logistic_eval <- benchmark_models("Logistic Regression", test_labels, logistic_probs, logistic_labels, processing_times$logistic_model)

knn_probs <- predict(knn_model, test_data, type = "prob")
knn_labels <- predict(knn_model, test_data, type = "raw")
knn_eval <- benchmark_models("KNN", test_labels, knn_probs, knn_labels, processing_times$knn_model)

rf_probs <- predict(rf_model, test_data, type = "prob")
rf_labels <- predict(rf_model, test_data, type = "raw")
rf_eval <- benchmark_models("Random Forest", test_labels, rf_probs, rf_labels, processing_times$rf_model)

cart_probs <- predict(cart_model, test_data, type = "prob")
cart_labels <- predict(cart_model, test_data, type = "raw")
cart_eval <- benchmark_models("CART", test_labels, cart_probs, cart_labels, processing_times$cart_model)

benchmark_table <- data.frame(
  Model = c("Logistic Regression", "KNN", "Random Forest", "CART"),
  Accuracy = c(logistic_eval$accuracy, knn_eval$accuracy, rf_eval$accuracy, cart_eval$accuracy),
  Precision = c(logistic_eval$precision, knn_eval$precision, rf_eval$precision, cart_eval$precision),
  Recall = c(logistic_eval$recall, knn_eval$recall, rf_eval$recall, cart_eval$recall),
  F1_Score = c(logistic_eval$f1_score, knn_eval$f1_score, rf_eval$f1_score, cart_eval$f1_score),
  Mean_AUC = c(logistic_eval$mean_auc, knn_eval$mean_auc, rf_eval$mean_auc, cart_eval$mean_auc),
  Log_Loss = c(logistic_eval$log_loss, knn_eval$log_loss, rf_eval$log_loss, cart_eval$log_loss),
  Processing_Time = c(logistic_eval$processing_time, knn_eval$processing_time, rf_eval$processing_time, cart_eval$processing_time)
)

save_individual_roc("Logistic_Regression", logistic_eval$roc_curves, img_dir)
save_individual_roc("KNN", knn_eval$roc_curves, img_dir)
save_individual_roc("Random_Forest", rf_eval$roc_curves, img_dir)
save_individual_roc("CART", cart_eval$roc_curves, img_dir)

save_confusion_matrix(logistic_eval$cm, "Logistic Regression", file.path(img_dir, "logistic_confusion_matrix.png"))
save_confusion_matrix(knn_eval$cm, "KNN", file.path(img_dir, "knn_confusion_matrix.png"))
save_confusion_matrix(rf_eval$cm, "Random Forest", file.path(img_dir, "rf_confusion_matrix.png"))
save_confusion_matrix(cart_eval$cm, "CART", file.path(img_dir, "cart_confusion_matrix.png"))

save_f1_scores(benchmark_table, file.path(img_dir, "f1_scores.png"))

write.csv(benchmark_table, file = "model_benchmark_results.csv", row.names = FALSE)

View(benchmark_table)


plot_accuracy <- function(benchmark_table) {
  ggplot(benchmark_table, aes(x = Model, y = Accuracy, fill = Model)) +
    geom_bar(stat = "identity", color = "black") +
    labs(title = "Accuracy Comparison", y = "Accuracy", x = "Model") +
    theme_minimal()
}

accuracy_plot <- plot_accuracy(benchmark_table)
print(accuracy_plot)


# ggsave("report/img/accuracy_comparison.png", plot = accuracy_plot, width = 8, height = 6)


plot_processing_time <- function(benchmark_table) {
  ggplot(benchmark_table, aes(x = Model, y = Processing_Time, fill = Model)) +
    geom_bar(stat = "identity", color = "black") +
    labs(title = "Processing Time Comparison", y = "Processing Time (s)", x = "Model") +
    theme_minimal()
}

processing_time_plot <- plot_processing_time(benchmark_table)
print(processing_time_plot)


# ggsave("report/img/processing_time_comparison.png", plot = processing_time_plot, width = 8, height = 6)


# Precision vs. Recall Scatter Plot
plot_precision_recall <- function(benchmark_table) {
  ggplot(benchmark_table, aes(x = Precision, y = Recall, color = Model, label = Model)) +
    geom_point(size = 4) +
    geom_text(vjust = -0.5) +
    labs(title = "Precision vs Recall", x = "Precision", y = "Recall") +
    theme_minimal()
}

precision_recall_plot <- plot_precision_recall(benchmark_table)
print(precision_recall_plot)


# ggsave("report/img/precision_vs_recall.png", plot = precision_recall_plot, width = 8, height = 6)

# Mean AUC Comparison
plot_mean_auc <- function(benchmark_table) {
  ggplot(benchmark_table, aes(x = Model, y = Mean_AUC, fill = Model)) +
    geom_bar(stat = "identity", color = "black") +
    labs(title = "Mean AUC Comparison", y = "Mean AUC", x = "Model") +
    theme_minimal()
}

mean_auc_plot <- plot_mean_auc(benchmark_table)
print(mean_auc_plot)


# ggsave("report/img/mean_auc_comparison.png", plot = mean_auc_plot, width = 8, height = 6)

# Log Loss Comparison
plot_log_loss <- function(benchmark_table) {
  ggplot(benchmark_table, aes(x = Model, y = Log_Loss, fill = Model)) +
    geom_bar(stat = "identity", color = "black") +
    labs(title = "Log Loss Comparison", y = "Log Loss", x = "Model") +
    theme_minimal()
}

log_loss_plot <- plot_log_loss(benchmark_table)
print(log_loss_plot)


# ggsave("report/img/log_loss_comparison.png", plot = log_loss_plot, width = 8, height = 6)



# Function: Log Loss Across Predictions (Line Plot)
plot_log_loss_predictions <- function(model_name, true_labels, predicted_probs, save_path) {
  log_loss_values <- -rowSums(model.matrix(~ factor(true_labels) - 1) * log(predicted_probs + 1e-15))
  log_loss_df <- data.frame(
    Prediction_Index = seq_along(log_loss_values),
    Log_Loss = log_loss_values
  )
  p <- ggplot(log_loss_df, aes(x = Prediction_Index, y = Log_Loss)) +
    geom_line(color = "blue") +
    labs(title = paste(model_name, "- Log Loss Across Predictions"),
         x = "Prediction Index", y = "Log Loss") +
    theme_minimal()
  ggsave(save_path, plot = p, width = 8, height = 6)
}

# Function: Log Loss by Class (Bar Plot)
plot_log_loss_by_class <- function(model_name, true_labels, predicted_probs, save_path) {
  true_labels_onehot <- model.matrix(~ factor(true_labels) - 1)
  log_loss_per_class <- colMeans(-true_labels_onehot * log(predicted_probs + 1e-15), na.rm = TRUE)
  log_loss_df <- data.frame(
    Class = colnames(predicted_probs),
    Log_Loss = log_loss_per_class
  )
  p <- ggplot(log_loss_df, aes(x = reorder(Class, Log_Loss), y = Log_Loss, fill = Class)) +
    geom_bar(stat = "identity", color = "black") +
    labs(title = paste(model_name, "- Log Loss by Class"),
         x = "Class", y = "Log Loss") +
    theme_minimal()
  ggsave(save_path, plot = p, width = 8, height = 6)
}

# Generate Plots for Each Model
plot_log_loss_predictions(
  "Logistic Regression", test_labels, logistic_probs,
  file.path(img_dir, "logistic_log_loss_predictions.png")
)
plot_log_loss_by_class(
  "Logistic Regression", test_labels, logistic_probs,
  file.path(img_dir, "logistic_log_loss_by_class.png")
)

plot_log_loss_predictions(
  "KNN", test_labels, knn_probs,
  file.path(img_dir, "knn_log_loss_predictions.png")
)
plot_log_loss_by_class(
  "KNN", test_labels, knn_probs,
  file.path(img_dir, "knn_log_loss_by_class.png")
)

plot_log_loss_predictions(
  "Random Forest", test_labels, rf_probs,
  file.path(img_dir, "rf_log_loss_predictions.png")
)
plot_log_loss_by_class(
  "Random Forest", test_labels, rf_probs,
  file.path(img_dir, "rf_log_loss_by_class.png")
)

plot_log_loss_predictions(
  "CART", test_labels, cart_probs,
  file.path(img_dir, "cart_log_loss_predictions.png")
)
plot_log_loss_by_class(
  "CART", test_labels, cart_probs,
  file.path(img_dir, "cart_log_loss_by_class.png")
)
