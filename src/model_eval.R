if (!require(caret)) install.packages("caret")
if (!require(pROC)) install.packages("pROC")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(randomForest)) install.packages("randomForest")

library(caret)
library(pROC)
library(ggplot2)
library(randomForest)

gc()
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("..")

load("models/trained_models.RData")
load("models/mnist_data.RData")

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
  mean_auc <- mean(auc_values, na.rm = TRUE)
  plot(roc_curves[[1]], col = "blue", main = paste("ROC Curves -", model_name), legacy.axes = TRUE)
  for (i in 2:length(roc_curves)) {
    plot(roc_curves[[i]], col = i, add = TRUE)
  }
  legend("bottomright", legend = colnames(predicted_probs), col = 1:ncol(predicted_probs), lty = 1)
  cm_table <- as.data.frame(cm$table)
  ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "blue") +
    ggtitle(paste("Confusion Matrix -", model_name)) +
    theme_minimal() +
    labs(x = "True Labels", y = "Predicted Labels")
  return(list(
    model = model_name,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    mean_auc = mean_auc,
    log_loss = log_loss,
    processing_time = processing_time
  ))
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

ggplot(benchmark_table, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", color = "black") +
  ggtitle("Model Accuracy Comparison") +
  theme_minimal() +
  labs(y = "Accuracy", x = "Model")

print(benchmark_table)
write.csv(benchmark_table, file = "model_benchmark_results.csv", row.names = FALSE)

View(benchmark_table)