#---------------------------------------------------------------------------
if (!require(caret)) install.packages("caret")
if (!require(pROC)) install.packages("pROC")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(randomForest)) install.packages("randomForest")

library(caret)
library(pROC)
library(ggplot2)
library(randomForest)
#---------------------------------------------------------------------------
# Load models, test data, and processing times
load("trained_models.RData") # Assumes 'logistic_model', 'knn_model', 'rf_model', 'processing_times'
load("mnist_data.RData")     # Assumes 'test_data', 'test_labels'

#---------------------------------------------------------------------------

benchmark_models <- function(model_name, true_labels, predicted_probs, predicted_labels, processing_time) {
  cm <- confusionMatrix(factor(predicted_labels), factor(true_labels))
  accuracy <- cm$overall['Accuracy']
  precision <- cm$byClass['Precision']
  recall <- cm$byClass['Recall']
  f1_score <- 2 * (precision * recall) / (precision + recall)
  roc_curve <- roc(true_labels, predicted_probs)
  auc_value <- auc(roc_curve)
  log_loss <- -mean(true_labels * log(predicted_probs) + (1 - true_labels) * log(1 - predicted_probs))
  
  plot(roc_curve, main = paste("ROC Curve -", model_name), col = "blue")
  
  return(list(
    model = model_name,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    auc = auc_value,
    log_loss = log_loss,
    processing_time = processing_time
  ))
}

#---------------------------------------------------------------------------

#  Logistic Regression
logistic_probs <- predict(logistic_model, test_data, type = "response")
logistic_labels <- ifelse(logistic_probs > 0.5, 1, 0)
logistic_eval <- benchmark_models("Logistic Regression", test_labels, logistic_probs, logistic_labels, processing_times$logistic_model)

#  KNN
knn_labels <- predict(knn_model, test_data)
knn_probs <- predict(knn_model, test_data, type = "prob")[, 2]
knn_eval <- benchmark_models("KNN", test_labels, knn_probs, knn_labels, processing_times$knn_model)

#  Random Forest
rf_probs <- predict(rf_model, test_data, type = "prob")[, 2]
rf_labels <- predict(rf_model, test_data)
rf_eval <- benchmark_models("Random Forest", test_labels, rf_probs, rf_labels, processing_times$rf_model)

#---------------------------------------------------------------------------


# Summary table
benchmark_table <- data.frame(
  Model = c("Logistic Regression", "KNN", "Random Forest"),
  Accuracy = c(logistic_eval$accuracy, knn_eval$accuracy, rf_eval$accuracy),
  Precision = c(logistic_eval$precision, knn_eval$precision, rf_eval$precision),
  Recall = c(logistic_eval$recall, knn_eval$recall, rf_eval$recall),
  F1_Score = c(logistic_eval$f1_score, knn_eval$f1_score, rf_eval$f1_score),
  AUC = c(logistic_eval$auc, knn_eval$auc, rf_eval$auc),
  Log_Loss = c(logistic_eval$log_loss, knn_eval$log_loss, rf_eval$log_loss),
  Processing_Time = c(logistic_eval$processing_time, knn_eval$processing_time, rf_eval$processing_time)
)

print(benchmark_table)
write.csv(benchmark_table, file = "model_benchmark_results.csv", row.names = FALSE)
