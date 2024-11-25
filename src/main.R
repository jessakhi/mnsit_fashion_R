# Install and load all required libraries
required_packages <- c(
  "caret", "pROC", "ggplot2", "randomForest", "glmnet", "doParallel", "ranger", 
  "rpart", "xgboost", "dplyr", "data.table", "reshape2", "imager", "FactoMineR", 
  "gridExtra", "grid", "Rtsne", "corrplot", "heatmaply", "plotly", "factoextra", 
  "rstudioapi"
)

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
}

# Set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("..")


img_dir <- "report/img"
if (!dir.exists(img_dir)) dir.create(img_dir, recursive = TRUE)

# Source all project scripts in the correct order
source("data_cleaning.R")
source("data_preparation.R")
source("eda.R")
source("feature_egineering.R")
source("model_building.R")
source("model_eval.R")

cat("Pipeline executed successfully. Outputs are saved in designated folders.")
