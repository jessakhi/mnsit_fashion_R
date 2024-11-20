# Main Script for Running the Project Pipeline
# -----------------------------------------------------------------------------

if (!require(data.table)) install.packages("data.table")
if (!require(rstudioapi)) install.packages("rstudioapi")

library(data.table)
library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("..")

# ----------------------------------------------------------------------------- 
# Step 1: Data Cleaning
cat("\nStep 1: Data Cleaning\n")
source("src/data_cleaning.R")
cat("\nData cleaning completed successfully!\n")

# ----------------------------------------------------------------------------- 
# Step 2: Exploratory Data Analysis (EDA)
cat("\nStep 2: Exploratory Data Analysis\n")
source("src/eda.R")
cat("\nEDA completed successfully!\n")

# ----------------------------------------------------------------------------- 
# Step 3: Feature Engineering
cat("\nStep 3: Feature Engineering\n")
source("src/feature_engineering.R")
cat("\nFeature engineering completed successfully!\n")

# ----------------------------------------------------------------------------- 
# Step 4: Model Building
cat("\nStep 4: Model Building\n")
source("src/model_building.R")
cat("\nModel building completed successfully!\n")

# ----------------------------------------------------------------------------- 
# Step 5: Model Evaluation
cat("\nStep 5: Model Evaluation\n")
source("src/model_eval.R")
cat("\nModel evaluation completed successfully!\n")

# ----------------------------------------------------------------------------- 
cat("\nPipeline execution completed successfully!\n")
