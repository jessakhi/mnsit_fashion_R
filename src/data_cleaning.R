# Data Cleaning ---------------------------------------------------------------
if (!require(dplyr)) install.packages("dplyr")
if (!require(caret)) install.packages("caret")
if (!require(data.table)) install.packages("data.table")
if (!require(rstudioapi)) install.packages("rstudioapi")

library(dplyr)
library(caret)
library(data.table)
library(rstudioapi)

#------------------------------------------------------------------------------

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # Set to src/
setwd("..") 


train_data <- fread("data/raw/fashion-mnist_train.csv")
test_data <- fread("data/raw/fashion-mnist_test.csv")

#------------------------------------------------------------------------------

train_data <- train_data %>% distinct()
test_data <- test_data %>% distinct()

#------------------------------------------------------------------------------


train_data <- as.data.frame(lapply(train_data, function(x) as.numeric(x) / 255))
test_data <- as.data.frame(lapply(test_data, function(x) as.numeric(x) / 255))
gc()

#------------------------------------------------------------------------------


set.seed(123)
train_index <- createDataPartition(train_data$label, p = 0.8, list = FALSE)
train_split <- train_data[train_index, ]
val_split <- train_data[-train_index, ]
gc()

#------------------------------------------------------------------------------

preprocessed_dir <- file.path(script_path, "../data/processed")
dir.create(preprocessed_dir, recursive = TRUE, showWarnings = FALSE)

write.csv(val_split, file.path(preprocessed_dir, "train_data.csv"), row.names = FALSE)
write.csv(train_split, file.path(preprocessed_dir, "train_split.csv"), row.names = FALSE)
write.csv(val_split, file.path(preprocessed_dir, "val_split.csv"), row.names = FALSE)
write.csv(test_data, file.path(preprocessed_dir, "test_data.csv"), row.names = FALSE)

cat("Data preprocessing completed. Processed files are saved in 'data/preprocessed'.\n")
gc()

#------------------------------------------------------------------------------