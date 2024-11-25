if (!require(dplyr)) install.packages("dplyr")
if (!require(data.table)) install.packages("data.table")

library(dplyr)
library(data.table)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # Set to src/
setwd("..")

# Load processed data
train_split <- fread("data/processed/train_split.csv")
val_split <- fread("data/processed/val_split.csv")
test_data <- fread("data/processed/test_data.csv")

# Flatten image data
flatten_images <- function(data) {
  labels <- data$label
  flattened <- data %>% select(-label)
  flattened$label <- labels
  return(flattened)
}

train_split_flattened <- flatten_images(train_split)
val_split_flattened <- flatten_images(val_split)
test_data_flattened <- flatten_images(test_data)

gc()

# Save flattened data
prepared_dir <- file.path("data", "prepared")
dir.create(prepared_dir, recursive = TRUE, showWarnings = FALSE)

write.csv(train_split_flattened, file.path(prepared_dir, "train_split_flattened.csv"), row.names = FALSE)
write.csv(val_split_flattened, file.path(prepared_dir, "val_split_flattened.csv"), row.names = FALSE)
write.csv(test_data_flattened, file.path(prepared_dir, "test_data_flattened.csv"), row.names = FALSE)

cat("Data preparation completed. Flattened data saved in 'data/prepared'.\n")
gc()
rm(list = ls())
