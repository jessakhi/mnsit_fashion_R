
if (!require(dplyr)) install.packages("dplyr")
if (!require(data.table)) install.packages("data.table")

library(dplyr)
library(data.table)

#------------------------------------------------------------------------------#


setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # Set to src/
setwd("..")


train_split <- fread("data/processed/train_split.csv")
val_split <- fread("data/processed/val_split.csv")
test_data <- fread("data/processed/test_data.csv")

#------------------------------------------------------------------------------#


flatten_images <- function(data) {
  labels <- data$label
  flattened <- as.data.frame(data %>% select(-label)) # Select all except the label
  flattened$label <- labels # Reattach the label column
  return(flattened)
}

train_split_flattened <- flatten_images(train_split)
val_split_flattened <- flatten_images(val_split)
test_data_flattened <- flatten_images(test_data)

gc()
#------------------------------------------------------------------------------#


processed_dir <- "data/prepared"
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)

write.csv(train_split_flattened, file.path(processed_dir, "train_split_flattened.csv"), row.names = FALSE)
write.csv(val_split_flattened, file.path(processed_dir, "val_split_flattened.csv"), row.names = FALSE)
write.csv(test_data_flattened, file.path(processed_dir, "test_data_flattened.csv"), row.names = FALSE)

cat("Data preparation completed. Flattened data saved in 'data/prepared'.\n")
