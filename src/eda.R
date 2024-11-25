
#-------------------------------------------------------------------


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("..")
train_data <- fread("data/raw/fashion-mnist_train.csv")
test_data <- fread("data/raw/fashion-mnist_test.csv")
gc()

objects <- c('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
#-------------------------------------------------------------------

#Viewing images

options(repr.plot.width = 10, repr.plot.height = 10)
par(mfcol = c(10, 10))
par(mar = c(0, 0, 1.5, 0), xaxs = 'i', yaxs = 'i')

for (i in 1:30) {
  img <- as.numeric(train_data[i, -1]) / 255
  img_matrix <- matrix(img, nrow = 28, ncol = 28, byrow = TRUE)
  img_matrix <- t(apply(img_matrix, 2, rev))
  label <- objects[as.numeric(train_data[i, 1]) + 1]
  image(1:28, 1:28, img_matrix, col = gray((0:255) / 255), xaxt = 'n', yaxt = 'n', main = label, cex.main = 0.8)
}

#-------------------------------------------------------------------

# See dimensions
cat("Dimensions of train data: ", dim(train_data), "\n")
cat("Dimensions of test data: ", dim(test_data), "\n")

#-------------------------------------------------------------------

# Check for Missing Values
na_count_train <- sum(is.na(train_data))
na_count_test <- sum(is.na(test_data))
cat("Number of missing values in train data:", na_count_train, "\n")
cat("Number of missing values in test data:", na_count_test, "\n")

#-------------------------------------------------------------------

# Check for Duplicate Rows
duplicate_count_train <- nrow(train_data) - nrow(unique(train_data))
duplicate_count_test <- nrow(test_data) - nrow(unique(test_data))
cat("Number of duplicate rows in train data:", duplicate_count_train, "\n")
cat("Number of duplicate rows in test data:", duplicate_count_test, "\n")

#-------------------------------------------------------------------

# Class Distribution
class_distribution <- train_data %>%
  count(label) %>%
  mutate(percentage = n / sum(n) * 100)

print(class_distribution)

ggplot(class_distribution, aes(x = factor(label, labels = objects), y = percentage)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Class Distribution in Train Data", x = "Class", y = "Percentage") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#-------------------------------------------------------------------


# Pixel Intensity Distribution
pixel_values <- as.numeric(unlist(train_data[, -1])) 
ggplot(data = data.frame(PixelValue = pixel_values), aes(x = PixelValue)) +
  geom_histogram(bins = 30, fill = "darkorange", color = "black") +
  labs(title = "Pixel Intensity Distribution", x = "Pixel Value", y = "Frequency") +
  theme_minimal()


#-------------------------------------------------------------------

# PCA

pca_data <- train_data[, -1]
pca_result <- PCA(as.matrix(pca_data), graph = FALSE)
pca_df <- as.data.frame(pca_result$ind$coord)
pca_df$label <- train_data$label

p <- ggplot(pca_df, aes(x = Dim.1, y = Dim.2, color = factor(label))) +
  geom_point(alpha = 0.7) +
  labs(title = "PCA of MNIST Fashion Dataset", x = "PC1", y = "PC2") +
  theme_minimal()

ggplotly(p)

#-------------------------------------------------------------------
# Variance explained

pca_data <- as.matrix(train_data[, -1])
pca_data <- pca_data[, apply(pca_data, 2, var) > 0]
pca_result <- prcomp(pca_data, scale. = TRUE)

explained_variance <- (pca_result$sdev)^2
explained_variance_ratio <- explained_variance / sum(explained_variance)
cumulative_variance <- cumsum(explained_variance_ratio)

variance_df <- data.frame(
  PC = seq_along(explained_variance_ratio),
  ExplainedVariance = explained_variance_ratio,
  CumulativeVariance = cumulative_variance
)

p <- ggplot(variance_df, aes(x = PC)) +
  geom_bar(aes(y = ExplainedVariance), stat = "identity", fill = "steelblue", alpha = 0.7) +
  geom_line(aes(y = CumulativeVariance), color = "red", size = 1) +
  geom_point(aes(y = CumulativeVariance), color = "red", size = 2) +
  labs(
    title = "Variance Explained by Principal Components (Full Dataset)",
    x = "Principal Component",
    y = "Variance Explained",
    caption = "Bar: Individual PCs, Line: Cumulative Variance"
  ) +
  theme_minimal()

ggplotly(p)


#-------------------------------------------------------------------

#t-SNE

train_data <- unique(train_data)
set.seed(42)
sampled_data <- train_data[sample(.N, 5000)]
labels <- as.factor(sampled_data[[1]])
features <- as.matrix(sampled_data[, -1])

set.seed(42)
tsne_result <- Rtsne(features, dims = 2, perplexity = 30, verbose = FALSE, max_iter = 500)

tsne_df <- data.frame(tsne_result$Y, label = objects[as.numeric(labels)])
colnames(tsne_df) <- c("x", "y", "label")
ggplot(tsne_df, aes(x = x, y = y, color = label)) +
  geom_point(alpha = 0.7) +
  labs(title = "t-SNE Visualization of Fashion-MNIST (Sampled Data)", x = "t-SNE 1", y = "t-SNE 2") +
  theme_minimal()


#-------------------------------------------------------------------

#Average image

options(repr.plot.width = 10, repr.plot.height = 10)
par(mfcol = c(2, 5))
par(mar = c(0, 0, 1.5, 0), xaxs = 'i', yaxs = 'i')

average_images <- train_data %>%
  group_by(label) %>%
  summarise(across(everything(), mean))
for (i in 1:nrow(average_images)) {
  avg_img <- as.numeric(average_images[i, -1]) / 255 
  avg_matrix <- matrix(avg_img, nrow = 28, ncol = 28, byrow = TRUE)
  avg_matrix <- t(apply(avg_matrix, 2, rev)) 
  label <- objects[as.numeric(average_images[i, 1]) + 1]
  image(1:28, 1:28, avg_matrix, col = gray((0:255) / 255), xaxt = 'n', yaxt = 'n', main = label, cex.main = 0.8)
}

rm(list = ls())