

# **MNIST Fashion Dataset Classification Project**

## **Overview**
This project is part of the TEB2043/TFB2063/TEB2164 Data Science course at Universiti Teknologi PETRONAS. The goal is to classify clothing items from the MNIST Fashion dataset using machine learning models. The workflow demonstrates an end-to-end data science process, including data cleaning, preprocessing, exploratory analysis, modeling, and evaluation. 

The models used are **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Random Forest**, evaluated on metrics such as Accuracy, Precision, Recall, F1-Score, AUC, and Log Loss.

---

## **Dataset**
The **MNIST Fashion dataset** consists of grayscale images of 10 clothing categories, intended as a replacement for the traditional MNIST handwritten digits dataset.

### **Details**
- **Training Set**: 60,000 images with labels.
- **Test Set**: 10,000 images with labels.
- **Classes**:
  1. T-shirt/top
  2. Trouser
  3. Pullover
  4. Dress
  5. Coat
  6. Sandal
  7. Shirt
  8. Sneaker
  9. Bag
  10. Ankle boot

### **Source**
Available via [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist) or the `keras` R package.

---

## **Objectives**
1. **Data Preprocessing**:
   - Flatten 28x28 images into vectors for compatibility with models.
   - Normalize pixel values between 0 and 1.
   - Split the dataset into training, validation, and test sets.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize class distributions and pixel intensity patterns.
   - Generate heatmaps, histograms, and sample visualizations.

3. **Modeling**:
   - Train Logistic Regression, KNN, and Random Forest classifiers.
   - Optimize hyperparameters using grid search and cross-validation.

4. **Evaluation**:
   - Compare models using metrics:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - AUC (Area Under the Curve)
     - Log Loss
   - Analyze training and inference times.

5. **Collaboration**:
   - Develop teamwork and task division among group members.

---

## **Pipeline**
```plaintext
project/
├── src/
│   ├── data_cleaning.R        # Clean and preprocess raw data
│   ├── data_preparation.R     # Flatten and normalize images
│   ├── eda.R                  # Perform exploratory data analysis
│   ├── feature_engineering.R  # Engineer features (if applicable)
│   ├── model_building.R       # Train Logistic Regression, KNN, and Random Forest
│   ├── model_eval.R           # Evaluate models and benchmark results
│   └── main.R                 # Run the full pipeline
├── data/
│   ├── raw/                   # Raw dataset files
│   ├── processed/             # Cleaned and preprocessed data
│   └── prepared/              # Flattened and normalized data for models
├── report/
│   ├── report.tex             # Final report source (LaTeX)
│   ├── insights.pdf           # Key insights and findings
│   └── model_comparison.png   # Visualization of model performance
├── README.md                  # Project documentation
├── .gitignore                 # Files and directories to ignore in Git
└── DS.zip                     # Final submission package
```

---

## **How to Run the Project**
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd project
   ```
2. Install required R packages:
   ```R
   install.packages(c("keras", "ggplot2", "caret", "tidymodels", "data.table", "dplyr", "pROC", "randomForest", "reshape2"))
   ```
3. Execute the pipeline:
   ```R
   source("src/main.R")
   ```

---

## **Machine Learning Models**
### **1. Logistic Regression**
- A baseline linear classifier that predicts probabilities using softmax activation.
- Efficient and interpretable, serving as a benchmark.

### **2. K-Nearest Neighbors (KNN)**
- A non-parametric model that predicts based on the majority class among its `k` nearest neighbors.
- Hyperparameters:
  - `k`: Number of neighbors (optimized via grid search).

### **3. Random Forest**
- An ensemble model based on decision trees, handling high-dimensional data and providing feature importance.
- Hyperparameters:
  - `ntree`: Number of trees.
  - `mtry`: Number of features considered for splitting.

### **Evaluation Metrics**
- **Accuracy**: Overall correctness of predictions.
- **Precision**: Fraction of true positives among predicted positives.
- **Recall**: Fraction of true positives among actual positives.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **AUC**: Measures the ability of the model to distinguish between classes.
- **Log Loss**: Penalizes incorrect predictions based on their probability estimates.

---

## **Deliverables**
- **Report**: A detailed PDF summarizing data analysis, modeling, and results.
- **Codebase**: Organized R scripts implementing the workflow.
- **Insights**: Key visualizations and findings.

---

## **Team Members**
| Name                 | Student ID  |
|----------------------|-------------|
| Jihane Essakhi       | 24004461    |
| [Name 2]             | [ID 2]      |
| Syed Fahim Hussain   | 22009863    |
| [Name 4]             | [ID 4]      |

---

## **Acknowledgments**
The project uses the MNIST Fashion dataset provided by Zalando Research. We acknowledge the use of R libraries (`keras`, `ggplot2`, `caret`, etc.) and generative AI tools (ChatGPT, GitHub Copilot) for guidance and productivity enhancement.

