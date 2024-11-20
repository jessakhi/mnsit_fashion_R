
# **MNIST Fashion Dataset Project**

## **Overview**
This project is part of the TEB2043/TFB2063/TEB2164 Data Science course at Universiti Teknologi PETRONAS. The aim is to apply data science techniques using R programming to analyze and build predictive models based on the MNIST Fashion dataset. The project showcases the end-to-end data science workflow, including data cleaning, exploratory data analysis (EDA), feature engineering, model building, and evaluation.

## **Dataset Description**
The **MNIST Fashion dataset** is a replacement for the traditional MNIST handwritten digits dataset. It contains grayscale images of 10 different clothing categories, each represented as a 28x28 pixel image. 

### **Dataset Details**
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

### **Dataset Source**
The dataset can be downloaded using the `keras` library in R or from [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist).

---

## **Project Objectives**
1. **Data Analysis**:
   - Perform data cleaning and preprocessing.
   - Visualize the dataset to gain insights into class distribution and pixel intensity patterns.
2. **Model Development**:
   - Build and evaluate machine learning models for image classification.
   - Compare performance using metrics like accuracy, precision, and recall.
3. **Collaboration**:
   - Develop teamwork skills by dividing tasks effectively among group members.
4. **Critical Thinking**:
   - Reflect on the role of generative AI tools in data science projects.

---

## **Project Structure**
```plaintext
project/
├── src/                     # Source code
│   ├── data_cleaning.R          
│   ├── eda.R  
│   ├── feature_engineering.R               
│   ├── model_building.R     
│   ├── model_eval.R
│   └── main.R               # Script to run the entire pipeline
├── data/                    # Dataset storage
│   ├── raw/                 
│   └── processed/           
├── report/                 # Final report and outputs
│   ├── report.tex          
│      
├── README.md                
├── .gitignore              
└── DS.zip       # Final submission package
```

---

## **How to Run the Project**
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd project
   ```
2. Ensure the required R packages are installed:
   ```R
   install.packages(c("keras", "ggplot2", "caret", "tidymodels", "rmarkdown"))
   ```
3. Execute the main pipeline:
   ```R
   source("src/main.R")
   ```


---

## **Deliverables**
- **Report**: A comprehensive PDF document detailing the project findings.


---

## **Team Members**
| Name             | Student ID  |
|------------------|-------------|
| Jihane Essakhi        | 24004461      |
| [Name 2]         | [ID 2]      |
| [Name 3]         | [ID 3]      |
| [Name 4]         | [ID 4]      |

---

## **Acknowledgments**
This project uses the MNIST Fashion dataset provided by Zalando Research. We also acknowledge the use of open-source R libraries and generative AI tools like ChatGPT and GitHub Copilot for guidance and productivity enhancement.

