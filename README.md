# Liver Patient Prediction 🧬

This project focuses on predicting whether a patient is suffering from a liver disease using machine learning techniques. The prediction is based on various biological and demographic features collected from patient records.

## 📁 Project Structure

* `Liver_Patient_Prediction.ipynb` – Main Jupyter Notebook for:

  * Data loading
  * Exploratory Data Analysis (EDA)
  * Data preprocessing
  * Model training & evaluation

## 📊 Dataset

The dataset contains liver patient records with the following attributes:

* Age
* Gender
* Total Bilirubin
* Direct Bilirubin
* Alkaline Phosphotase
* Alamine Aminotransferase (ALT)
* Aspartate Aminotransferase (AST)
* Total Proteins
* Albumin
* Albumin and Globulin Ratio
* Target column: `Dataset` (1 = Liver patient, 2 = Non-liver patient)

> Note: This dataset is publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29).

## 🔍 Key Steps

### ✅ Data Preprocessing

* Missing value treatment (e.g., imputing Albumin and Globulin Ratio)
* Encoding categorical variables (e.g., `Gender`)
* Feature scaling

### 📈 Exploratory Data Analysis

* Distribution plots
* Correlation heatmap
* Feature relationships with the target

### 🧠 Model Training

Several ML algorithms were implemented:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

### 📊 Model Evaluation

Metrics used for model performance:

* Accuracy
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

## 🏆 Best Model

The best-performing model (based on evaluation metrics) is:

* **Random Forest Classifier**

## 🛠 Requirements

Install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 🚀 How to Run

1. Clone the repository or download the notebook.
2. Open `Liver_Patient_Prediction.ipynb` in Jupyter Notebook.
3. Run each cell sequentially to reproduce results and train models.

## 📌 Conclusion

This project provides a practical approach to healthcare-related ML classification problems. With proper tuning and feature selection, the prediction accuracy can be improved further for real-world deployment.


