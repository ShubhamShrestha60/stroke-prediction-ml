# Big Data Analytics for Early Stroke Prediction Using Machine Learning

This project presents a machine learning-based framework for early prediction of stroke using real-world clinical and demographic data. It integrates explainable AI and advanced data balancing techniques to enhance model interpretability and fairness in predictions—crucial in healthcare applications.

## 🧠 Objective

To build a robust and interpretable stroke prediction model using:
- **Random Forest** algorithm
- **SMOTE + Undersampling** to address class imbalance
- **SHAP values** for feature importance and transparency
- **Stratified K-Fold Cross Validation** for robust evaluation

## 📊 Dataset

- Source: Kaggle's healthcare dataset
- Total records: 5110
- Features include: Age, Gender, Hypertension, Heart Disease, BMI, Glucose level, Smoking status, etc.
- Target variable: Stroke (binary classification)

## 🔄 Methodology

The methodology is broken into 7 core phases:

1. **Data Collection**: Publicly available dataset with real clinical features
2. **Preprocessing**:
   - Imputed missing values (e.g., BMI)
   - Removed irrelevant features (e.g., ID)
   - Applied one-hot encoding and normalization
3. **Handling Class Imbalance**:
   - Used **SMOTE** and **Random Undersampling** for balanced classes
4. **Model Development**:
   - Trained **Random Forest** as baseline and improved model
5. **Explainability with SHAP**:
   - Identified most influential features: Age, Glucose, BMI
6. **Hyperparameter Tuning**:
   - Used `RandomizedSearchCV` with 5-fold cross-validation
7. **Evaluation**:
   - Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC

## 📈 Results

| Model         | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|---------------|----------|-----------|--------|----------|---------|
| Baseline (Imbalanced) | 94% | 0.00      | 0.00   | 0.00     | 0.79    |
| SMOTE Model   | 92%      | 0.91      | 0.93   | 0.92     | 0.976   |

- Balanced dataset drastically improved stroke case detection.
- SHAP confirmed medically relevant features.
- Final model offers strong interpretability + clinical validity.

## 🛠️ Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- SHAP
- Matplotlib, Seaborn
