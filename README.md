# heart-disease-prediction-random-forest
 This project builds a Random Forest Classifier to predict heart disease presence from clinical features. The dataset contains patient attributes such as age, sex, chest pain type, resting blood pressure, cholesterol, maximum heart rate, ECG results, exercise-induced angina, old peak, and other health indicators.
Project Workflow
Data Loading & Inspection
Load the CSV dataset and check data types and missing values.
Exploratory Data Analysis (EDA)
Analyze the target distribution and feature patterns.
Visualize relationships between features and the target.
Feature Engineering
Prepare input features (X) and target label (y).
Convert the dataset into a format suitable for machine learning.
Train/Test Split
Split the data into training and testing sets using stratification.
Model Training
Train a Random Forest model with default parameters.
Train a tuned Random Forest using GridSearchCV.
Model Evaluation
Evaluate results using:
Confusion Matrix
Classification Report (precision, recall, F1-score)
Feature Importance
Visualize the most important features influencing predictions.
Conclusion
Compare default vs tuned model performance and summarize key findings.
Why Random Forest?
Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting. It can capture non-linear relationships and provides feature importance insights.

Results
The project compares model performance using classification metrics such as:

Accuracy
Precision, Recall, F1-score
Confusion matrix analysis
