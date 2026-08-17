Machine Learning Classification Assignment 2
1. Problem Statement
The objective of this project is to implement and compare multiple machine learning classification models on a common classification dataset.
The implemented models are:
1.	Logistic Regression
2.	Decision Tree Classifier
3.	K-Nearest Neighbor (kNN) Classifier
4.	Gaussian Naive Bayes
5.	Random Forest Classifier
The models are evaluated using the following performance metrics:
•	Accuracy
•	AUC Score
•	Precision
•	Recall
•	F1 Score
•	Matthews Correlation Coefficient (MCC)
An interactive Streamlit web application is also developed to allow users to upload test data, select a classification model, and view the corresponding evaluation results.
________________________________________
2. Dataset Description
Dataset
The project uses the Breast Cancer Wisconsin (Diagnostic) dataset.
The dataset is a binary classification dataset containing measurements computed from digitized images of breast mass samples.
Dataset Size
•	Number of instances: 569
•	Number of features: 30
•	Number of classes: 2
The dataset therefore satisfies the assignment requirements of a minimum of 500 instances and 12 features.
Target Classes
The two target classes are:
•	Malignant
•	Benign
Data Preparation
The dataset was divided into training and testing sets using an 80:20 split.
Training data: 80%
Testing data: 20%
A fixed random state of 42 was used to make the experiment reproducible.
Feature scaling using StandardScaler was applied to models where feature magnitude can affect model performance, particularly Logistic Regression and kNN.
________________________________________
3. GitHub Repository Link
GitHub Repository:
Replace this line with your actual GitHub repository URL after creating the repository.
<YOUR_GITHUB_REPOSITORY_LINK>
________________________________________
4. Models Used
4.1 Logistic Regression
Logistic Regression is a linear classification algorithm used to estimate the probability of a binary class.
Standardization was applied before training the Logistic Regression model.
________________________________________
4.2 Decision Tree Classifier
The Decision Tree Classifier creates a tree-like structure by recursively splitting the data based on feature values.
The model was trained using the training dataset and evaluated using the test dataset.
________________________________________
4.3 K-Nearest Neighbor (kNN)
The kNN classifier predicts the class of a sample based on the classes of its nearest neighbors.
A value of:
k = 5
was used.
Feature standardization was applied before kNN classification.
________________________________________
4.4 Gaussian Naive Bayes
Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes’ theorem.
The Gaussian version was selected because the input features are continuous numerical measurements.
________________________________________
4.5 Random Forest
Random Forest is an ensemble learning method that combines multiple decision trees to produce a more robust classification model.
The implementation uses:
Number of trees = 200
Random state = 42
________________________________________
5. Evaluation Metrics
The following metrics were calculated for every model.
Accuracy
Accuracy measures the proportion of correctly classified samples among all samples.
AUC
AUC measures the ability of the classifier to distinguish between the two classes across different classification thresholds.
Precision
Precision measures the proportion of predicted positive samples that are actually positive.
Recall
Recall measures the proportion of actual positive samples that are correctly identified.
F1 Score
F1 Score is the harmonic mean of precision and recall.
Matthews Correlation Coefficient (MCC)
MCC measures the quality of binary classifications while taking all four confusion-matrix categories into account.
________________________________________
6. Model Comparison
The following table contains the evaluation results obtained from the experiments.
Replace the ... values below with the actual values from model_results.csv.
ML Model Name	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	…	…	…	…	…	…
Decision Tree	…	…	…	…	…	…
kNN	…	…	…	…	…	…
Naive Bayes	…	…	…	…	…	…
Random Forest	…	…	…	…	…	…
________________________________________
7. Observations on Model Performance
Logistic Regression
Logistic Regression provides a strong baseline for the classification problem. Since the model is based on a linear decision boundary, its performance depends on how well the classes can be separated using a linear relationship between the features.
Observation: Replace this statement with an observation based on the actual metric values obtained from the experiment.
________________________________________
Decision Tree
The Decision Tree model can capture nonlinear relationships between the input features. However, individual decision trees can be sensitive to the training data and may overfit.
Observation: Replace this statement with an observation based on the actual metric values obtained from the experiment.
________________________________________
kNN
The kNN classifier uses neighboring observations to determine the predicted class. Because distance calculations are involved, feature scaling is important.
Observation: Replace this statement with an observation based on the actual metric values obtained from the experiment.
________________________________________
Naive Bayes
Gaussian Naive Bayes is computationally efficient and works by modelling the probability distributions of the features for each class.
Observation: Replace this statement with an observation based on the actual metric values obtained from the experiment.
________________________________________
Random Forest
Random Forest combines multiple decision trees and generally provides better robustness than a single decision tree.
Observation: Replace this statement with an observation based on the actual metric values obtained from the experiment.
________________________________________
8. Overall Winner
The overall winner should be selected after comparing the Accuracy, AUC, Precision, Recall, F1 Score and MCC values.
Overall Winner:
Replace with the model that performs best based on your actual experimental results.
The final selection should be justified using the evaluation metrics rather than selecting a model without considering the results.
________________________________________
9. Streamlit Application
An interactive Streamlit application was developed for the trained classification models.
The application provides the following functionality:
•	Upload test data in CSV format
•	Select a machine learning model
•	Display Accuracy
•	Display AUC
•	Display Precision
•	Display Recall
•	Display F1 Score
•	Display MCC
•	Display confusion matrix
•	Display classification report
•	Display predictions
Models Available in the Application
•	Logistic Regression
•	Decision Tree
•	kNN
•	Naive Bayes
•	Random Forest
________________________________________
10. Project Structure
ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_results.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── metadata.pkl
________________________________________
11. How to Run the Project Locally
Step 1: Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_LINK>
Step 2: Open the project
cd ML_Assignment_2
Step 3: Install dependencies
pip install -r requirements.txt
Step 4: Run the Streamlit application
streamlit run app.py
Step 5: Open the application
Open:
http://localhost:8501
________________________________________
12. Streamlit Deployment
The application is intended to be deployed using Streamlit Community Cloud.
Live Streamlit App
Replace this line with your actual deployed Streamlit URL.
<YOUR_STREAMLIT_APP_LINK>
________________________________________
13. Test Data
The file:
test_data.csv
contains the test dataset used to evaluate the trained classification models.
The CSV contains the required feature columns and the target column.
________________________________________
14. Reproducibility
The experiments use:
random_state = 42
where applicable.
This allows the train-test split and Random Forest experiment to be reproduced consistently.
________________________________________
15. Technologies Used
•	Python
•	Pandas
•	NumPy
•	Scikit-learn
•	Joblib
•	Streamlit
•	Matplotlib
•	Seaborn
________________________________________
16. Conclusion
This project implements and compares multiple machine learning classification algorithms using a common dataset.
The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score and MCC.
The Streamlit application provides an interactive interface for uploading test data, selecting a model and viewing its classification performance.
The final model selection is based on the actual experimental results obtained from the evaluation metrics.
