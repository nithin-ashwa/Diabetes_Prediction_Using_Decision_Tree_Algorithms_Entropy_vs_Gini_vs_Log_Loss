🩺 Diabetes Prediction using Decision Tree Classifier
This repository demonstrates a simple implementation of Decision Tree classifiers to predict the presence of diabetes using the Pima Indians Diabetes Dataset. The code compares different splitting criteria (entropy, gini, and log_loss) and evaluates model performance.

📁 Dataset
File: diabetes.csv

Source: UCI Machine Learning Repository

Features:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Outcome (target variable)

🧠 Algorithms Used
Decision Tree Classifier from scikit-learn

Comparison of criteria:

Entropy (Information Gain)

Gini Index

Log Loss (Cross-Entropy)

🛠️ Libraries Used
python
Copy
Edit
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import graphviz
🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/diabetes-decision-tree.git
cd diabetes-decision-tree
Install the required libraries:

bash
Copy
Edit
pip install pandas numpy scikit-learn graphviz
Run the notebook or script. Ensure the dataset (diabetes.csv) is in the correct path or update the path accordingly:

python
Copy
Edit
data = pd.read_csv("/content/drive/MyDrive/Dataset ML/diabetes.csv")
📊 Results
Models trained and tested using a 90-10 split

Accuracy is computed and compared across criteria

A classification report is printed for detailed evaluation

Decision trees are visualized using plot_tree and graphviz

🖼️ Example Visualization
python
Copy
Edit
from sklearn import tree
tree.plot_tree(model)

# Graphviz visualization
graphviz.Source(export_graphviz(model, feature_names=x.columns, filled=True))
📌 Conclusion
This project showcases how decision trees can be applied for medical diagnosis tasks and how model performance may vary depending on the splitting criterion.

📜 License
This project is open-source and available under the MIT License.

