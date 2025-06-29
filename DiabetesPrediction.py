import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
df = pd.read_csv('/content/drive/MyDrive/ModelTrain/datasetForModel.csv')
# Assuming df is already loaded
print(df.head())

X = df.drop(['diabetes', 'gender', 'smoking_history'], axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
# Fit the model
model.fit(X_train, y_train)


# Visualize the decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'])
plt.show()

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# User input for prediction
print("\n--- Diabetes Prediction ---")
try:
    age = float(input("Enter age: "))
    hypertension = int(input("Enter hypertension (1 for Yes, 0 for No): "))
    heart_disease = int(input("Enter heart disease (1 for Yes, 0 for No): "))
    bmi = float(input("Enter BMI: "))
    HbA1c_level = float(input("Enter HbA1c level: "))
    blood_glucose_level = float(input("Enter blood glucose level: "))
    # Prepare the input as a DataFrame or array
    user_data = pd.DataFrame([[age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]],
                             columns=['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
    # Make prediction
    prediction = model.predict(user_data)
    # Display the result
    if prediction[0] == 1:
        print("\nThe model predicts: Diabetes")
    else:
        print("\nThe model predicts: No Diabetes")
except ValueError:
    print("Invalid input. Please enter numeric values.")