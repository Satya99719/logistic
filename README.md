# logistic
classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset (replace 'your_dataset.csv' with your actual file path)
# Ensure your dataset has a column for the target variable (e.g., 'Class' for breast_cancer dataset)
data = pd.read_csv('/content/breast_cancer.csv')

# Define features (X) and target variable (y)
# Assuming 'Class' is the target variable for breast_cancer dataset
X = data.drop('Class', axis=1)  # Features (all columns except 'Class')
y = data['Class']  # Target variable ('Class')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train, 20% test

# Initialize and train a Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000)  # You can adjust parameters as needed
logreg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
