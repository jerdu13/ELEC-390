import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# Load data from CSV file
df = pd.read_csv('Louie\'s-Dataset/louie-data-combined.csv')

# Extract features and target variable
X = df.iloc[:, :6]  # Features are the first 6 columns
y = df.iloc[:, 6]   # Target variable is the 7th column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing and logistic regression
pipeline = make_pipeline(
    StandardScaler(),        # Standardize features
    LogisticRegression()    # Logistic Regression model
)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Predict on the testing data
y_pred = pipeline.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot predicted classes
plt.figure(figsize=(8, 6))
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='coolwarm')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Predicted Classes (0: Walking, 1: Jumping)')
plt.show()

