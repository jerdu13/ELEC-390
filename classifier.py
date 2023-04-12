import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.preprocessing import PolynomialFeatures

# Load data from CSV file
df = pd.read_csv('ryans-dataset/ryan-data-combined.csv')

# Extract features and target variable
X = df.iloc[:, :6]  # Features are the first 6 columns
y = df.iloc[:, 6]   # Target variable is the 7th column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature scaling using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perform class resampling to address class imbalance
X_train_resampled, y_train_resampled = resample(X_train[y_train == 0], y_train[y_train == 0],
                                                replace=True, n_samples=sum(y_train == 1), random_state=42)
X_train_resampled = pd.DataFrame(X_train_resampled, columns=X.columns)
X_train_resampled = pd.concat([X_train_resampled, pd.DataFrame(X_train[y_train == 1], columns=X.columns)])
y_train_resampled = pd.concat([y_train_resampled, y_train[y_train == 1]])

# Create a pipeline for preprocessing and logistic regression
pipeline = make_pipeline(
    PolynomialFeatures(degree=2),  # Add polynomial features with degree 2
    LogisticRegression(solver='liblinear', max_iter=1000)  # Logistic Regression model with solver='liblinear' and max_iter=1000
)

# Perform cross-validation
scores = cross_val_score(pipeline, X_train_resampled, y_train_resampled, cv=5)
print("Cross-validated accuracy:", scores.mean())

# Fit the pipeline to the training data
pipeline.fit(X_train_resampled, y_train_resampled)

# Predict on the testing data
y_pred = pipeline.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot predicted classes
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Predicted Classes (0: Walking, 1: Jumping)')
plt.show()


