import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV

# Load data from CSV file
df = pd.read_csv('Louie\'s-Dataset/louie-data-combined.csv')

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

# Create base classifiers
logreg = LogisticRegression(C=1.0, solver='lbfgs', random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Create stacking classifier
estimators = [('logreg', logreg), ('rf', rf), ('gbm', gbm)]
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=logreg)

# Define hyperparameter grid for grid search
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10],
    'rf__n_estimators': [50, 100, 200],
    'gbm__n_estimators': [50, 100, 200],
}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(stacking_classifier, param_grid, cv=5)
grid_search.fit(X_train_resampled, y_train_resampled)

# Print best hyperparameter values
print("Best Hyperparameters:")
print(grid_search.best_params_)

# Fit the stacking classifier with best hyperparameter values
stacking_classifier_best = grid_search.best_estimator_
stacking_classifier_best.fit(X_train_resampled, y_train_resampled)

# Predict on test set
y_pred = stacking_classifier_best.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))




