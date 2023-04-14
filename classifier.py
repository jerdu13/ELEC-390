import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.preprocessing import PolynomialFeatures
# for visualizing learning curve
from sklearn.model_selection import LearningCurveDisplay, learning_curve
# for visualizing confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
# for visualizing decision boundary
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA

### ORIGINAL TRAIN-TEST splitting (incorrect?) ###
# # Load data from CSV file
# df = pd.read_csv('shuffling-splitting-ready\\normalized-ryan-data-combined.csv')

# # Extract features and target variable
# X = df.iloc[:, 1:4]  # Features are the first 6 columns
# y = df.iloc[:, 5]   # Target variable is the 7th column

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


### ADDED FROM train-test.py -> will delete and just import hdf5 eventually ###
dset1 = pd.read_csv("shuffling-splitting-ready\\normalized-jeremy-data-combined.csv")
dset2 = pd.read_csv("shuffling-splitting-ready\\normalized-louie-data-combined.csv")
dset3 = pd.read_csv("shuffling-splitting-ready\\normalized-ryan-data-combined.csv")

pieces = [dset1, dset2, dset3]
dset = pd.concat(pieces)

sampling_rate = 100     # 100 data points collected each second by Phyphox
window_size = 5*sampling_rate
ratio = 0.9

rows = dset.shape[0] # number of rows in combined csv
num_windows = int(rows/window_size)

# initialize np array for segmented data
segments = np.zeros((num_windows, window_size, dset.shape[1])) # 3-D array. Can be thought of as an array of windows, where each window has 100 rows and 5 columns

for i in range(num_windows):
    segments[i] = dset.iloc[i * window_size:(i + 1) * window_size].values

# shuffle segmented data
np.random.seed(42)
np.random.shuffle(segments)

train_size = int(num_windows*ratio)

train_data = segments[:train_size]  # first 90% for train
test_data = segments[train_size:]   # remaining 10% for test

# must consolidate all rows -> turn 3-d (windows, rows, columns) to 2-d (rows, columns)
flat_train_data = train_data.reshape(train_data.shape[0]*train_data.shape[1], train_data.shape[2])

flat_test_data = test_data.reshape(test_data.shape[0]*test_data.shape[1], test_data.shape[2])


flat_train_data = pd.DataFrame(flat_train_data, columns=['Time (s)','Acceleration x (m/s^2)','Acceleration y (m/s^2)','Acceleration z (m/s^2)','Absolute acceleration (m/s^2)','Action','Placement'])
flat_test_data = pd.DataFrame(flat_test_data, columns=['Time (s)','Acceleration x (m/s^2)','Acceleration y (m/s^2)','Acceleration z (m/s^2)','Absolute acceleration (m/s^2)','Action','Placement'])


# then organize into train and test splits, ensuring X only sees accel values
X_train = flat_train_data.iloc[:,1:5]
y_train = flat_train_data.iloc[:,5]
X_test = flat_test_data.iloc[:,1:5]
y_test = flat_test_data.iloc[:,5]


# Perform feature scaling using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perform class resampling to address class imbalance
X_train_resampled, y_train_resampled = resample(X_train[y_train == 0], y_train[y_train == 0],
                                                replace=True, n_samples=sum(y_train == 1), random_state=42)
X_train_resampled = pd.DataFrame(X_train_resampled, columns=dset.columns[1:5])
X_train_resampled = pd.concat([X_train_resampled, pd.DataFrame(X_train[y_train == 1], columns=dset.columns[1:5])])
y_train_resampled = pd.concat([y_train_resampled, y_train[y_train == 1]])



# Create a pipeline for preprocessing and logistic regression              
pipeline = make_pipeline(
    PolynomialFeatures(degree=2),  # Add polynomial features with degree 2
    LogisticRegression(solver='liblinear', max_iter=10000)  # Logistic Regression model with solver='liblinear' and max_iter=1000
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

# obtain classification recall
recall = recall_score(y_test, y_pred)
print("Recall is: ", recall)

# create and plot learning curve
trsz, trsc, tesc = learning_curve(pipeline, X_train_resampled, y_train_resampled, train_sizes=np.linspace(0.01, 1.0, 10))
display = LearningCurveDisplay(train_sizes=trsz, train_scores=trsc, test_scores=tesc, score_name='Accuracy Score')
display.plot()
plt.show()

# plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()



# # Plot predicted classes
# plt.figure(figsize=(8, 6))
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.title('Predicted Classes (0: Walking, 1: Jumping)')
# plt.show()


# # New - plotting decision boundary
# pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=2))

# X_train_pca = pca_pipe.fit_transform(X_train)
# X_test_pca = pca_pipe.fit_transform(X_test)

# clf = LogisticRegression(max_iter=10000)

# clf.fit(X_train_pca, y_train)

# y_pred_pca = clf.predict(X_test_pca)

# acc = accuracy_score(y_test, y_pred_pca)
# print('Accuracy of PCA is: ',acc)

# disp = DecisionBoundaryDisplay.from_estimator(clf, X_train_pca, response_method="predict", xlabel='X1', ylabel='X2', alpha=0.5)

# disp.ax_.scatter(X_train_pca[:,0], X_train_pca[:,1], c=y_train, cmap='coolwarm')

# plt.show()