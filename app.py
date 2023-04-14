import sys
import csv
import joblib
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
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFileDialog, QPushButton, QDesktopWidget, QTableWidget, QTableWidgetItem, QSizePolicy, QGridLayout


class CsvTable(QTableWidget):
    def __init__(self, data, *args):
        super(CsvTable, self).__init__(*args)
        self.data = data
        self.setRowCount(len(data))
        self.setColumnCount(len(data.columns))
        self.setHorizontalHeaderLabels(data.columns)
        self.set_data()
        self.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.verticalHeader().setDefaultAlignment(Qt.AlignCenter)

    def set_data(self):
        for i, row in enumerate(self.data.values):
            for j, item in enumerate(row):
                cell = QTableWidgetItem(str(item))
                cell.setTextAlignment(Qt.AlignCenter)  # Center cell data
                self.setItem(i, j, cell)


class CsvFileUpload(QWidget):
    def __init__(self):
        super().__init__()
        self.initDisplay()

    def initDisplay(self):
        self.label = QLabel('Please upload a CSV file.', self)
        self.vert_layout = QVBoxLayout()

        self.button = QPushButton('Select CSV file.', self)
        self.button.clicked.connect(self.openFileDialog)

        self.vert_layout.addWidget(self.label, alignment=Qt.AlignCenter)
        self.vert_layout.addWidget(self.button, alignment=Qt.AlignCenter)
        self.vert_layout.addStretch(1)  # Add stretch back

        self.table_container = QVBoxLayout()  # Create a separate QVBoxLayout container for the table
        self.table_container.addStretch(1)
        self.vert_layout.addLayout(self.table_container)  # Add the table container to the main layout
        self.setLayout(self.vert_layout)
    def openFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        processed = self.processData(file_name)
        self.displayCsvTable(processed)
    def processData(self, file):
        if file:

            model = joblib.load('jump-walk_model.joblib') # dump pre trained model from classifier

            dset = pd.read_csv(file)
            sampling_rate = 100
            window_size = 5 * sampling_rate
            ratio = 0.9

            rows = dset.shape[0]
            num_windows = int(rows // window_size)

            segments = np.zeros((num_windows, window_size, dset.shape[1]))

            for i in range(num_windows):
                segments[i] = dset.iloc[i * window_size:(i + 1) * window_size].values

            np.random.seed(42)
            np.random.shuffle(segments)

            train_size = int(num_windows * ratio)
            test_data = segments[train_size:]

            flat_test_data = test_data.reshape(test_data.shape[0] * test_data.shape[1], test_data.shape[2])
            flat_test_data = pd.DataFrame(flat_test_data,
                                          columns=['Time (s)', 'Acceleration x (m/s^2)', 'Acceleration y (m/s^2)',
                                                   'Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)',
                                                   'Action',
                                                   'Placement'])

            X_test = flat_test_data.iloc[:, 1:5]
            y_test = flat_test_data.iloc[:, 5]

            # Perform feature scaling using MinMaxScaler
            scaler = MinMaxScaler()
            X_test = scaler.fit_transform(X_test)

            # Predict on the testing data
            y_pred = model.predict(X_test)

            titles = ['Window', 'Action']
            result_map = {0: 'Walking', 1: 'Jumping'}
            with open('results.csv', 'w', newline='') as outputfile:
                write = csv.writer(outputfile)
                write.writerow(titles)
                for i, value in enumerate(y_pred):
                    write.writerow([i, (result_map[y_pred[i]])])
            outputfile.close()
            return 'results.csv'
    def displayCsvTable(self, file):
        if file:
            data = pd.read_csv(file)
            csv_table = CsvTable(data)
            csv_table.setMinimumSize(400, 800)
            # Clear the table container
            while self.table_container.count():
                child = self.table_container.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            self.table_container.addWidget(csv_table, alignment=Qt.AlignCenter)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initDisplay()

    def initDisplay(self):
        self.setWindowTitle('ELEC 390 - Jumping/Walking Classifier')
        content = CsvFileUpload()
        self.setCentralWidget(content)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
