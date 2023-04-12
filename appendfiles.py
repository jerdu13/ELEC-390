import numpy as np
import glob

# Read multiple CSV files and append them into one numpy array
# change the path to your relative path

path = 'jeremys-dataset\\*.csv'
csv_files = glob.glob(path)
combined_data = []
for file in csv_files:
    data = np.genfromtxt(file, delimiter=',', skip_header=1)
    combined_data.append(data)
combined_data = np.concatenate(combined_data, axis=0)

# Save combined data as a new CSV file
np.savetxt('jeremy-data-combined.csv', combined_data, delimiter=',')