import os
import glob

# Set directory path where the CSV files are located
directory_path = 'Louie\'s-Dataset/Walkinginhand-Dataset'

# Set name for output file
output_filename = 'combinedwalkinghand_csv.csv'

# Change working directory to directory_path
os.chdir(directory_path)

# Get list of CSV files in directory
file_extension = '*.csv'
all_csv_files = [i for i in glob.glob(file_extension)]

# Combine data from CSV files into one file
with open(output_filename, 'w') as outfile:
    for index, csv_file in enumerate(all_csv_files):
        with open(csv_file, 'r') as infile:
            if index != 0: # skip header for all files except the first one
                next(infile)
            outfile.write(infile.read())

print(f"CSV files have been appended into {output_filename}.")
