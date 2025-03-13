import pandas as pd

df = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Features_Time_Frequency_Comparission_Splitted_Whole\final_merged_output.csv"
print("Checking for NaN values in the dataset:")
print(df.isna().sum())  # Check missing values in each column

#print("\nChecking for zero standard deviation:")
#std_values = df.std()
#print(std_values[std_values == 0])  # Print columns with zero standard deviation
