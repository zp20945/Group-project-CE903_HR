import pandas as pd

# Loading the two datasets
df1 = pd.read_csv(r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Preprocessed_more_3\Time_Frequency_Features.csv")
df2 = pd.read_csv(r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Preprocessed_more_3\PoincarePlots\All_Participants_PoincareMetrics.csv")

# Standardizing column names
df2.rename(columns={'Stimulus': 'SourceStimuliName'}, inplace=True)

# Merging the dataframes on 'Participant' and 'Stimulus'
merged_df = pd.merge(df1, df2, on=['Participant', 'SourceStimuliName'], how='inner')

# Saving the merged file
output = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Preprocessed_more_3\merged_features.csv"
merged_df.to_csv(output, index=False)

print("Merged file saved as 'merged_features.csv'")


