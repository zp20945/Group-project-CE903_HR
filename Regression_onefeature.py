import pandas as pd
import numpy as np
import os
import shutil  # Import shutil to handle directory deletion
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = r"D:\uni\Group_proj\features_norm_med_audio_arousal.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Ensure column names are correctly stripped of any whitespace
df.columns = df.columns.str.strip()

# Identify unique participants
participants = df['Participant'].unique()

# Create a new folder for saving results (if it doesn't exist)
output_folder = r"D:\uni\Group_proj\LinearRegression_Results"
if os.path.exists(output_folder):
    print(f"Warning: {output_folder} exists. Deleting the old folder.")
    shutil.rmtree(output_folder)

os.makedirs(output_folder)

# Store results for predictions
results_predictions = []

# Store results for evaluation metrics
results_metrics = []

# List of features to check one by one
features = ['SDNN', 'RMSSD']

# Function to calculate correlation, p-value, and plot
def plot_correlation(x, y, feature_name, participant_name, participant_folder):
    # Calculate the correlation coefficient and p-value
    corr, p_val = pearsonr(x, y)

    # Create the correlation plot
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, label=f'Correlation: {corr:.2f}')
    plt.plot(x, sm.OLS(y, sm.add_constant(x)).fit().fittedvalues, color='red', label=f'Linear Fit (R²={r2_score(y, sm.OLS(y, sm.add_constant(x)).fit().fittedvalues):.2f})')
    plt.title(f'Participant: {participant_name} - Correlation: {feature_name} vs Arousal')
    plt.xlabel(feature_name)
    plt.ylabel('Arousal')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(participant_folder, f"correlation_{feature_name}_Participant_{participant_name}.png"))  # Save the graph
    plt.close()
    
    return corr, p_val

# Perform regression and store predictions and metrics for each participant
for participant in participants:
    print(f"Evaluating participant: {participant}")
    
    # Create a subfolder for each participant
    participant_folder = os.path.join(output_folder, f"Participant_{participant}")
    if not os.path.exists(participant_folder):
        os.makedirs(participant_folder)

    # Split the data into training and testing sets
    train_data = df[df['Participant'] != participant]
    test_data = df[df['Participant'] == participant]
    
    # Define target
    y_train = train_data['Arousal']
    y_test = test_data['Arousal']
    
    mse_sdnn = 0
    mse_rmssd = 0
    r2_sdnn = 0
    r2_rmssd = 0
    
    # First feature: SDNN
    X_train_sdnn = train_data[['SDNN']]
    X_test_sdnn = test_data[['SDNN']]
    
    model_sdnn = LinearRegression()  # Linear Regression model
    model_sdnn.fit(X_train_sdnn, y_train)  # Train the model
    y_pred_sdnn = model_sdnn.predict(X_test_sdnn)  # Predict with the model
    
    # Evaluate performance for SDNN
    mse_sdnn = mean_squared_error(y_test, y_pred_sdnn)
    r2_sdnn = r2_score(y_test, y_pred_sdnn)
    
    # Calculate correlation and p-value for SDNN
    corr_sdnn, p_val_sdnn = plot_correlation(test_data['SDNN'], y_test, 'SDNN', participant, participant_folder)
    
    # Second feature: RMSSD
    X_train_rmssd = train_data[['RMSSD']]
    X_test_rmssd = test_data[['RMSSD']]
    
    model_rmssd = LinearRegression()  # Linear Regression model
    model_rmssd.fit(X_train_rmssd, y_train)  # Train the model
    y_pred_rmssd = model_rmssd.predict(X_test_rmssd)  # Predict with the model
    
    # Evaluate performance for RMSSD
    mse_rmssd = mean_squared_error(y_test, y_pred_rmssd)
    r2_rmssd = r2_score(y_test, y_pred_rmssd)
    
    # Calculate correlation and p-value for RMSSD
    corr_rmssd, p_val_rmssd = plot_correlation(test_data['RMSSD'], y_test, 'RMSSD', participant, participant_folder)
    
    # Append predictions for this participant (SDNN and RMSSD predictions)
    for stimulus, pred_s, pred_r, actual in zip(test_data['Stimulus'], y_pred_sdnn, y_pred_rmssd, y_test):
        results_predictions.append([participant, stimulus, pred_s, pred_r, actual])
    
    # Append evaluation metrics for SDNN and RMSSD to the results
    results_metrics.append([participant, mse_sdnn, r2_sdnn, corr_sdnn, p_val_sdnn, mse_rmssd, r2_rmssd, corr_rmssd, p_val_rmssd])
    
    # Save individual metrics for each participant in their own CSV
    participant_metrics_df = pd.DataFrame([[participant, corr_sdnn, p_val_sdnn, r2_sdnn, corr_rmssd, p_val_rmssd, r2_rmssd]],
                                          columns=["Participant", "Correlation_SDNN", "P-Value_SDNN", "R²_SDNN", "Correlation_RMSSD", "P-Value_RMSSD", "R²_RMSSD"])
    participant_metrics_file = os.path.join(participant_folder, "metrics.csv")
    participant_metrics_df.to_csv(participant_metrics_file, index=False)
    
    print(f'Participant: {participant} | MSE_SDNN: {mse_sdnn:.4f} | R²_SDNN: {r2_sdnn:.4f}')
    print(f'Participant: {participant} | MSE_RMSSD: {mse_rmssd:.4f} | R²_RMSSD: {r2_rmssd:.4f}')
    print(f'Participant: {participant} | Correlation_SDNN: {corr_sdnn:.4f} | p-value_SDNN: {p_val_sdnn:.4f}')
    print(f'Participant: {participant} | Correlation_RMSSD: {corr_rmssd:.4f} | p-value_RMSSD: {p_val_rmssd:.4f}')

# Save prediction results to CSV with headers
output_file_predictions = os.path.join(output_folder, "lopo_results_predictions_lr.csv")
output_df_predictions = pd.DataFrame(results_predictions, columns=["Participant", "Stimulus", "Prediction_based_on_SDNN", "Prediction_based_on_RMSSD", "Actual"])
output_df_predictions.to_csv(output_file_predictions, index=False)

# Save metrics results to CSV with headers
output_file_metrics = os.path.join(output_folder, "lopo_results_metrics_lr.csv")
output_df_metrics = pd.DataFrame(results_metrics, columns=["Participant", "MSE_SDNN", "R²_SDNN", "Correlation_SDNN", "P-Value_SDNN", "MSE_RMSSD", "R²_RMSSD", "Correlation_RMSSD", "P-Value_RMSSD"])
output_df_metrics.to_csv(output_file_metrics, index=False)

print(f'Predictions saved to {output_file_predictions}')
print(f'Metrics saved to {output_file_metrics}')
