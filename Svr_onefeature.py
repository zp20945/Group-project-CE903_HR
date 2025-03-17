import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import shutil

# Load the dataset
file_path = r"D:\uni\Group_proj\features_norm_med_audio_arousal.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Ensure column names are correctly stripped of any whitespace
df.columns = df.columns.str.strip()

# Identify unique participants
participants = df['Participant'].unique()

# Store results for evaluation metrics and predictions
results = []
metrics_results = []

# Define raw output file paths (in the main folder)
parent_folder = r"D:\uni\Group_proj\SVR_Results"  # Parent folder to store all participant data

# Delete the parent folder if it exists (to avoid appending to old data)
if os.path.exists(parent_folder):
    print(f"Warning: {parent_folder} exists. Deleting the old folder.")
    shutil.rmtree(parent_folder)

# Create the parent folder again after deletion
os.makedirs(parent_folder)

# Define CSV paths for all predictions and metrics
output_file_predictions = os.path.join(parent_folder, "svr_predictions.csv")
output_file_metrics = os.path.join(parent_folder, "svr_metrics.csv")

# Delete the output files if they exist (to avoid appending to old data)
if os.path.exists(output_file_predictions):
    os.remove(output_file_predictions)

if os.path.exists(output_file_metrics):
    os.remove(output_file_metrics)

# Loop over participants
for participant in participants:
    # Split the data into training and testing sets
    train_data = df[df['Participant'] != participant]
    test_data = df[df['Participant'] == participant]
    
    # Define target
    y_train = train_data['Arousal']
    y_test = test_data['Arousal']
    
    # List of features to iterate over
    features = ['SDNN', 'RMSSD']

    # Initialize metrics storage for this participant
    metrics_for_participant = []

    for feature in features:
        # Define feature for training
        X_train = train_data[[feature]]
        X_test = test_data[[feature]]
        
        # Create an SVR model with RBF kernel (non-linear model)
        model = SVR(kernel='rbf')
        
        # Train the SVR model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        corr, p_value = pearsonr(y_test, y_pred)
        
        # Store metrics for each feature in the participant's list
        metrics_for_participant.append({
            'Participant': participant,
            'Feature': feature,
            'MSE': mse,
            'R2': r2,
            'MAE': mae,
            'Correlation': corr,
            'P_value': p_value
        })
        
        # Store predictions for each stimulus
        for stimulus, pred, actual in zip(test_data['Stimulus'], y_pred, y_test):
            results.append([participant, stimulus, feature, pred, actual])
        
        # Create a folder for the participant inside the parent folder if it doesn't exist
        participant_folder = os.path.join(parent_folder, f"Participant_{participant}")
        if not os.path.exists(participant_folder):
            os.makedirs(participant_folder)
        
        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test[feature], y_test, color='blue', label=f'Actual data ({feature})', alpha=0.6)
        plt.plot(X_test[feature], y_pred, color='red', label=f'SVR Fit ({feature})', linewidth=2)
        
        # Adding labels and title
        plt.title(f'SVR Model Fit (RBF Kernel) for Participant {participant} - Feature: {feature}')
        plt.xlabel(f'{feature}')
        plt.ylabel('Arousal')
        plt.legend()
        
        # Save plot as an image in the participant's folder
        plot_image_path = os.path.join(participant_folder, f"svr_plot_{participant}_{feature}.png")
        plt.savefig(plot_image_path)
        plt.close()  # Close the plot to avoid overlapping in future iterations

    # After processing both features for a participant, append the metrics to the global list
    metrics_results.extend(metrics_for_participant)

    # Save the participant's metrics to a CSV inside their folder
    participant_metrics_df = pd.DataFrame(metrics_for_participant)
    participant_metrics_path = os.path.join(participant_folder, f"metrics_{participant}.csv")
    participant_metrics_df.to_csv(participant_metrics_path, index=False)

# After looping through all participants, save the results:
# Save predictions to CSV in the parent folder
output_df_predictions = pd.DataFrame(results, columns=["Participant", "Stimulus", "Feature", "Predicted", "Actual"])
output_df_predictions.to_csv(output_file_predictions, index=False)

# Save metrics to CSV in the parent folder
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_csv(output_file_metrics, index=False)

print(f"Results saved to {output_file_predictions}")
print(f"Metrics saved to {output_file_metrics}")
