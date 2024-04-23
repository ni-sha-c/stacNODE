import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('../test_result/Time/epoch_times.csv')

# Compute the average elapsed time
average_elapsed_time = df['Elapsed Time (seconds)'].mean()

print(f'Average Elapsed Time: {average_elapsed_time:.6f} seconds')


# Average elapsed time to compute Jacobian: 0.024679 seconds
# For each epoch: 0.076779 seconds