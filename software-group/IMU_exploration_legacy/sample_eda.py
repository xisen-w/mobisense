import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('software-group/data-working/sample_imu_data.csv')

# Inspect the data
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Plot histograms for each sensor value
df.hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Sensor Values')
plt.show()

# Pairplot to see relationships between variables
sns.pairplot(df)
plt.suptitle('Pairplot of Sensor Values')
plt.show()

# Time series plot
plt.figure(figsize=(15, 5))
for column in df.columns[:-1]:  # Exclude timestamp
    plt.plot(df['timestamp'], df[column], label=column)
plt.legend()
plt.title('Time Series of Sensor Values')
plt.xlabel('Timestamp')
plt.ylabel('Sensor Values')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()