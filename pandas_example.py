# Importing the required libraries
import pandas as pd


# Load the Iris dataset into a pandas DataFrame
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(url, header=None, names=column_names)


df.to_csv("iris.csv", index=False)
df = pd.read_csv("iris.csv")

# Display the shape of the DataFrame
print("Shape of the DataFrame:", df.shape)

# Display the data types of each column
print("\nData types of each column:")
print(df.dtypes)

# Display the first 5 rows of the DataFrame
print("\nFirst 5 rows of the DataFrame:")
print(df.head())

# Display summary statistics of the DataFrame
print("\nSummary statistics of the DataFrame:")
print(df.describe())

# Analyze individual columns
# Mean of sepal_length
mean_sepal_length = df['sepal_length'].mean()
print(f"\nMean of sepal_length: {mean_sepal_length}")

# Median of sepal_width
median_sepal_width = df['sepal_width'].median()
print(f"Median of sepal_width: {median_sepal_width}")

# Standard deviation of petal_length
std_petal_length = df['petal_length'].std()
print(f"Standard deviation of petal_length: {std_petal_length}")

# Analyze columns in groups
# Group by 'class' and calculate mean for each group
grouped_mean = df.groupby('class').mean()
print("\nMean of each feature grouped by class:")
print(grouped_mean)

# Group by 'class' and calculate multiple statistics for each group
grouped_stats = df.groupby('class').agg(['mean', 'std', 'min', 'max'])
print("\nMultiple statistics of each feature grouped by class:")
print(grouped_stats)

# Additional functions
# Count the number of occurrences of each class
class_count = df['class'].value_counts()
print("\nCount of each class:")
print(class_count)

# Check for missing values in the DataFrame
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Drop duplicates if any
df.drop_duplicates(inplace=True)
print("\nDropped duplicates, new shape:", df.shape)
