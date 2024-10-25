import pandas as pd
from scipy.stats import chi2_contingency

# Load the earthquake dataset
file_path = '/mnt/data/earthquakes.csv'
df = pd.read_csv(file_path)

# Display the dataset structure
print(df.head())

# Create a contingency table for 'type' and 'alert'
contingency_table = pd.crosstab(df['type'], df['alert'])
print("Contingency Table:")
print(contingency_table)

# Perform the Chi-square test
stat, p, dof, expected = chi2_contingency(contingency_table)

# Print the results
print("\nChi-Square Test Results")
print("Statistic:", stat)
print("p-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)

# Interpret the p-value
alpha = 0.05
if p <= alpha:
    print("\nConclusion: Dependent (reject H0)")
else:
    print("\nConclusion: Independent (H0 holds true)")
