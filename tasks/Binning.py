# Discretization by Binning methods 
# Distance Binning and Frequency Binning

import pandas as pd
import numpy as np
 
df = pd.read_csv("D:/Usha/Prefect/Assignment1_trial/data/credit_card_fraud_dataset.csv")
print(df)

# 1. Distance binning
# Formula -> interval = (max-min) / Number of Bins
# Let us consider the 'Age' continuous value column for binning
min_value = df['Amount'].min()
max_value = df['Amount'].max()
print(min_value)
print(max_value)

# Suppose the bin size is 1000
# linspace returns evenly spaced numbers over a specified interval. 
# Returns num evenly spaced samples, calculated over the interval [start, stop].
bins = np.linspace(min_value,max_value,1000)
print(bins)

labels = ['A(0-1000)', 'A(>1000 - 2000)', 'A(>2000 - 3000)', 'A(>3000 - 4000)','A(>4000 - 5000)']

# We can use the cut() function to convert the numeric values of the column Age into the categorical values.
# We need to specify the bins and the labels.
df['bins_dist'] = pd.cut(df['Amount'], bins=bins, labels=labels, include_lowest=True)
print(df['bins_dist'])
# print(df['bins_dist'].values.tolist())

# If you want equal distribution of the items in your bins, use qcut . 
# If you want to define your own numeric bin ranges, then use cut

# 2. Frequency Binning
df['bin_freq'] = pd.qcut(df['Amount'], q=4, precision=1, labels=labels)
print(df['bin_freq'])
# print(df['bin_freq'].values.tolist())