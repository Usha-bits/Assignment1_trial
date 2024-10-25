# Compute the Pearson Correlation between the features 'Age' and 'Duration of Stay' in Covid dataset
import pandas as pd
from scipy.stats import pearsonr
from matplotlib import pyplot

# Import your data into Python
df = pd.read_csv("D:/Usha/Prefect/Assignment1_trial/data/credit_card_fraud_dataset.csv")
 
# Convert dataframe into series
list1 = df['TransactionID']
list2 = df['Amount']
 
# Apply the pearsonr()
corr, _ = pearsonr(list1, list2)
print('Pearson correlation: %.3f' % corr)

# Pearson correlation: 0.205 (Moderate Positive correlation)
# Interpretaton:
# As the age of the patient increases, days of stay in hospital also increases

# Draw a Plot of the relationship
# 'Age' on the X Axis and 'Days of Stay' on the Y axis
pyplot.scatter(list1, list2)
pyplot.show()