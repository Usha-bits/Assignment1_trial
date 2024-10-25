# Demonstrate One-hot encoding and Label encoding in Python
# 1. Importing the Libraries
import pandas as pd
## import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
 
# 2. Reading the file
df = pd.read_csv("D:/Usha/Prefect/Assignment1_trial/data/credit_card_fraud_dataset.csv")
print(df)

#3. Apply Label encoding to the field 'TransactionType'
# Import LabelEncoder
le = LabelEncoder()  
df['TransactionType']=le.fit_transform(df['TransactionType'])
print(df['TransactionType'])

# For Covid_Severity field
## le = LabelEncoder()  
## df['Covid_Severity']=le.fit_transform(df['Covid_SeverityDescription'])
## print(df['Covid_Severity'])

# Represent Gender using One-Hot Encoding
# importing one hot encoder 
print(df['TransactionType'].value_counts())
one_hot_encoded_data = pd.get_dummies(df, columns = ['TransactionType'])
#one_hot_encoded_data = one_hot_encoded_data.astype(int)
print(one_hot_encoded_data)

# One hot encoding for Covid_Severity field
## print(df['Covid_Severity'].value_counts())
## from sklearn.preprocessing import OneHotEncoder
## one_hot_encoded_data = pd.get_dummies(df, columns = ['Covid_Severity'])
## print(one_hot_encoded_data)

# For multiple columns
#one_hot_encoded_data = pd.get_dummies(df, columns = ['Covid_Severity', 'Gender'])
#print(one_hot_encoded_data)