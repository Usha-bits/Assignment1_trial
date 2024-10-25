# from tkinter import TRUE
import pandas as pd
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
df = pd.read_csv("D:/Usha/Prefect/Assignment1_trial/data/credit_card_fraud_dataset.csv")
print(df)

# Correlation Matrix - Internally uses Pearson Correlation
cor = df.corr()

# Plotting Heatmap
plt.figure(figsize = (10,6))
sns.heatmap(cor, annot=True)
plt.show()