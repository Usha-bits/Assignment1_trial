# from tkinter import TRUE
import pandas as pd
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
df = pd.read_csv("D:/Usha/Prefect/Assignment1_trial/data/earthquakes.csv")
print(df)

numeric_df = df.select_dtypes(include=['number']) 
# Correlation Matrix - Internally uses Pearson Correlation
cor = numeric_df.corr()

# Plotting Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cor, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()